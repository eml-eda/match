from abc import ABC,abstractmethod
import os
from pathlib import Path
import subprocess
from typing import List

import mako
from match.opt.generator import ScheduleGenerator
from match.target.memory_inst import MemoryInst
import tvm.relay
from tvm.relay.dataflow_pattern import match as is_pattern_matching
from match.partition.partitioning_pattern import PartitioningPattern
from tvm.relay.dataflow_pattern import CallPattern,AttrPattern,AltPattern
from match.utils import save_codegen_schedule,save_schedule_search_res
from functools import partial
import tvm 

import traceback as tb

HOST_MEM_SIZE = 8192

class PatternResult:
    """Class that stores all the information that may be relevant to cache a result of a node that we want to compile
    """
    def __init__(self,match_target_pattern,match_node,schedule=None,latency=0,energy=0):
        self.match_target_pattern=match_target_pattern
        self.match_node=match_node
        self.schedule=schedule
        self.latency=latency
        self.energy=energy

    def get_schedule(self):
        return self.schedule

    def set_schedule(self,schedule):
        self.schedule=schedule

    def get_latency(self):
        return self.latency

    def set_latency(self,latency):
        self.latency=latency

    def get_energy(self):
        return self.energy

    def set_energy(self,energy):
        self.energy=energy

    def __eq__(self,other):
        return self.match_target_pattern.name==other.match_target_pattern.name and self.match_node==other.match_node

def mock_func(pattern):
    return True

def get_lenght_pattern(pattern):
    if isinstance(pattern, CallPattern):
        return 1+get_lenght_pattern(pattern.args[0])
    elif isinstance(pattern,AttrPattern):
        return get_lenght_pattern(pattern.pattern)
    elif isinstance(pattern,AltPattern):
        return max(get_lenght_pattern(pattern.left),get_lenght_pattern(pattern.right))
    return 0

class MatchTargetPattern:
    """Support pattern class that includes information about the execution module
    """
    def __init__(self,exec_module,module_pattern,name:str="conv2d",original_name:str="conv2d",match_additional_checks=mock_func,idx=0):
        self.exec_module=exec_module
        self.name=name
        self.original_name=original_name
        self.pattern=module_pattern.pattern
        self.ordered_operation=module_pattern.ordered_operation
        self.additional_checks=module_pattern.additional_checks
        self.match_additional_checks=match_additional_checks
        self.idx=idx

    def set_match_additional_checks(self,match_additional_checks):
        self.match_additional_checks=match_additional_checks

    def set_idx(self,idx):
        self.idx=idx

class MatchTarget(ABC):
    """Class that represents a heterogeneous target, this implementation defines this class through the singleton pattern
    """
    def __init__(self,exec_modules,name:str="match",optimize_param:str="latency",**kwargs):
        if self.singleton_instantiated(**kwargs):
            return
        self.name=name
        # can choose between riscv_cpu, arm_cpu, micro, and more, look at tvm/python/tvm/target/target.py
        self.cpu_type="riscv_cpu"
        # enable USMP or not?
        self.static_mem_plan=True
        # which algorithm to use in case we use USMP, can be greedy etc.
        # hill_climb looks the best overall but you can play with it
        self.static_mem_plan_algorithm="hill_climb"
        self.match_patterns=[]
        self.exec_modules=[]
        self.exec_modules_dict=dict()
        self.disabled_exec_modules=[]
        self.load_file_to_ext_mem_fn = "load_file"
        self.load_to_ext_mem_fn = "memcpy_to_ext"
        self.free_external_mem = "free_ext_mem"
        self.allocate_ext_mem = "load_ext_mem"
        self.load_from_ext_mem_fn = "memcpy_from_ext"
        self.optimize_param="energy" if optimize_param=="energy" else "latency"
        self.tvm_runtime_include_path=os.path.dirname(__file__)+"/../libs/c/static/default/include/tvm_runtime.h"
        self.tvm_runtime_src_path=os.path.dirname(__file__)+"/../libs/c/static/default/src/tvm_runtime.c"
        self.crt_config_path=os.path.dirname(__file__)+"/../libs/c/static/default/include/crt_config.h"
        self.makefile_path=os.path.dirname(__file__)+"/../libs/c/static/default/Makefile"
        self.main_template_path=os.path.dirname(__file__)+"/../libs/c/mako/default/src/main.c"
        self.model_generative_apis_src_path=os.path.dirname(__file__)+"/../libs/c/mako/default/src/generative_model_apis.c"
        self.model_generative_apis_include_path=os.path.dirname(__file__)+"/../libs/c/mako/default/include/generative_model_apis.h"
        self.default_inputs_src_path=os.path.dirname(__file__)+"/../libs/c/mako/match/src/default_inputs.c"
        self.default_inputs_include_path=os.path.dirname(__file__)+"/../libs/c/mako/match/include/default_inputs.h"
        self.start_get_timestamp_api = "clock"
        self.end_get_timestamp_api = "clock"
        self.timestamp_type = "clock_t"
        self.timestamp_to_ms = "* CLOCKS_PER_SEC/1000"
        self.alloc_fn = "malloc"
        self.free_fn = "free"
        self.clean_funcs=[]
        self.init_funcs=[]
        self.include_list=[]
        self.input_macros=""
        self.__cached_pattern_results__=[]
        for exec_module in exec_modules:
            self.add_exec_module(exec_module)
            self.exec_modules_dict[exec_module.name]=exec_module

    def singleton_instantiated(self,**kwargs):
        prev_kwargs_ = dict() if not hasattr(self,"prev_kwargs") else self.prev_kwargs
        self.prev_kwargs=kwargs
        if prev_kwargs_==kwargs and hasattr(self,"match_patterns"):
            return True

    # we want a singleton for caching purposes
    def __new__(class_, *args, **kwargs):
        if not hasattr(class_,"_instance"):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def gen_libs_and_main(self,models,default_model,out_path):
        abs_out_path = str(Path(out_path).absolute())
        subprocess.getoutput(f"cp {self.tvm_runtime_include_path} {abs_out_path}/include/tvm_runtime.h")
        subprocess.getoutput(f"cp {self.tvm_runtime_src_path} {abs_out_path}/src/tvm_runtime.c")
        subprocess.getoutput(f"cp {self.crt_config_path} {abs_out_path}/include/crt_config.h")
        subprocess.getoutput(f"cp {self.makefile_path} {abs_out_path}/Makefile")
        models_ = models
        match_inputs, match_outputs = models_[default_model].get_match_inputs_and_outputs()
        templates_data = {
            "target":self,
            "match_inputs":match_inputs,
            "match_outputs":match_outputs,
            "default_model":default_model,
            "models":models_,
            "runtime":"",
            "golden_cpu_model":models_[default_model].golden_cpu_model,
            "benchmarking":models_[default_model].benchmark_model,
            "bench_iterations":int(models_[default_model].benchmark_model),
            "handle_out_fn":models_[default_model].handle_out_fn,
            "app":"match",
        }
        with open(abs_out_path+f"/include/{default_model}/default_inputs.h","w") as inp_file:
            inp_file.write(mako.template.Template(filename=self.default_inputs_include_path).render(**templates_data))
        with open(abs_out_path+f"/src/{default_model}/default_inputs.c","w") as inp_file:
            inp_file.write(mako.template.Template(filename=self.default_inputs_src_path).render(**templates_data))
        with open(abs_out_path+"/src/main.c","w") as main_file:
            main_file.write(mako.template.Template(filename=self.main_template_path).render(**templates_data))
        # with open(abs_out_path+"/src/match/generative_model_apis.c","w") as model_api_file:
            # model_api_file.write(mako.template.Template(filename=self.model_generative_apis_src_path).render(**templates_data))
        # with open(abs_out_path+"/include/match/generative_model_apis.h","w") as model_api_file:
            # model_api_file.write(mako.template.Template(filename=self.model_generative_apis_include_path).render(**templates_data))

    def get_match_pattern_from_pattern_name(self,pattern_name:str="conv2d"):
        for pt in self.match_patterns:
            if pt.name==pattern_name:
                return pt
        return None

    def get_layer_from_module(self,mod:tvm.ir.IRModule,exec_module_name:str="",pattern_name:str="conv2d"):
        """Function to retrieve the schedule with caching of a certain TVM module

        Args:
            mod (tvm.ir.IRModule): module to compile
            pattern_name (str, optional): Name of the pattern that partitioned this module. Defaults to "conv2d".

        Returns:
            Schedule
        """
        node=mod.body.op.body
        match_pt=self.get_match_pattern_from_pattern_name(pattern_name=f"{self.name}.{exec_module_name}.{pattern_name}")
        schedule_gen = ScheduleGenerator(node=node,args_list=mod.body.args,
                                         target=self,
                                         exec_module=match_pt.exec_module,
                                         pattern_name=match_pt.original_name,
                                         partitioned=True,
                                         pattern_inst=match_pt)
        schedule_gen.parse()
        match_node=schedule_gen.get_match_node()
        pt_res=PatternResult(match_pt,match_node)
        schedule,latency,energy=self.find_in_cached_list(pt_res)
        if schedule is None:
            try:
                schedule_gen.generate()
            except Exception as exc:
                raise Exception("No valid loop ordering found")
            schedule_gen.apply_constraints()
            schedule=schedule_gen.schedule
            latency=schedule_gen.latency
            energy=schedule_gen.energy
            pt_res.set_schedule(schedule)
            pt_res.set_latency(latency)
            pt_res.set_energy(energy)
            self.add_pt_res_to_cache(pt_res)
        save_codegen_schedule(node,schedule,latency,energy)
        return schedule,match_node,match_pt.exec_module,latency,energy

    def add_pt_res_to_cache(self,pt_res):
        self.__cached_pattern_results__.append(pt_res)

    def find_in_cached_list(self,pattern_result):
        for cached_pt_res in self.__cached_pattern_results__:
            if cached_pt_res==pattern_result:
                return cached_pt_res.schedule,cached_pt_res.latency,cached_pt_res.energy
        return None,None,None

    def add_exec_modules(self,exec_modules):
        for exec_module in exec_modules:
            self.add_exec_module(exec_module)

    def evaluate_pattern(self,node,match_pt):
        """Search for the schedule of a node and its expected latency and energy performances

        Args:
            node (tvm.relay.Call): Node that can be partitioned
            match_pt (MatchTargetPattern): Pattern used to partition the node

        Returns:
            Number,Number: latency and energy consumption results of the node with the given pattern
        """
        schedule_gen = ScheduleGenerator(node=node,args_list=[],
                                         target=self,
                                         exec_module=match_pt.exec_module,
                                         pattern_name=match_pt.original_name,
                                         partitioned=False,
                                         pattern_inst=match_pt)
        schedule_gen.parse()
        match_node=schedule_gen.get_match_node()
        pt_res=PatternResult(match_pt,match_node)
        schedule,latency,energy=self.find_in_cached_list(pt_res)
        if schedule is not None:
            return latency,energy
        else:
            try:
                schedule_gen.generate()
            except Exception as exc:
                raise Exception(f"[TARGET]: No valid loop ordering found, {exc}")
            schedule_gen.apply_constraints()
            schedule=schedule_gen.schedule
            latency=schedule_gen.latency
            energy=schedule_gen.energy
            pt_res.set_schedule(schedule)
            pt_res.set_latency(latency)
            pt_res.set_energy(energy)
            save_schedule_search_res(match_pt.name,latency,energy,schedule,match_pt.pattern().partition(node))
            self.add_pt_res_to_cache(pt_res)
            return latency,energy

    def is_better_result(self,old_latency,old_energy,new_latency,new_energy):
        """Evaluate if the new results are better than the old ones based on a preference

        Args:
            old_latency (Number)
            old_energy (Number)
            new_latency (Number)
            new_energy (Number)

        Returns:
            bool: New etter than old?
        """
        if self.optimize_param=="latency":
            return new_latency<old_latency or (new_latency==old_latency and new_energy<old_energy) 
        elif self.optimize_param=="energy":
            return new_latency<old_latency or (new_latency==old_latency and new_energy<old_energy)
        else:
            return new_latency<old_latency or (new_latency==old_latency and new_energy<old_energy) 

    def match_additional_checks_(self,node,match_pt:MatchTargetPattern=None):
        """Function used to evaluate if the node is fully supported by the pattern, and if so if this is the best pattern wholly supporting
        this node

        Args:
            node (tvm.relay.Call): node to partition
            match_pt (MatchTargetPattern): pattern to evaluate

        Returns:
            bool: is this the best pattern supporting this node?
        """
        print(f"-------------------\n[PATTERN MATCHER] Node matched pattern {match_pt.name}, checking additional conditions")
        # is pattern fully supported?
        # node_add_checks = tvm.relay.transform.InferType()(tvm.ir.IRModule().from_expr(match_pt.pattern().partition(node)))["main"].body.op.body
        if not isinstance(node, tvm.relay.Call):
            print(f"[PATTERN MATCHER] Node is not a call, weird behaviour")
        node_add_checks = node
        if match_pt.additional_checks(node_add_checks):
            # if supported get latency and energy of pattern
            try:
                latency,energy=self.evaluate_pattern(node,match_pt)
                print(f"[PATTERN MATCHER] Node is supported by {match_pt.name} with expected latency {latency} and expected energy {energy}")
            except Exception as exc:
                print(f"[PATTERN MATCHER] Node failed to be evaluated with pattern {match_pt.name}")
                return False
            # check all the patterns that are after me
            for other_pt in self.match_patterns[match_pt.idx+1:]:
                if other_pt.exec_module.name in self.disabled_exec_modules:
                    continue
                # if pattern is fully matching get results
                # node_add_checks = tvm.relay.transform.InferType()(tvm.ir.IRModule().from_expr(other_pt.pattern().partition(node)))["main"].body.op.body
                node_add_checks = node
                if is_pattern_matching(other_pt.pattern(),node):
                    print(f"[PATTERN MATCHER] Node matched pattern {match_pt.name}, checking additional conditions")
                    if other_pt.additional_checks(node_add_checks):
                        try:
                            other_pt_latency,other_pt_energy=self.evaluate_pattern(node,other_pt)
                            print(f"[PATTERN MATCHER] Node is also supported by {other_pt.name} with expected latency {other_pt_latency} and expected energy {other_pt_energy}\n")
                        except Exception as exc:
                            print(f"[PATTERN MATCHER] Node failed to be evaluated with pattern {other_pt.name}")
                            continue
                        # if the result gathered by this other matching pattern is better break all
                        # this is due to the fact that this pattern will be matched later and finally
                        # the best pattern will return True
                        if self.is_better_result(latency,energy,other_pt_latency,other_pt_energy):
                            print(f"[PATTERN MATCHER] Pattern {other_pt.name} is expected to be better for this node than {match_pt.name}, refusing current pattern")
                            return False
                        else:
                            print(f"[PATTERN MATCHER] Pattern {match_pt.name} is expected to be better for this node than {other_pt.name}, checking remaining patterns")
                    else:
                        print(f"[PATTERN MATCHER] Matched pattern {other_pt.name} didnt satisfy the additional conditions")
            # best fully supported pattern for these set of nodes
            return True
        else:
            print(f"[PATTERN MATCHER] Matched pattern {match_pt.name} didnt satisfy the additional conditions")
        return False
    
    def sort_match_patterns(self):
        """Sort the pattern list based on the number of operators present on the pattern itself
        """
        self.match_patterns=sorted(self.match_patterns,key=lambda m_pt:-get_lenght_pattern(m_pt.pattern()))
        for idx,m_pt in enumerate(self.match_patterns):
            m_pt.set_idx(idx)
        
    def add_exec_module(self,exec_module):
        """Add a single exec module to the target, and add its patterns

        Args:
            exec_module (ExecModule): unit to add to the target
        """
        for module_pt in exec_module.partitioning_patterns():
            match_pt=MatchTargetPattern(exec_module,module_pt,name=f"{self.name}.{exec_module.name}.{module_pt.name}",original_name=module_pt.name)
            match_additional_checks=partial(self.match_additional_checks_,match_pt=match_pt)
            match_pt.set_match_additional_checks(match_additional_checks)
            self.match_patterns.append(match_pt)
        self.exec_modules.append(exec_module)
        # sort again cause we added new patterns
        self.sort_match_patterns()

    def partitioning_patterns(self):
        """patterns of the whole target

        Returns:
            List[PartitioningPattern]: list of pattern supported by the target sorted
        """
        return [
            PartitioningPattern(name=m_pt.name,pattern=m_pt.pattern,
                                ordered_operation=m_pt.ordered_operation,additional_checks=m_pt.match_additional_checks)
            for m_pt in self.match_patterns
            if m_pt.exec_module.name not in self.disabled_exec_modules
        ]

    @property
    def soc_memory_bytes(self):
        for mem in self.host_memories()[::-1]:
            if not mem.external:
                return mem.k_bytes*1024
        return 0
    
    @property
    def host_memory(self):
        for mem in self.host_memories()[::-1]:
            if not mem.external:
                return mem.name
        return ""
    
    def host_memories(self):
        return [
            # from lower level to higher level memories
            MemoryInst(name="HOST_MEM",k_bytes=HOST_MEM_SIZE),
        ]
    
    def memory_hierarchy_for_pt(self, exec_module, pattern_name):
        host_memories = self.host_memories()
        module_memories = exec_module.module_memories()
        module_memories = exec_module.update_memories_for_pt(module_memories, pattern_name=pattern_name)
        return {
            "output": [mem for mem in module_memories if "output" in mem.tensor_types]+host_memories,
            "var": [mem for mem in module_memories if "var" in mem.tensor_types]+host_memories,
            "const": [mem for mem in module_memories if "const" in mem.tensor_types]+host_memories,
            "intermediate": [mem for mem in module_memories if "intermediate" in mem.tensor_types]+host_memories,
        }

    def memory_list_for_pt(self, exec_module, pattern_name):
        host_memories = self.host_memories()
        module_memories = exec_module.module_memories()
        module_memories = exec_module.update_memories_for_pt(module_memories, pattern_name=pattern_name)
        return module_memories+host_memories

    def adjust_network(self,opts):
        return []

    def transform_after_partitioning(self,opts):
        pipeline=[]
        for exec_module in self.exec_modules:
            pipeline+=exec_module.adjust_network(opts=opts)
        pipeline+=self.adjust_network(opts=opts)
        return pipeline
    
    def network_transformations(self,opts):
        return []
    
    def transform_before_partitioning(self,opts):
        pipeline=self.network_transformations(opts=opts)
        for exec_module in self.exec_modules:
            pipeline+=exec_module.network_transformations(opts=opts)
        return pipeline
    
    def disable_exec_module(self,exec_module_name:str=""):
        self.disabled_exec_modules.append(exec_module_name)
    
class DefaultMatchTarget(MatchTarget):
    def __init__(self):
        super(DefaultMatchTarget,self).__init__([
        ],name="default")