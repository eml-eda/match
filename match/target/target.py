from abc import ABC,abstractmethod
from tvm.relay.dataflow_pattern import match as is_pattern_matching
from match.partition.partitioning_pattern import PartitioningPattern
from tvm.relay.dataflow_pattern import CallPattern,AttrPattern
from match.codegen.temporal_mapping_generator import TemporalMappingGenerator
from functools import partial
import tvm 

class PatternResult:
    def __init__(self,match_target_pattern,layer_data,temporal_mapping=None,latency=0,energy=0):
        self.match_target_pattern=match_target_pattern
        self.layer_data=layer_data
        self.temporal_mapping=temporal_mapping
        self.latency=latency
        self.energy=energy

    def get_temporal_mapping(self):
        return self.temporal_mapping

    def set_temporal_mapping(self,temporal_mapping):
        self.temporal_mapping=temporal_mapping

    def get_latency(self):
        return self.latency

    def set_latency(self,latency):
        self.latency=latency

    def get_energy(self):
        return self.energy

    def set_energy(self,energy):
        self.energy=energy

    def __eq__(self,other):
        return self.match_target_pattern==other.match_target_pattern and self.layer_data==other.layer_data

def mock_func(pattern):
    return True

def get_lenght_pattern(pattern):
    if isinstance(pattern, CallPattern):
        return 1+get_lenght_pattern(pattern.args[0])
    elif isinstance(pattern,AttrPattern):
        return get_lenght_pattern(pattern.pattern)
    return 0

class MatchTargetPattern:
    def __init__(self,exec_module,module_pattern,name:str="conv2d",match_additional_checks=mock_func,idx=0):
        self.exec_module=exec_module
        self.name=name
        self.pattern=module_pattern.pattern
        self.additional_checks=module_pattern.additional_checks
        self.match_additional_checks=match_additional_checks
        self.idx=idx

    def set_match_additional_checks(self,match_additional_checks):
        self.match_additional_checks=match_additional_checks

    def set_idx(self,idx):
        self.idx=idx

class MatchTarget(ABC):
    def __init__(self,exec_modules,name:str="match",optimize_param:str="latency"):
        if self.singleton_instantiated():
            return
        self.name=name
        self.match_patterns=[]
        self.exec_modules=[]
        self.optimize_param="energy" if optimize_param=="energy" else "latency"
        self.__cached_pattern_results__=[]
        for exec_module in exec_modules:
            self.add_exec_module(exec_module)

    def singleton_instantiated(self):
        return hasattr(self,"match_patterns")

    # we want a singleton for caching purposes
    def __new__(class_, *args, **kwargs):
        if not hasattr(class_,"_instance"):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def get_match_pattern_from_pattern_name(self,pattern_name:str="conv2d"):
        for pt in self.match_patterns:
            if pt.name==pattern_name:
                return pt
        return None

    def get_layer_from_module(self,mod:tvm.ir.IRModule,pattern_name:str="conv2d"):
        node=mod.body.op.body
        match_pt=self.get_match_pattern_from_pattern_name(pattern_name=f"{self.name}.{pattern_name}")
        tmapgen = TemporalMappingGenerator(node=node,exec_module=match_pt.exec_module,pattern_name=match_pt.pattern)
        tmapgen.generate_workload()
        layer_data=tmapgen.get_layer_data()
        tmapgen.set_exec_module_for_layer()
        pt_res=PatternResult(match_pt,layer_data)
        temporal_mapping,latency,energy=self.find_in_cached_list(pt_res)
        return temporal_mapping,layer_data,match_pt.exec_module

    def add_pt_res_to_cache(self,pt_res):
        self.__cached_pattern_results__.append(pt_res)

    def find_in_cached_list(self,pattern_result):
        for cached_pt_res in self.__cached_pattern_results__:
            if cached_pt_res==pattern_result:
                return cached_pt_res.temporal_mapping,cached_pt_res.latency,cached_pt_res.energy
        return None,None,None

    def add_exec_modules(self,exec_modules):
        for exec_module in exec_modules:
            self.add_exec_module(exec_module)

    def evaluate_pattern(self,node,match_pt):
        tmapgen = TemporalMappingGenerator(node=node,exec_module=match_pt.exec_module,pattern_name=match_pt.pattern)
        tmapgen.generate_workload()
        layer_data=tmapgen.get_layer_data()
        pt_res=PatternResult(match_pt,layer_data)
        #breakpoint()
        temporal_mapping,latency,energy=self.find_in_cached_list(pt_res)
        if temporal_mapping is not None:
            return latency,energy
        else:
            tmapgen.generate_temporal_mapping()
            tmapgen.constraint_temporal_mapping()
            temporal_mapping=tmapgen.get_temporal_mapping()
            latency=tmapgen.get_latency()
            energy=tmapgen.get_energy()
            pt_res.set_temporal_mapping(temporal_mapping)
            pt_res.set_latency(latency)
            pt_res.set_energy(energy)
            self.add_pt_res_to_cache(pt_res)
            return latency,energy

    def is_better_result(self,old_latency,old_energy,new_latency,new_energy):
        if self.optimize_param=="latency":
            return new_latency<old_latency or (new_latency==old_latency and new_energy<old_energy) 
        elif self.optimize_param=="energy":
            return new_latency<old_latency or (new_latency==old_latency and new_energy<old_energy)
        else:
            return new_latency<old_latency or (new_latency==old_latency and new_energy<old_energy) 

    def match_additional_checks_(self,node,match_pt:MatchTargetPattern=None):
        #breakpoint()
        # is pattern fully supported?
        if match_pt.additional_checks(node):
            # if supported get latency and energy of pattern
            latency,energy=self.evaluate_pattern(node,match_pt)
            # check all the patterns that are after me
            for other_pt in self.match_patterns[match_pt.idx+1:]:
                # if pattern is fully matching get results
                if is_pattern_matching(other_pt.pattern(),node) and other_pt.additional_checks(node):
                    other_pt_latency,other_pt_energy=self.evaluate_pattern(node,other_pt)
                    # if the result gathered by this other matching pattern is better break all
                    # this is due to the fact that this pattern will be matched later and finally
                    # the best pattern will return True
                    if self.is_better_result(latency,energy,other_pt_latency,other_pt_energy):
                        return False
            # best fully supported pattern for these set of nodes
            return True
        return False
    
    def sort_match_patterns(self):
        self.match_patterns=sorted(self.match_patterns,key=lambda m_pt:-get_lenght_pattern(m_pt.pattern))
        for idx,m_pt in enumerate(self.match_patterns):
            m_pt.set_idx(idx)
        
    def add_exec_module(self,exec_module):
        for module_pt in exec_module.partitioning_patterns():
            match_pt=MatchTargetPattern(exec_module,module_pt,name=f"{self.name}.{module_pt.name}",)
            match_additional_checks=partial(self.match_additional_checks_,match_pt=match_pt)
            match_pt.set_match_additional_checks(match_additional_checks)
            self.match_patterns.append(match_pt)
        self.exec_modules.append(exec_module)
        # sort again cause we added new patterns
        self.sort_match_patterns()

    def partitioning_patterns(self):
        return [
            PartitioningPattern(m_pt.name,m_pt.pattern,m_pt.match_additional_checks)
            for m_pt in self.match_patterns
        ]

    def adjust_network(self,opts):
        pipeline=[]
        for exec_module in self.exec_modules:
            pipeline+=exec_module.network_transformations(opts=opts)
        return pipeline
    
    def network_transformations(self,opts):
        return []