from abc import ABC,abstractmethod
from tvm.relay.dataflow_pattern import match as is_pattern_matching
from match.partition.partitioning_pattern import PartitioningPattern
from tvm.relay.dataflow_pattern import CallPattern,AttrPattern,AltPattern
from match.codegen.temporal_mapping_generator import TemporalMappingGenerator
from match.utils import save_codegen_schedule,save_schedule_search_res
from functools import partial
import tvm 

class PatternResult:
    """Class that stores all the information that may be relevant to cache a result of a node that we want to compile
    """
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
        self.match_patterns=[]
        self.exec_modules=[]
        self.exec_modules_dict=dict()
        self.disabled_exec_modules=[]
        self.optimize_param="energy" if optimize_param=="energy" else "latency"
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

    def get_match_pattern_from_pattern_name(self,pattern_name:str="conv2d"):
        for pt in self.match_patterns:
            if pt.name==pattern_name:
                return pt
        return None

    def get_layer_from_module(self,mod:tvm.ir.IRModule,exec_module_name:str="",pattern_name:str="conv2d"):
        """Function to retrieve the temporal mapping with caching of a certain TVM module

        Args:
            mod (tvm.ir.IRModule): module to compile
            pattern_name (str, optional): Name of the pattern that partitioned this module. Defaults to "conv2d".

        Returns:
            Temporal mapping as a list, the data of the fused layer and the module used to compile it
        """
        node=mod.body.op.body
        match_pt=self.get_match_pattern_from_pattern_name(pattern_name=f"{self.name}.{exec_module_name}.{pattern_name}")
        tmapgen = TemporalMappingGenerator(node=node,args_list=mod.body.args,exec_module=match_pt.exec_module,pattern_name=match_pt.original_name,partitioned=True,pattern_inst=match_pt)
        tmapgen.generate_workload()
        tmapgen.set_exec_module_for_layer()
        layer_data=tmapgen.get_layer_data()
        spatial_mapping=tmapgen.get_spatial_mapping()
        pt_res=PatternResult(match_pt,layer_data)
        temporal_mapping,latency,energy=self.find_in_cached_list(pt_res)
        if temporal_mapping is None:
            try:
                tmapgen.generate_temporal_mapping()
            except Exception as exc:
                raise Exception("No valid loop ordering found")
            tmapgen.constraint_temporal_mapping()
            temporal_mapping=tmapgen.get_temporal_mapping()
            latency=tmapgen.get_latency()
            energy=tmapgen.get_energy()
            pt_res.set_temporal_mapping(temporal_mapping)
            pt_res.set_latency(latency)
            pt_res.set_energy(energy)
            self.add_pt_res_to_cache(pt_res)
        save_codegen_schedule(node,temporal_mapping,spatial_mapping)
        return temporal_mapping,layer_data,match_pt.exec_module,latency,energy

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
        """Search for the temporal mapping of a fused layer and its expected latency and energy performances

        Args:
            node (tvm.relay.Call): Node that can be partitioned
            match_pt (MatchTargetPattern): Pattern used to partition the node

        Returns:
            Number,Number: latency and energy consumption results of the node with the given pattern
        """
        tmapgen = TemporalMappingGenerator(node=node,args_list=[],exec_module=match_pt.exec_module,pattern_name=match_pt.original_name,partitioned=False,pattern_inst=match_pt)
        tmapgen.generate_workload()
        tmapgen.set_exec_module_for_layer()
        layer_data=tmapgen.get_layer_data()
        pt_res=PatternResult(match_pt,layer_data)
        temporal_mapping,latency,energy=self.find_in_cached_list(pt_res)
        if temporal_mapping is not None:
            return latency,energy
        else:
            try:
                tmapgen.generate_temporal_mapping()
            except Exception as exc:
                raise Exception("No valid loop ordering found")
            tmapgen.constraint_temporal_mapping()
            temporal_mapping=tmapgen.get_temporal_mapping()
            latency=tmapgen.get_latency()
            energy=tmapgen.get_energy()
            pt_res.set_temporal_mapping(temporal_mapping)
            pt_res.set_latency(latency)
            pt_res.set_energy(energy)
            save_schedule_search_res(match_pt.name,latency,energy,temporal_mapping,match_pt.pattern().partition(node))
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
        # is pattern fully supported?
        if match_pt.additional_checks(node):
            # if supported get latency and energy of pattern
            try:
                latency,energy=self.evaluate_pattern(node,match_pt)
                print(f"\nNode is supported by {match_pt.name} with expected latency {latency} and expected energy {energy}\n")
            except Exception as exc:
                return False
            # check all the patterns that are after me
            for other_pt in self.match_patterns[match_pt.idx+1:]:
                if other_pt.exec_module.name in self.disabled_exec_modules:
                    continue
                # if pattern is fully matching get results
                if is_pattern_matching(other_pt.pattern(),node) and other_pt.additional_checks(node):
                    try:
                        other_pt_latency,other_pt_energy=self.evaluate_pattern(node,other_pt)
                        print(f"\nNode is also supported by {other_pt.name} with expected latency {other_pt_latency} and expected energy {other_pt_energy}\n")
                    except Exception as exc:
                        continue
                    # if the result gathered by this other matching pattern is better break all
                    # this is due to the fact that this pattern will be matched later and finally
                    # the best pattern will return True
                    if self.is_better_result(latency,energy,other_pt_latency,other_pt_energy):
                        return False
            # best fully supported pattern for these set of nodes
            return True
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

    def adjust_network(self,opts):
        pipeline=[]
        for exec_module in self.exec_modules:
            pipeline+=exec_module.adjust_network(opts=opts)
        return pipeline
    
    def network_transformations(self,opts):
        pipeline=[]
        for exec_module in self.exec_modules:
            pipeline+=exec_module.network_transformations(opts=opts)
        return pipeline
    
    def disable_exec_module(self,exec_module_name:str=""):
        self.disabled_exec_modules.append(exec_module_name)
    
class DefaultMatchTarget(MatchTarget):
    def __init__(self):
        super(DefaultMatchTarget,self).__init__([
        ],name="default")