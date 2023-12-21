
from abc import ABC,abstractmethod
from math import ceil
from typing import Any, Dict,List,Type
from match.target.cost_model import ZigZagMatchCostModel
from match.partition.partitioning_pattern import PartitioningPattern
from match.target.memory_inst import MemoryInst
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant
import tvm
import numpy as np
import sys

class MemoryApis:
    """All the memory APIs that are used by MATCH on the templates
    """
    def __init__(self):
        self.startup_memory_and_set_pattern="match_startup_memory_and_set_pattern"
        self.shutdown_mem="match_shutdown_mem"
        self.copy_out_curr_computation="match_copy_out_curr_computation"
        self.copy_out_prev_computation="match_copy_out_prev_computation"
        self.pointer_offset={
            "O":"match_pointer_offset_O",
            "W":"match_pointer_offset_W",
            "I":"match_pointer_offset_I",
            "X":"match_pointer_offset_X",
            "Y":"match_pointer_offset_Y",
        }
        self.mem_transfer={
            "O":"match_mem_transfer_O",
            "W":"match_mem_transfer_W",
            "I":"match_mem_transfer_I",
            "X":"match_mem_transfer_X",
            "Y":"match_mem_transfer_Y",
        }
        self.pattern_constants_loading="match_pattern_constants_loading"

class ComputationalApis:
    """All the APIs relating to the computational part that are used later by MATCH templates
    """
    def __init__(self):
        self.init_other_kernel_params="match_init_other_kernel_params"
        self.innermost_computation="match_innermost_computation"

class PlatformApis:
    """All the APIs for the management of the platform that are used by templates of MATCH
    """
    def __init__(self):
        self.init_platform="match_init_platform"

class SyncApis:
    """All the APIs for the synchronization that are used by templates of MATCH
    """
    def __init__(self):
        self.async_transfers="match_async_transfers"
        self.prev_computation="match_prev_computation"
        self.curr_computation="match_curr_computation"
        self.sync_multilevel_transfer="match_sync_multilevel_transfer"

class MatchTypes:
    """MACROS and types that can be used by MATCH
    """
    def __init__(self):
        self.mem_data_macro_and_type="unsigned int"
        self.kernel_struct="match_kernel"

class ExecModule(ABC):
    """Unit that will handle the compuation of a layer
    """
    def __init__(self,name:str="default_exec_module"):
        self.name=name
        self.FULL_DIM = sys.maxsize
        self.optimal_spatial_mapping = None
        self.platform_memories = None
        self.mem_apis= MemoryApis()
        self.sync_apis= SyncApis()
        self.comp_apis= ComputationalApis()
        self.platform_apis= PlatformApis()
        self.types= MatchTypes()
        self.include_list=[
            "match_dimensions.h",
            "match_kernel.h",
            "match_tile_indexes.h",
            "match_mem.h",
            "match_sync.h",
            "match_target_params.h",
        ]

    def partitioning_patterns(self):

        def conv2d_pattern():
            """Create pattern for conv2D with optional fused relu."""
            #breakpoint()
            return is_op("nn.conv2d")(
                    wildcard(), wildcard()
            )


        def fully_connected_pattern():
            """Create pattern for nn.dense with optional fused relu."""

            return is_op("nn.dense")(
                wildcard(), wildcard()
            )
        
        def element_wise_add_pattern():
            """Create pattern for element-wise-add with optional fused relu."""

            cast_a = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
            cast_b = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
            return is_op("add")(cast_a, cast_b)
        
        def no_checks(pattern):
            return True

        return [
            PartitioningPattern(name="default_conv2d",pattern=conv2d_pattern,additional_checks=no_checks),
            PartitioningPattern(name="default_dense",pattern=fully_connected_pattern,additional_checks=no_checks),
            PartitioningPattern(name="default_add",pattern=element_wise_add_pattern,additional_checks=no_checks)
        ]
    
    def network_transformations(self,opts):
        return []
    
    def hw_aware_template_params(self):
        raise dict()
    
    def memories_def(self,operands):
        """define the memory hierarchy of the unit by setting self.platform_memories

        Args:
            operands (List[Str]): list of operands
        """
        self.platform_memories = [
            # from lower level to higher level memories
            MemoryInst(name="l1_mem",k_bytes=32,operands=operands),
            MemoryInst(name="l2_mem",k_bytes=128,operands=operands,r_ports=1,w_ports=1,rw_ports=0),
        ]

    def limit_spatial_mapping_to(self,dim_size:int=1,optimal_spat:int=1):
        # find greater common denominator that is smaller than the optimal spatial mapping
        spatial_dim=dim_size
        if dim_size>optimal_spat:
            for div_val in [_+1 for _ in range(optimal_spat)][::-1]:
                if (dim_size%div_val)==0:
                    spatial_dim=div_val
                    break
        return spatial_dim

    def optimal_spatial_mapping_def(self, pattern_name: str = "conv_2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        """Define the optimal spatial mapping for the current node

        Args:
            pattern_name (str, optional): analyzed pattern. Defaults to "conv_2d".
            dim_sizes (Dict[str,int], optional): sizes of each dimension. Defaults to {}.
            layer_attrs (Dict, optional): attributes specific to the layer. Defaults to {}.
        """
        self.optimal_spatial_mapping = [ ("K",1), ("OY",1) ]

    def get_optimal_spat_size(self,optimal_spat:int=1,dim_size:int=1):
        if optimal_spat==self.FULL_DIM:
            return dim_size
        else:
            return optimal_spat

    def spatial_mapping(self,dim_sizes:Dict[str,int]={},pattern_name:str="conv_2d",operands:List[str]=["O","W","I"],pattern_operations:List[str]=["nn.conv2d"],layer_attrs:Dict={}):
        if self.optimal_spatial_mapping is None:
            self.optimal_spatial_mapping_def(pattern_name=pattern_name,dim_sizes=dim_sizes,layer_attrs=layer_attrs)
            if self.optimal_spatial_mapping is None:
                self.optimal_spatial_mapping = [ ("K",1), ("OY",1) ]
        def op_link_name(operand: str="O"):
            if operand=="O":
                return "O"
            elif operand in ["I","X"]:
                return "I1"
            else:
                return "I2"
        return {
            pattern_name:
                {
                    "core_allocation": 1,
                    "spatial_mapping":  {f"D{opt_idx+1}":(opt_sptmap[0],self.limit_spatial_mapping_to(\
                        dim_size=dim_sizes[opt_sptmap[0]],optimal_spat=self.get_optimal_spat_size(opt_sptmap[1],dim_sizes[opt_sptmap[0]])))\
                                        for opt_idx,opt_sptmap in enumerate(self.optimal_spatial_mapping)},
                    "memory_operand_links": {op:op_link_name(op) for op in operands},
                    "unordered_loops":["FX","FY"]+(["C"] if "dense" not in pattern_operations else []),
                }
        }
    
    def adjust_dimensions_and_precision(self,loop_dim_size:Dict[str,int]={},pr_loop_dim_size:Dict[str,int]={},
                                        operand_precision:Dict[str,int]={}, strides:List[int]=[1,1], pattern_name:str="conv2d"):
        return loop_dim_size,pr_loop_dim_size,operand_precision,operand_precision
    
    def cost_model(self):
        """Function that defines the used cost model to guide the schedule search

        Returns:
            Class: class itself(not an instance) of the used cost model
        """
        return ZigZagMatchCostModel
    
    def adjust_temporal_mapping(self,temporal_mapping:List=[],layer_data:Any=None):
        return temporal_mapping
    
    def template_data(self):
        return {}
    
    def weights_and_constants(self,layer_arguments:Dict={}):
        """define how the weights and constants of a layer must be saved in C on the generated code

        Args:
            layer_arguments (Dict, optional): Dict of the arguments(parameters) for the node. Defaults to {}.
        """
        def c_friendly_npvalue(arr):
            # params: arr is expected to be a numpy version of the value, it should be an array but it may be also just a single value
            if len(arr.shape)>0:
                # this is actually an array and not a single value
                arr=arr.reshape([arr.shape[0]]).astype(np.uint8)
                return f'{{{str(list(arr))[1:len(str(list(arr)))-1]}}}'
            else:
                return str(arr)
        def bytaze(value):
            return np.frombuffer(value.tobytes(),dtype='uint8')
        arguments=np.array([],dtype=np.uint8)
        single_constants=dict()
        for layer_arg_name,layer_arg_val in layer_arguments.items():
            if isinstance(layer_arg_val, tvm.relay.Constant):
                if len(layer_arg_val.data.shape)==0:
                    single_constants[layer_arg_name]=str(layer_arg_val.data)
                else:
                    constbytes=bytaze(layer_arg_val.data.numpy())
                    arguments=np.concatenate((arguments,constbytes))
        return {
            "value":c_friendly_npvalue(arguments),
            "len":arguments.shape[0],
            "shape":f"[{ceil(arguments.shape[0])}]",
            "single_costants":single_constants,
        }
    
    def additional_kernel_parameters(self):
        return dict()
    
    def types_def(self):
        return

    def match_types(self):
        self.types_def()
        return self.types

    def mem_apis_def(self):
        """Functions that set the memory related APIs of the unit
        """
        return

    def match_mem_apis(self):
        self.mem_apis_def()
        return self.mem_apis
    
    def sync_apis_def(self):
        """Functions that set the synchronization related APIs of the unit
        """
        return
    
    def match_sync_apis(self):
        self.sync_apis_def()
        return self.sync_apis
    
    def comp_apis_def(self):
        """Functions that set the computation related APIs of the unit
        """
        return
    
    def match_comp_apis(self):
        self.comp_apis_def()
        return self.comp_apis
    
    def platform_apis_def(self):
        """Functions that set the platform related APIs of the unit
        """
        return
    
    def match_platform_apis(self):
        self.platform_apis_def()
        return self.platform_apis

    def def_include_list(self):
        """Functions that sets the list of headers to include additionally in the template
        """
        return
    
    def match_include_list(self):
        self.def_include_list()
        return self.include_list

    def operand_memories(self,operands):
        return {
            operand:[mem.name for mem in self.platform_memories if operand in mem.operands][::-1]
            for operand in operands
        }
    
    