
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
    def __init__(self, pattern_name="conv2d",layout_per_operand={"O":"NCHW","I":"NCHW","W":"NCHW","X":"NCHW","Y":"NCHW"}):
        self.startup_memory="match_startup_memory"
        self.shutdown_mem="match_shutdown_mem"
        self.copy_out_curr_computation="match_copy_out_curr_computation"
        self.copy_out_prev_computation="match_copy_out_prev_computation"
        self.pointer_offset={
            "O":f"match_pointer_offset_{layout_per_operand['O']}_O",
            "W":f"match_pointer_offset_{layout_per_operand['W']}_W",
            "I":f"match_pointer_offset_{layout_per_operand['I']}_I",
            "X":f"match_pointer_offset_{layout_per_operand['X']}_X",
            "Y":f"match_pointer_offset_{layout_per_operand['Y']}_Y",
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
    def __init__(self, pattern_name="conv2d"):
        self.init_other_kernel_params="match_init_other_kernel_params"
        self.innermost_computation="match_innermost_computation"
        self.specific_pattern=pattern_name

class PlatformApis:
    """All the APIs for the management of the platform that are used by templates of MATCH
    """
    def __init__(self, pattern_name="conv2d"):
        self.init_platform="match_init_platform"
        self.set_task_id="match_task_id"

class SyncApis:
    """All the APIs for the synchronization that are used by templates of MATCH
    """
    def __init__(self, pattern_name="conv2d"):
        self.async_transfers="match_async_transfers"
        self.prev_computation="match_prev_computation"
        self.curr_computation="match_curr_computation"
        self.sync_multilevel_transfer="match_sync_multilevel_transfer"
        self.wait_input_transfers="match_wait_input_transfers"
        self.wait_output_transfers="match_wait_output_transfers"

class MatchTypes:
    """MACROS and types that can be used by MATCH
    """
    def __init__(self, pattern_name="conv2d"):
        self.mem_data_macro_and_type="unsigned int"
        self.kernel_struct="match_kernel"

class ExecModule(ABC):
    """Unit that will handle the compuation of a layer
    """
    def __init__(self,name:str="default_exec_module",
                 specific_patterns=[],
                 src_path:str="",
                 inc_path:str=""):
        self.name=name
        self.FULL_DIM = sys.maxsize
        self.optimal_spatial_mapping = None
        self.platform_memories = None
        self.default_include_list=[
            "match_dimensions.h",
            "match_kernel.h",
            "match_tile_indexes.h",
            "match_mem.h",
            "match_sync.h",
            "match_target_params.h",
        ]
        self.specific_patterns=specific_patterns
        self.specific_pattern=""
        self.layout_operand=dict()
        self.src_path=src_path
        self.inc_path=inc_path
        self.module_options=dict()

    def partitioning_patterns(self):

        def conv2d_pattern():
            """Create pattern for conv2D with optional fused relu."""
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

        return [
            PartitioningPattern(name="default_conv2d",pattern=conv2d_pattern,ordered_operation="nn.conv2d"),
            PartitioningPattern(name="default_dense",pattern=fully_connected_pattern,ordered_operation="nn.dense"),
            PartitioningPattern(name="default_add",pattern=element_wise_add_pattern,ordered_operation="add")
        ]
    
    def network_transformations(self,opts):
        return []
    
    def hw_aware_template_params(self):
        raise dict()

    def memories_def(self,pattern_name,operands):
        """define the memory hierarchy of the unit by setting self.platform_memories

        Args:
            operands (List[Str]): list of operands
        """
        return [
            # from lower level to higher level memories
            # TEST: set now L1 to 9 kB just to force TILING 
            MemoryInst(name="l1_mem",k_bytes=90,operands=operands,double_buffering_support=True),
            MemoryInst(name="l2_mem",k_bytes=1408,operands=operands,r_ports=1,w_ports=1,rw_ports=0),
        ]

    def get_all_memories_names(self):
        return [m.name for m in self.memories_def(pattern_name="conv2d",operands=["O","I","W"])]

    def match_memories(self,pattern_name,operands):
        self.platform_memories=self.memories_def(pattern_name,operands)

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
        return [ ("K",1), ("OY",1) ]

    def match_optimal_spatial_mapping(self, pattern_name: str = "conv_2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        self.optimal_spatial_mapping=self.optimal_spatial_mapping_def(pattern_name=pattern_name,dim_sizes=dim_sizes,layer_attrs=layer_attrs)

    def specific_pattern_def(self, pattern_name: str = "conv_2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        return pattern_name

    def match_specific_pattern(self, pattern_name: str = "conv_2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        self.specific_pattern=self.specific_pattern_def(pattern_name=pattern_name,dim_sizes=dim_sizes,layer_attrs=layer_attrs)

    def get_optimal_spat_size(self,optimal_spat:int=1,dim_size:int=1):
        if optimal_spat==self.FULL_DIM:
            return dim_size
        else:
            return optimal_spat

    def spatial_mapping(self,layer_data=None,pattern_name:str="conv_2d",pattern_inst=None):
        if self.optimal_spatial_mapping is None:
            self.optimal_spatial_mapping_def(pattern_name=pattern_name,dim_sizes=layer_data.layer_attrs["loop_sizes"],layer_attrs=layer_data.layer_attrs)
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
                        dim_size=layer_data.layer_attrs["loop_sizes"][opt_sptmap[0]],optimal_spat=self.get_optimal_spat_size(opt_sptmap[1],layer_data.layer_attrs["loop_sizes"][opt_sptmap[0]])))\
                                        for opt_idx,opt_sptmap in enumerate(self.optimal_spatial_mapping)},
                    "memory_operand_links": {op:op_link_name(op) for op in layer_data.operands},
                    "unordered_loops":["FX","FY","C"],#+(["C"] if "nn.dense" != pattern_inst.ordered_operation else []),
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
    
    def weights_and_constants(self,pattern_name,layer_data,layer_arguments:List=[]):
        """define how the weights and constants of a layer must be saved in C on the generated code

        Args:
            layer_arguments (List, optional): Dict of the arguments(parameters) for the node. Defaults to [].
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
        for (layer_arg_name,layer_arg_val) in layer_arguments:
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
    
    def additional_kernel_parameters(self,pattern_name):
        return dict()
    
    def layout_per_operand_def(self,pattern_name,specific_pattern,operands):
        return dict()

    def match_layout_operand(self,pattern_name,specific_pattern,operands):
        self.layout_operand=self.layout_per_operand_def(pattern_name=pattern_name,specific_pattern=specific_pattern,operands=operands)
        for operand_ in [op_ for op_ in ["O","I","W","X","Y"] if op_ not in self.layout_operand]:
            self.layout_operand[operand_]="NCHW"
        return self.layout_operand 

    def types_def(self,match_types: MatchTypes=MatchTypes()):
        return match_types

    def match_types(self, pattern_name):
        return self.types_def(MatchTypes(pattern_name=pattern_name))

    def mem_apis_def(self,memory_apis: MemoryApis=MemoryApis()):
        """Functions that set the memory related APIs of the unit
        """
        return memory_apis

    def match_mem_apis(self,pattern_name):
        return self.mem_apis_def(MemoryApis(pattern_name=pattern_name,layout_per_operand=self.layout_operand))
    
    def sync_apis_def(self,sync_apis: SyncApis=SyncApis()):
        """Functions that set the synchronization related APIs of the unit
        """
        return sync_apis
    
    def match_sync_apis(self,pattern_name):
        return self.sync_apis_def(SyncApis(pattern_name=pattern_name))
    
    def comp_apis_def(self,computational_apis: ComputationalApis=ComputationalApis()):
        """Functions that set the computation related APIs of the unit
        """
        return computational_apis
    
    def match_comp_apis(self,pattern_name):
        return self.comp_apis_def(ComputationalApis(pattern_name=pattern_name))
    
    def platform_apis_def(self,platform_apis: PlatformApis=PlatformApis()):
        """Functions that set the platform related APIs of the unit
        """
        return platform_apis
    
    def match_platform_apis(self,pattern_name):
        return self.platform_apis_def(PlatformApis(pattern_name=pattern_name))

    def def_include_list(self):
        """Functions that sets the list of headers to include additionally in the template
        """
        return []
    
    def match_include_list(self,pattern_name):
        ex_module_inc_list=self.def_include_list(pattern_name)
        return self.default_include_list+ex_module_inc_list

    def operand_memories(self,operands):
        return {
            operand:[mem.name for mem in self.platform_memories if operand in mem.operands][::-1]
            for operand in operands
        }
    
    def add_option_to_module(self,option,value):
        self.module_options[option]=value

    def adjust_network(self,opts):
        return []

    def generate_architecture_for(self,dse:str='zigzag',optimal_spatial_mapping:Any=None,platform_memories:Any=None,layer_data:Any=None):
        return None
