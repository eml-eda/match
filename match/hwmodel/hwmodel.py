
from abc import ABC,abstractmethod
from math import ceil
from typing import Any, Dict,List,Type
from match.hwmodel.memory_inst import MemoryInst
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant
import tvm
import numpy as np
import sys
#from match.codegen.layer_data import LayerData,TWO_INPUTS_WORKLOADS
from zigzag.classes.cost_model.specialized_latency_cost_model import SpecializedLatencyCostModel

TWO_INPUTS_WORKLOADS=(
    "element_wise_sum"
)

class HwModel(ABC):
    """
    Abstract base class for a temporal engine
    """
    def __init__(self):
        self.FULL_DIM = sys.maxsize
        self.optimal_spatial_mapping = None
        self.platform_memories = None

    def weights_def(self,layer_arguments):
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
        arguments=np.array()
        for layer_arg in layer_arguments:
            arguments=np.concatenate(arguments,bytaze(layer_arg))
        return {
            "value":c_friendly_npvalue(arguments),
            "shape":f"[{ceil(arguments.shape[0])}]"
        }
    
    def partitioning_patterns(self):

        def conv2d_pattern():
            """Create pattern for conv2D with optional fused relu."""
            #breakpoint()
            conv2d = is_op("nn.conv2d")(
                    wildcard(), wildcard()
            )
            return conv2d is not None


        def fully_connected_pattern():
            """Create pattern for nn.dense with optional fused relu."""

            fc = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            return fc is not None
        
        def element_wise_add_pattern():
            """Create pattern for element-wise-add with optional fused relu."""

            cast_a = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
            cast_b = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
            add = is_op("add")(cast_a, cast_b)
            return add is not None
        
        def no_checks(pattern):
            return True

        return [
            {   
                "name":"default_conv2d",
                "pattern_matcher":conv2d_pattern,
                "pattern_limitations":no_checks,
            },
            {
                "name":"default_dense",
                "pattern_matcher":fully_connected_pattern,
                "pattern_limitations":no_checks,
            },
            {
                "name":"default_add",
                "pattern_matcher":element_wise_add_pattern,
                "pattern_limitations":no_checks,
            },
        ]
    
    def network_transformations(self,opts):
        return []
    
    def hw_aware_template_params(self):
        raise dict()
    
    def memories_def(self,operands):
        self.platform_memories = [
            # from lower level to higher level memories
            MemoryInst(name="l1_mem",k_bytes=32,operands=operands),
            MemoryInst(name="l2_mem",k_bytes=128,operands=operands,r_ports=1,w_ports=1,rw_ports=0),
        ]

    def limit_spatial_mapping_to(self,dim_size:int=1,optimal_spat:int=1):
        spatial_dim=dim_size
        if dim_size>optimal_spat:
            for div_val in [_+1 for _ in range(optimal_spat)][::-1]:
                if (dim_size%div_val)==0:
                    spatial_dim=div_val
                    break
        return spatial_dim

    def optimal_spatial_mapping_def(self, workload_name: str = "conv_2d",dim_sizes:Dict[str,int]={}):
        self.optimal_spatial_mapping = [ ("K",1), ("OY",1) ]

    def get_optimal_spat_size(self,optimal_spat:int=1,dim_size:int=1):
        if optimal_spat==self.FULL_DIM:
            return dim_size
        else:
            return optimal_spat

    def spatial_mapping(self,dim_sizes:Dict[str,int]={},workload_name:str="conv_2d",operands:List[str]=["O","W","I"]):
        if self.optimal_spatial_mapping is None:
            self.optimal_spatial_mapping_def(workload_name=workload_name,dim_sizes=dim_sizes)
        return {
            workload_name:
                {
                    "core_allocation": 1,
                    "spatial_mapping":  {f"D{opt_idx+1}":(opt_sptmap[0],self.limit_spatial_mapping_to(\
                        dim_size=dim_sizes[opt_sptmap[0]],optimal_spat=self.get_optimal_spat_size(opt_sptmap[1],dim_sizes[opt_sptmap[0]])))\
                                        for opt_idx,opt_sptmap in enumerate(self.optimal_spatial_mapping)},
                    "memory_operand_links": {"O": "O", "I": "I1", "W": "I2"} if workload_name not in TWO_INPUTS_WORKLOADS else {'O': 'O', 'X': 'I1', 'Y': 'I2'},#{op:op for op in operands},
                    "fixed_loops":["FX","FY"]+(["C"] if workload_name!="dense" else []),
                }
        }
    
    def adjust_dimensions_and_precision(self,loop_dim_size:Dict[str,int]={},pr_loop_dim_size:Dict[str,int]={},
                                        operand_precision:Dict[str,int]={}, strides:List[int]=[1,1], workload_name:str="conv2d"):
        return loop_dim_size,pr_loop_dim_size,operand_precision,operand_precision
    
    def cost_model(self):
        return SpecializedLatencyCostModel([])
    
    def adjust_temporal_mapping(self,temporal_mapping:List=[],layer_data:Any=None):
        return temporal_mapping
    
    def template_data(self):
        return {}
    
    def weights_and_constants(self,layer_arguments:Dict={}):
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
        for layer_arg_val in layer_arguments.values():
            if isinstance(layer_arg_val, tvm.relay.Constant):
                constbytes=bytaze(layer_arg_val.data.numpy())
                arguments=np.concatenate((arguments,constbytes))
        return {
            "value":c_friendly_npvalue(arguments),
            "shape":f"[{ceil(arguments.shape[0])}]"
        }
    
    def additional_kernel_parameters(self):
        return dict()
    
    def apis_names(self):
        return {
            "memory":"match_memory",
            "kernel":"match_kernel",
        }
    
    def operand_memories(self,operands):
        return {
            operand:[mem.name for mem in self.platform_memories if operand in mem.operands][::-1]
            for operand in operands
        }
    
    