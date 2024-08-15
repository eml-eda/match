from math import ceil
from typing import Dict, List

import numpy as np
from match.target.gap9.cluster.cost_model import Gap9ClusterCostModel
from match.target.gap9.cluster.network_transformations import network_transformations as gap9network_transformations
from match.target.gap9.cluster.network_transformations import adjust_network as gap_adjust_net
from match.target.gap9.cluster.partitioning_patterns import partitioning_patterns as gap9partitioning_patterns
from match.target.exec_module import ExecModule, PlatformApis, MemoryApis, SyncApis, ComputationalApis, MatchTypes
from match.target.memory_inst import MemoryInst
import os
import tvm

class Gap9Cluster(ExecModule):
    def __init__(self,**kwargs):
        super(Gap9Cluster, self).__init__(name="cluster",
                                          specific_patterns=[
                                              "pointwise_conv2d",
                                              "depthwise_conv2d_less_4",
                                              "depthwise_conv2d",
                                              "conv2d",
                                              "elemwise_add",
                                              "dense",
                                              "dense_out"
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include",
                                          **kwargs)
        self.L1_SIZE=90 if "l1_size" not in kwargs else kwargs["l1_size"]

    def optimal_spatial_mapping_def(self, pattern_name: str = "gap9cluster_conv2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        conv2d_patterns=[
            "conv2d_bnorm_requant",
            "conv2d_bias_add_requant",
            "conv2d_bias_add",
        ]
        dense_patterns=[
            "dense_bnorm_requant",
            "dense_bias_add_requant",
            "dense_out"
        ]
        if pattern_name in conv2d_patterns and (dim_sizes['FY']*dim_sizes['FX'])==1:
            return [
                ("OY",8),("OX",2),("K",4)
            ]
        elif pattern_name in conv2d_patterns and (dim_sizes['FY']*dim_sizes['FX'])<4 and layer_attrs["nn.conv2d_depthwise"]:
            return [
                ("K",8),("OX",4),("OY",self.FULL_DIM)
            ]
        elif pattern_name in conv2d_patterns and layer_attrs["nn.conv2d_depthwise"]:
            return [
                ("K",8),("OX",4),("OY",self.FULL_DIM)
            ]
        elif pattern_name in conv2d_patterns:
            return [
                ("OY",8),("OX",2),("K",4)
            ]
        elif pattern_name=='add_requant':
            return [
                ("OY",8),("OX",2)
            ]
        elif pattern_name in dense_patterns:
            # TODO: K 8 C 4
            return [
                ("K",8)
            ]
        else:
            # DEFAULT LIKE CONV2D
            return [
                ("OY",8),("OX",2),("K",4)
            ]
    
    def specific_pattern_def(self, pattern_name: str = "conv_2d", dim_sizes: Dict[str, int] = ..., layer_attrs: Dict = ...):
        conv2d_patterns=[
            "conv2d_bnorm_requant",
            "conv2d_bias_add_requant",
            "conv2d_bias_add",
        ]
        dense_patterns=[
            "dense_bnorm_requant",
            "dense_bias_add_requant",
        ]
        if pattern_name in conv2d_patterns and (dim_sizes['FY']*dim_sizes['FX'])==1:
            return "pointwise_conv2d"
        elif pattern_name in conv2d_patterns and (dim_sizes['FY']*dim_sizes['FX'])<4 and layer_attrs["nn.conv2d_depthwise"]:
            return "depthwise_conv2d_less_4"
        elif pattern_name in conv2d_patterns and layer_attrs["nn.conv2d_depthwise"]:
            return "depthwise_conv2d"
        elif pattern_name in conv2d_patterns:
            return "conv2d"
        elif pattern_name=='add_requant':
            return "elemwise_add"
        elif pattern_name in dense_patterns:
            return "dense"
        elif pattern_name=="dense_out":
            return "dense_out"
        else:
            # DEFAULT LIKE CONV2D
            return "conv2d"

    def memories_def(self, pattern_name, operands):
        memories = [
            # from lower level to higher level memories
            # TEST: set now L1 to 9 kB just to force TILING 
            MemoryInst(name="l1_mem",k_bytes=self.L1_SIZE,operands=operands,double_buffering_support=True),
            MemoryInst(name="l2_mem",k_bytes=1408,operands=operands,r_ports=1,w_ports=1,rw_ports=0),
        ]
        if pattern_name=="add_requant":
            memories[0].double_buffering_support=False

        def buffers_for_l1_mem(layer_data,pattern_name,specific_pattern):
            breakpoint()
            buff_mem=0
            # buffer for the cores of the accelerator (weights dimensions)
            if (pattern_name not in ['add_requant','dense_out','dense_bnorm_requant','dense_bias_add_requant']) and (specific_pattern not in ['pointwise_conv2d']):
                buff_mem=2*layer_data.loop_dim_size['C']*layer_data.loop_dim_size['FY']*layer_data.loop_dim_size['FX']
            # buff for each core
            buff_mem*=8
            # bias
            if pattern_name!="add_requant":
                buff_mem+=layer_data.loop_dim_size['K']*4
            if pattern_name not in ['conv2d_bias_add_requant','conv2d_bias_add','dense_bias_add_requant','add_requant','dense_out']:
                buff_mem+=layer_data.loop_dim_size['K']*4
            return buff_mem
        
        memories[0].buffer_for_layer_func=buffers_for_l1_mem
        return memories
    
    def partitioning_patterns(self):
        return gap9partitioning_patterns()

    def network_transformations(self,opts):
        return gap9network_transformations(opts=opts)

    def def_include_list(self,patter_name):
        return ["cluster_mem.h","cluster_comp.h"]

    def mem_apis_def(self,mem_apis: MemoryApis=MemoryApis()):
        mem_apis.copy_out_curr_computation="cluster_copy_out_curr_computation"
        mem_apis.copy_out_prev_computation="cluster_copy_out_prev_computation"
        mem_apis.mem_transfer={
            "I":"cluster_mem_transfer_I",
            "X":"cluster_mem_transfer_X",
            "Y":"cluster_mem_transfer_Y",
            "W":"cluster_mem_transfer_W",
            "O":"cluster_mem_transfer_O",
        }
        mem_apis.pattern_constants_loading="cluster_pattern_constant_loading"
        mem_apis.shutdown_mem="cluster_shutdown_mem"
        mem_apis.startup_memory="cluster_startup_memory"
        return mem_apis

    def comp_apis_def(self,comp_apis: ComputationalApis=ComputationalApis()):
        comp_apis.innermost_computation="cluster_kernel_function_wrapper"
        comp_apis.init_other_kernel_params="cluster_init_other_kernel_params"
        comp_apis.specific_pattern=self.specific_pattern
        return comp_apis
    
    def platform_apis_def(self,platform_apis: PlatformApis=PlatformApis()):
        platform_apis.init_platform="cluster_init_platform"
        return platform_apis
    
    def sync_apis_def(self,sync_apis: SyncApis=SyncApis()):
        sync_apis.async_transfers="cluster_wait_any_transfer"
        sync_apis.curr_computation="cluster_wait_curr_computation"
        sync_apis.prev_computation="cluster_wait_prev_computation"
        return sync_apis

    def types_def(self,types: MatchTypes=MatchTypes()):
        types.kernel_struct="cluster_kernel"
        types.mem_data_macro_and_type="GAP_L2_DATA uint8_t"
        return types

    def cost_model(self):
        return Gap9ClusterCostModel
    
    def layout_per_operand_def(self, pattern_name, specific_pattern, operands):
        return {operand:"NHWC" for operand in operands}
    
    def adjust_network(self, opts):
        return gap_adjust_net(opts=opts)
    
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
                    if layer_arg_name=="nn.conv2d.param.0":
                        constbytes=bytaze(layer_arg_val.data.numpy().transpose((0,2,3,1)))
                    elif layer_arg_name=="nn.dense.param.0":
                        constbytes=bytaze(layer_arg_val.data.numpy().transpose((0,1)))
                    else:
                        constbytes=bytaze(layer_arg_val.data.numpy())
                    arguments=np.concatenate((arguments,constbytes))
        return {
            "value":c_friendly_npvalue(arguments),
            "len":arguments.shape[0],
            "shape":f"[{ceil(arguments.shape[0])}]",
            "single_costants":single_constants,
        }