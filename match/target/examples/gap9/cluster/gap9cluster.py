from math import ceil, prod
from typing import Dict, List

import numpy as np
from match.target.examples.gap9.cluster.cost_model import Gap9ClusterCostModel
from match.target.examples.gap9.cluster.network_transformations import network_transformations as gap9network_transformations
from match.target.examples.gap9.cluster.network_transformations import adjust_network as gap_adjust_net
from match.target.examples.gap9.cluster.partitioning_patterns import partitioning_patterns as gap9partitioning_patterns
from match.target.exec_module import ExecModule, PlatformApis, MemoryApis, SyncApis, ComputationalApis, MatchTypes
from match.target.memory_inst import MemoryInst
import os
import tvm

class Gap9Cluster(ExecModule):
    def __init__(self,**kwargs):
        super(Gap9Cluster, self).__init__(name="cluster",
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include",
                                          **kwargs)
        self.L1_SIZE=90 if "l1_size" not in kwargs else kwargs["l1_size"]

    def zigzag_optimal_spatial_mapping_def(self, match_node=None, pattern_name = "conv_2d"):
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
        fx_fy = prod(match_node.ops["conv2d"].kernel_size)
        dw = match_node.ops["conv2d"].depthwise
        if pattern_name in conv2d_patterns and fx_fy==1:
            return [
                ("OY",8),("OX",2),("K",4)
            ]
        elif pattern_name in conv2d_patterns and fx_fy<4 and dw:
            return [
                ("K",8),("OX",8),("OY",self.FULL_DIM)
            ]
        elif pattern_name in conv2d_patterns and dw:
            return [
                ("K",8),("OX",8),("OY",self.FULL_DIM),
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
    
    def specific_pattern_def(self, match_node=None, pattern_name = "conv_2d"):
        conv2d_patterns=[
            "conv2d_bnorm_requant",
            "conv2d_bias_add_requant",
            "conv2d_bias_add",
        ]
        dense_patterns=[
            "dense_bnorm_requant",
            "dense_bias_add_requant",
        ]
        fx_fy = prod(match_node.ops["conv2d"].kernel_size)
        dw = match_node.ops["conv2d"].depthwise
        if pattern_name in conv2d_patterns and fx_fy==1:
            return "pointwise_conv2d"
        elif pattern_name in conv2d_patterns and fx_fy<4 and dw:
            return "depthwise_conv2d_less_4"
        elif pattern_name in conv2d_patterns and dw:
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

        def buffers_for_l1_mem(match_node,pattern_name,specific_pattern):
            buff_mem=0
            # buffer for the cores of the accelerator (weights dimensions)
            NUM_CORES = 8
            NO_IM2COL_PTS_LIST = ['add_requant','dense_out','dense_bnorm_requant','dense_bias_add_requant']
            NO_IM2COL_SPEC_PTS_LIST = ['pointwise_conv2d']
            IS_DW = "depthwise_conv2d" in specific_pattern
            conv = match_node.ops["conv2d"] if "conv2d" in match_node.ops else None
            inp_tensors_names = [n for n in match_node.var_tensors.keys()]
            out_tensors_names = [n for n in match_node.output_tensors.keys()]
            inp_dims = [d.size for d in match_node.var_tensors[inp_tensors_names[0]].dims]
            out_dims = [d.size for d in match_node.output_tensors[out_tensors_names[0]].dims]
            k_size = (1,1) if conv is None else conv.kernel_size
            pad = (0,0,0,0) if conv is None else conv.padding
            if IS_DW:
                buff_mem = k_size[0] * (inp_dims[2] + sum([pad[0],pad[2]])) + k_size[0]
            elif (pattern_name not in NO_IM2COL_PTS_LIST) and (specific_pattern not in NO_IM2COL_SPEC_PTS_LIST):
                buff_mem=2*inp_dims[1]*k_size[0]*k_size[1]
            # buff for each core
            buff_mem*=NUM_CORES
            # bias
            if pattern_name!="add_requant":
                buff_mem+=out_dims[1]*4
            if pattern_name not in ['conv2d_bias_add_requant','conv2d_bias_add','dense_bias_add_requant','add_requant','dense_out']:
                buff_mem+=out_dims[1]*4
            return buff_mem
        
        memories[0].buffer_for_layer_func=buffers_for_l1_mem
        return memories
    
    def partitioning_patterns(self):
        return gap9partitioning_patterns()

    def network_transformations(self,opts):
        return gap9network_transformations(opts=opts)

    def def_include_list(self,pattern_name):
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

    def zigzag_cost_model(self):
        return Gap9ClusterCostModel
    
    def adjust_network(self, opts):
        return gap_adjust_net(opts=opts)
    
    def weights_and_constants(self, match_node, pattern_name):
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
        conv2d = "conv2d" in match_node.ops_occurrences
        dense = "nn.dense" in match_node.ops_occurrences
        for (layer_arg_name,layer_arg_val) in match_node.const_tensors.items():
            if len(layer_arg_val.data.shape)==0:
                single_constants[layer_arg_name]=str(layer_arg_val.data)
            else:
                if layer_arg_name=="FunctionVar_0_1" and conv2d:
                    constbytes=bytaze(layer_arg_val.data.transpose((0,2,3,1)))
                elif layer_arg_name=="FunctionVar_0_1" and dense:
                    constbytes=bytaze(layer_arg_val.data.transpose((0,1)))
                else:
                    constbytes=bytaze(layer_arg_val.data)
                arguments=np.concatenate((arguments,constbytes))
        return {
            "value":c_friendly_npvalue(arguments),
            "len":arguments.shape[0],
            "shape":f"[{ceil(arguments.shape[0])}]",
            "single_costants":single_constants,
        }