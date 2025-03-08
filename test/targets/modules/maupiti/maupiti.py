import os
from typing import List
from match.target.exec_module import ComputationalApis, ExecModule
from patterns import maupiti_patterns
from transform import maupiti_adjust_network, maupiti_network_transformations
from match.target.memory_inst import MemoryInst
import tvm
import numpy as np


class MaupitiKernels(ExecModule):
    def __init__(self):
        super(MaupitiKernels, self).__init__(
            name="maupiti",
            specific_patterns=[
                "conv2d_bnorm_requant",
                "maxpool2d",
                "dense_out",
            ],
            src_path=os.path.dirname(__file__) + "/src",
            inc_path=os.path.dirname(__file__) + "/include",
        )
        self.KERNEL_MEM = 128
        self.module_options["MAUPITI_KERNEL_MEM"] = self.KERNEL_MEM
        self.tiling_active = False

    # def optimal_spatial_mapping_def(
    #     self,
    #     pattern_name: str = "conv2d",
    #     dim_sizes: Dict[str, int] = {},
    #     layer_attrs: Dict = {},
    # ):
    #     # k out channels
    #     # c in channels
    #     # oy out height
    #     # ox out width
    #     return [("K", 32), ("OY", 2)]

    def partitioning_patterns(self):
        return maupiti_patterns()

    def network_transformations(self, opts):
        return maupiti_network_transformations()

    def adjust_network(self, opts):
        return maupiti_adjust_network()

    def def_include_list(self, patter_name):
        return ["maupiti.h"]

    def comp_apis_def(self, comp_apis: ComputationalApis = ComputationalApis()):
        comp_apis.innermost_computation = "maupiti_kernels"
        comp_apis.compute_tile = "maupiti_kernels"
        return comp_apis

    def memories_def(self, pattern_name, operands):
        return [
            # from lower level to higher level memories
            MemoryInst(
                name="KERNEL_MEM",
                k_bytes=self.KERNEL_MEM,
                operands=operands,
                r_ports=1,
                w_ports=1,
                rw_ports=0,
            ),
        ]

    # If it's a dense
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
            "shape":f"[{np.ceil(arguments.shape[0])}]",
            "single_costants":single_constants,
        }