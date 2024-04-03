from math import ceil
from typing import Dict, List
from match.target.gap9.ne16.cost_model import Gap9NE16CostModel
from match.target.gap9.ne16.network_transformations import network_transformations as gap9network_transformations
from match.target.gap9.ne16.partitioning_patterns import partitioning_patterns as gap9partitioning_patterns
from match.target.exec_module import ExecModule, PlatformApis, MemoryApis, SyncApis, ComputationalApis, MatchTypes
import os
import numpy as np
import numpy.typing as npt

import tvm

class Gap9NE16(ExecModule):
    def __init__(self):
        super(Gap9NE16, self).__init__(name="NE16",
                                          specific_patterns=[
                                              "conv2d",
                                              "depthwise_conv2d",
                                          ],
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include")

    def optimal_spatial_mapping_def(self, pattern_name: str = "gap9NE16_conv2d",dim_sizes:Dict[str,int]={},layer_attrs:Dict={}):
        if pattern_name=="conv2d_bnorm_requant" and (dim_sizes['FY']*dim_sizes['FX'])==1:
            return [
                ("OY",4),("OX",4),("K",4)
            ]
        elif pattern_name=="conv2d_bnorm_requant" and (dim_sizes['FY']*dim_sizes['FX'])==8:
            return [
                ("K",8),("OX",4),("OY",self.FULL_DIM)
            ]
        elif layer_attrs["nn.conv2d_depthwise"]:
            return [
                ("K",8),("OX",4),("OY",self.FULL_DIM)
            ]
        else:
            # DEFAULT LIKE CONV2D
            return [
                ("OY",8),("OX",2),("K",4)
            ]
    
    def specific_pattern_def(self, pattern_name: str = "conv_2d", dim_sizes: Dict[str, int] = ..., layer_attrs: Dict = ...):
        if pattern_name=="conv2d_bnorm_requant" and (dim_sizes['FY']*dim_sizes['FX'])==1:
            return "conv2d"
        elif pattern_name=="conv2d_bnorm_requant" and (dim_sizes['FY']*dim_sizes['FX'])==8:
            return "conv2d"
        elif layer_attrs["nn.conv2d_depthwise"]:
            return "depthwise_conv2d"
        else:
            # DEFAULT LIKE CONV2D
            return "conv2d"

    #def memories_def(self, pattern_name, operands):
    #    memories=super().memories_def(pattern_name=pattern_name,operands=operands)
    #    if pattern_name!="add_requant":
    #        memories[0].double_buffering_support=True
    #
    #    def buffers_for_l1_mem(layer_data,pattern_name):
    #        buff_mem=0
    #        # buffer for the cores of the accelerator (weights dimensions)
    #        if pattern_name!='add_requant' :
    #            buff_mem=2*layer_data.loop_dim_size['C']*layer_data.loop_dim_size['FY']*layer_data.loop_dim_size['FX']
    #        # buff for each core
    #        buff_mem*=8
    #        # bias
    #        if pattern_name!="add_requant":
    #            buff_mem+=layer_data.loop_dim_size['K']*4
    #        return buff_mem
    #    
    #    memories[0].buffer_for_layer_func=buffers_for_l1_mem
    #    return memories
    
    def partitioning_patterns(self):
        return gap9partitioning_patterns()

    def network_transformations(self,opts):
        return gap9network_transformations(opts=opts)

    def def_include_list(self,patter_name):
        return ["ne16_mem.h","ne16_comp.h"]

    def mem_apis_def(self,mem_apis: MemoryApis=MemoryApis()):
        mem_apis.copy_out_curr_computation="ne16_copy_out_curr_computation"
        mem_apis.mem_transfer={
            "I":"ne16_mem_transfer_I",
            "W":"ne16_mem_transfer_W",
            "O":"ne16_mem_transfer_O",
        }
        mem_apis.pattern_constants_loading="ne16_pattern_constant_loading"
        mem_apis.shutdown_mem="ne16_shutdown_mem"
        mem_apis.startup_memory="ne16_startup_memory"
        return mem_apis

    def comp_apis_def(self,comp_apis: ComputationalApis=ComputationalApis()):
        comp_apis.innermost_computation="ne16_kernel_function_wrapper"
        comp_apis.specific_pattern=self.specific_pattern
        return comp_apis
    
    def platform_apis_def(self,platform_apis: PlatformApis=PlatformApis()):
        platform_apis.init_platform="ne16_init_platform"
        platform_apis.set_task_id="ne16_set_task_id"
        return platform_apis
    
    def sync_apis_def(self,sync_apis: SyncApis=SyncApis()):
        sync_apis.curr_computation="ne16_wait_curr_computation"
        sync_apis.wait_input_transfers="ne16_wait_input_transfers"
        sync_apis.wait_output_transfers="ne16_wait_output_transfers"
        return sync_apis

    def types_def(self,types: MatchTypes=MatchTypes()):
        #types.kernel_struct="cluster_kernel"
        types.mem_data_macro_and_type="GAP_L2_DATA uint8_t"
        return types

    def cost_model(self):
        return Gap9NE16CostModel
    
    def layout_per_operand_def(self, pattern_name, specific_pattern, operands):
        return {operand:"NHWC" for operand in operands}
    
    @staticmethod
    def weightEncode(
        weight: npt.NDArray[np.uint8], bits: int, depthwise: bool = False
    ) -> npt.NDArray[np.uint8]:
        """Unroll weight into expected memory format

        Expected weight shape is (cout, cin, height, width).
        The output shape is: (cout, cinMajor, Bits, height x width, cinMinorBytes),
        where cinMajor is the ceil(cin / CIN_SUBTILE) and cinMinor has to be padded with 0 to CIN_SUBTILE.
        """
        if depthwise:
            weight = weight.transpose(1, 0, 2, 3)  # Swap cout and cin

        cout, cin, height, width = weight.shape

        # Pad cin to be divisible with CIN_SUBTILE
        if cin % 16 != 0:
            cinPad = 16 - cin % 16
            weight = np.pad(
                weight,
                ((0, 0), (0, cinPad), (0, 0), (0, 0)),
                "constant",
                constant_values=0,
            )
            cin = cin + cinPad

        # Reshape into (cout, cinMajor, cinMinor, flattened spatial, 1)
        # The 1 at the end is required by the unpacking
        cinMajor = cin // 16
        cinMinor = 16
        weight = weight.reshape(cout, cinMajor, cinMinor, height * width, 1)

        # Unpack 'bits' bits in little order, e.g. bits=4: 3 => [1, 1, 0, 0]
        # (cout, cinMajor, cinMinor, flattened spatial, Bits)
        weight = np.unpackbits(weight.astype(np.uint8), axis=-1, count=bits, bitorder="little")

        # Shuffle bits so that the final shape is:
        # (cout, cinMajor, Bits, flattened spatial, cinMinor)
        weight = weight.transpose(0, 1, 4, 3, 2)

        # Prepare for packing
        # (cout, cinMajor, Bits, flattened spatial, cinMinorBytes, 8)
        cinMinorBytes = int(np.ceil(cinMinor / 8))
        weight = np.stack(np.split(weight, cinMinorBytes, axis=-1), axis=-2)

        # Pack
        # (cout, cinMajor, Bits, flattened spatial, cinMinorBytes)
        weight = np.packbits(weight, axis=-1, bitorder="little")

        return weight.flatten()

    def weights_and_constants(self,pattern_name,layer_data,layer_arguments:List=[]):
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
        for idx,(layer_arg_name,layer_arg_val) in enumerate(layer_arguments):
            if isinstance(layer_arg_val, tvm.relay.Constant):
                if len(layer_arg_val.data.shape)==0:
                    single_constants[layer_arg_name]=str(layer_arg_val.data)
                else:
                    if "nn.conv2d" in layer_arg_name:
                        constbytes=self.weightEncode(layer_arg_val.data.numpy(),8,"nn.conv2d_depthwise" in layer_data.layer_attrs and layer_data.layer_attrs["nn.conv2d_depthwise"])
                    else:
                        constbytes=bytaze(layer_arg_val.data.numpy())
                    arguments=np.concatenate((arguments,constbytes))
        return {
            "value":c_friendly_npvalue(arguments),
            "len":arguments.shape[0],
            "shape":f"[{ceil(arguments.shape[0])}]",
            "single_costants":single_constants,
        }