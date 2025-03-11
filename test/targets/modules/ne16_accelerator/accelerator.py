

import os
from match.partition.partitioning_pattern import PartitioningPattern
from match.partition.utils import add_checks_get_first_op
from match.target.exec_module import ComputationalApis, ExecModule, MemoryApis, ModuleLib, PlatformApis, SyncApis
from match.target.memory_inst import MemoryInst
from .cost_model import NE16AcceleratorCostModel
from .weights import NE16_transform_weights
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant

class NE16Accelerator(ExecModule):
    def __init__(self, num_cores: int=8, l1_kb_size: int=64,
                 l2_kb_size: int=512, l3_kb_size: int=8912,
                 async_dma: bool=False, **kwargs):
        super().__init__(name="NE16",
                         libs_required={
                            "pulp_nn": ModuleLib(name="pulp_nn", base_path=os.path.dirname(__file__)+"/../libs/pulp_nn"),
                            "pulp_cluster": ModuleLib(name="pulp_cluster", base_path=os.path.dirname(__file__)+"/../libs/pulp_cluster"),
                            "pulp_mem": ModuleLib(name="pulp_mem", base_path=os.path.dirname(__file__)+"/../libs/pulp_mem"),
                            "pulp_utils": ModuleLib(name="pulp_utils", base_path=os.path.dirname(__file__)+"/../libs/pulp_utils"),
                            "pulp_neural": ModuleLib(name="pulp_neural", base_path=os.path.dirname(__file__)+"/../libs/pulp_neural"),
                            "pulp_nnx": ModuleLib(name="pulp_nnx", base_path=os.path.dirname(__file__)+"/../libs/pulp_nnx"),
                        })
        self.NUM_CORES = num_cores
        self.L1_SCRATCHPAD_KB_SIZE = l1_kb_size
        self.L2_SHARED_MEM_KB_SIZE = l2_kb_size
        self.L3_FLASH_KB_SIZE = l3_kb_size
        self.ASYNC_DMA = async_dma
    
    def include_list(self):
        return [
            "pulp_cluster/cluster",
            "pulp_neural/neural_engine",
        ]

    def module_memories(self):
        return [
            # from lower level to higher level memories
            MemoryInst(name="L1_SCRATCHPAD",k_bytes=self.L1_SCRATCHPAD_KB_SIZE,sw_controlled=True),
        ]
    
    def update_constants(self, match_node = None, pattern_name = "conv2d"):
        for w_tensor in match_node.const_tensors.values():
            if "dense" in w_tensor.name:
                w_tensor.data = w_tensor.data.reshape((w_tensor.dims[0], w_tensor[1]) + (1, 1))
                w_tensor.layout = "NCHW"
                w_tensor.dims = [w_tensor.dims[0], match_node.default_dim, w_tensor.dims[1]]
            elif "conv2d" in w_tensor.name:
                if w_tensor.layout=="HWIO":
                    w_tensor.data = w_tensor.data.transpose(3,2,0,1)
                    w_tensor.dims = [w_tensor.dims[3], w_tensor.dims[2], w_tensor.dims[0], w_tensor.dims[1]]
                w_tensor.layout = "NCHW"
            if "dense" in w_tensor.name or "conv2d" in w_tensor.name:
                depthwise = "conv2d" in w_tensor.name and match_node.ops["conv2d"].depthwise
                w_tensor.data = NE16_transform_weights(weight=w_tensor.data, bits=w_tensor.bits,
                                                       depthwise=depthwise)
                if depthwise:
                    w_tensor.dims = [ match_node.default_dim, w_tensor.dims[0], w_tensor.dims[2], w_tensor.dims[3], w_tensor.dims[0]]
                else:
                    w_tensor.dims = [w_tensor.dims[0], w_tensor.dims[1], w_tensor.dims[2], w_tensor.dims[3], w_tensor.dims[1]]
                w_tensor.layout = "NCHWc16"
                w_tensor.num_dims = 5
                

    def zigzag_optimal_spatial_mapping_def(self, match_node = None, pattern_name = "conv2d"):
        # from ne16_task_defs.h
        # #define NE16_SUBTILE_INPUT_CHANNEL (16)
        # #define NE16_SUBTILE_OUTPUT_HEIGHT (3)
        # #define NE16_SUBTILE_OUTPUT_WIDTH (3)
        # #define NE16_SUBTILE_OUTPUT_CHANNEL (32)
        # input channels are automatically taken to the max value...
        subtile_out_chs = 32
        if pattern_name=="depthwise_conv2d":
            subtile_out_chs = 16
        return [
            ("K", subtile_out_chs), ("OY", 3), ("OX", 3)
        ]
    
    def zigzag_cost_model(self):
        return NE16AcceleratorCostModel
    
    def partitioning_patterns(self):
        def conv_pt_requant():
            #Create pattern for a 2D Conv block, with bias and ReLU.
            conv2d = is_op("nn.conv2d")(
                wildcard(), wildcard()
            )
            conv2d = is_op("cast")(conv2d) | conv2d
            bias_add = is_op("nn.bias_add")(conv2d, wildcard())
            scale = is_op("multiply")(conv2d, wildcard()) | is_op("multiply")(wildcard(), conv2d)
            bias = is_op("add")(scale, wildcard()) | is_op("add")(wildcard(), scale)
            right_shift = is_op("right_shift")(bias_add | bias, is_constant())
            clip = is_op("clip")(right_shift)
            cast = is_op("cast")(clip)
            return cast
        
        def only_out_uint8(node):
            return add_checks_get_first_op(node, "cast").attrs.dtype=="uint8"

        def only_std_convs(node):
            conv = add_checks_get_first_op(node, "nn.conv2d")
            if not only_out_uint8(node):
                return False
            kernel_size, strides = tuple([int(i) for i in conv.attrs.kernel_size]), tuple([int(i) for i in conv.attrs.strides])
            out_chs = conv.args[1].checked_type.shape[3]
            # only 1x1 and 3x3 convs allowed by pulp-nnx
            if kernel_size not in ((1,1), (3,3)):
                return False
            # only single or double stride allowed by pulp-nnx
            if strides not in ((1,1), (2,2)):
                return False
            # single or depthwise convolutions allowed, not grouped ones
            # depthwise ones have a specific pattern so refuse if not single grouping
            if conv.attrs.groups!=1:
                return False
            # only even output channels are allowed in 2x2 strides convs
            if out_chs%2 and strides[0]==2:
                return False
            if conv.attrs.data_layout!="NHWC":
                return False
            return True
        
        def only_dw_convs(node):
            conv = add_checks_get_first_op(node, "nn.conv2d")
            if not only_out_uint8(node):
                return False
            kernel_size, strides = tuple([int(i) for i in conv.attrs.kernel_size]), tuple([int(i) for i in conv.attrs.strides])
            out_chs = conv.args[1].checked_type.shape[3]
            # only 1x1 and 3x3 convs allowed by pulp-nnx
            if kernel_size not in ((3,3)):
                return False
            # only single or double stride allowed by pulp-nnx
            if strides not in ((1,1), (2,2)):
                return False
            # depthwise convolutions only
            if conv.attrs.groups!=out_chs and out_chs!=1:
                return False
            # only even output channels are allowed in 2x2 strides convs
            if out_chs%2 and strides[0]==2:
                return False
            if conv.attrs.data_layout!="NHWC":
                return False
            return True
        
        return [
            PartitioningPattern(name="conv2d",pattern=conv_pt_requant,additional_checks=only_std_convs),
            PartitioningPattern(name="depthwise_conv2d",pattern=conv_pt_requant,additional_checks=only_dw_convs),
        ]
    
    def platform_apis_def(self, platform_apis: PlatformApis=None, pattern_name: str="conv2d"):
        platform_apis.init_platform = "offload_to_pulp_cluster"
        platform_apis.init_module = "neural_engine_lib_init"
        platform_apis.free_module = "neural_engine_lib_close"
        return platform_apis
    
    def mem_apis_def(self, memory_apis: MemoryApis=None, pattern_name="conv2d"):
        memory_apis.mem_transfer = "handle_dma_transfer"
        memory_apis.init_memory["L1_SCRATCHPAD"] = "init_l1_scratchpad_memory"
        memory_apis.free_memory["L1_SCRATCHPAD"] = "free_l1_scrachpad_memory"
        return memory_apis
    
    def sync_apis_def(self, sync_apis: SyncApis=None, pattern_name: str="conv2d"):
        sync_apis.wait_load = "wait_l1_dma_transfers"
        sync_apis.wait_store = "wait_l1_dma_transfers"
        sync_apis.wait_tile_computation = "wait_neural_engine_compute"
        sync_apis.must_sync_after_store = True
        sync_apis.must_sync_after_computation = True
        return sync_apis
    
    def comp_apis_def(self, computational_apis: ComputationalApis=None, pattern_name: str="conv2d"):
        computational_apis.compute_tile = "neural_engine_compute_tile"
        return computational_apis