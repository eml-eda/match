import os
import math

from match.node.node import MatchNode
from match.ops.conv2d import MatchOpConv2D
from match.partition.utils import add_checks_get_first_op
from match.schedule.buffer import MatchMemBuffer
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ComputationalApis, ExecModule, MemoryApis, ModuleLib, PlatformApis, SyncApis
from match.cost_model.examples.pulp_cluster import PulpClusterCostModel
from match.target.memory_inst import MemoryInst
from match.tensor.tensor import MatchTensor
from match.partition.partitioning_pattern import PartitioningPattern

from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, has_dtype


class Spatz(ExecModule):
    def __init__(
        self,
        num_cores: int = 8,
        l1_kb_size: int = 64,
        l2_kb_size: int = 512,
        l3_kb_size: int = 8912,
        async_dma: bool = False,
    ):
        cur_path = os.path.dirname(__file__)
        super(Spatz, self).__init__(
            name="spatz",
            libs_required={
                "carfield_lib": ModuleLib(
                    name="carfield_lib", base_path=cur_path + "/libs/carfield_lib"
                ),
                "spatz_kernels": ModuleLib(
                    name="spatz_kernels", base_path=cur_path + "/libs/spatz_kernels"
                )
            },
        )
        self.NUM_CORES = num_cores
        self.L1_SCRATCHPAD_KB_SIZE = l1_kb_size
        self.L2_SHARED_MEM_KB_SIZE = l2_kb_size
        self.L3_FLASH_KB_SIZE = l3_kb_size
        self.ASYNC_DMA = async_dma
        # self.schedule_engine = "basic"
        # Requires separate compilation
        self.separate_build = True
        # Execution model is Symmetric Multiprocessing
        self.is_smp = True
        # Shared memory extern address variable
        self.shared_memory_extern_addr = "spatz_args"
        # Timer functions
        self.timer_start_fn = "spatz_timer_start"
        self.timer_stop_fn = "spatz_timer_stop"
        # Host functions specific to this exec module
        self.host_send_task_fn = "spatz_send_task_mbox"
        self.host_wait_end_of_task_fn = "spatz_wait_end_of_task_mbox"
        

    def include_list(self):
        return ["carfield_lib/spatz"]

    def module_memories(self):
        return [
            # from lower level to higher level memories
            MemoryInst(name="MEM_L1_SPATZ", k_bytes=self.L1_SCRATCHPAD_KB_SIZE, sw_controlled=True),
        ]

    def zigzag_optimal_spatial_mapping_def(
        self, match_node: MatchNode = None, pattern_name="conv2d"
    ):
        if pattern_name == "pointwise_conv2d":
            return [("OY", 8), ("OX", 2), ("K", 4)]
        elif pattern_name == "depthwise_conv2d":
            return [("K", 8), ("OX", 8), ("OY", self.FULL_DIM)]
        elif pattern_name == "conv2d":
            return [("OY", 8), ("OX", 2), ("K", 4)]
        elif pattern_name == "add_requant":
            return [("OY", 8), ("OX", 2)]
        elif "dense" in pattern_name:
            # TODO: K 8 C 4
            return [("K", 8)]
        else:
            # DEFAULT LIKE CONV2D
            return [("OY", 8), ("OX", 2), ("K", 4)]

    def zigzag_cost_model(self):
        return PulpClusterCostModel

    def update_constants(self, match_node: MatchNode=None, pattern_name: str="conv2d"):
        for w_tensor in match_node.const_tensors.values():
            if "dense" in w_tensor.name and pattern_name == "flatten_dense_out":
                if w_tensor.layout != "CN":
                    w_tensor.data = w_tensor.data.transpose(1,0)
                    w_tensor.dims = [w_tensor.dims[1], w_tensor.dims[0]]
                w_tensor.layout = "CN"
            elif "dense" in w_tensor.name and "dense_fp16" in pattern_name:
                # RedMulE expects weights in CN layout
                if w_tensor.layout != "CN":
                    w_tensor.data = w_tensor.data.transpose(1,0)
                    w_tensor.dims = [w_tensor.dims[1], w_tensor.dims[0]]
                w_tensor.layout = "CN"
            elif "conv2d" in w_tensor.name:
                if w_tensor.layout == "HWIO":
                    w_tensor.data = w_tensor.data.transpose(3,0,1,2)
                    w_tensor.dims = [w_tensor.dims[3], w_tensor.dims[0], w_tensor.dims[1], w_tensor.dims[2]]
                elif w_tensor.layout == "OIHW":
                    w_tensor.data = w_tensor.data.transpose(0,2,3,1)
                    w_tensor.dims = [w_tensor.dims[0], w_tensor.dims[2], w_tensor.dims[3], w_tensor.dims[1]]
                w_tensor.layout = "OHWI"

    def set_buffers_for_schedule(self, match_node: MatchNode=None, schedule: MatchSchedule=None,
                                 pattern_name: str="conv2d", engine: str="ZigZag"):
        if engine=="ZigZag" and "conv2d" in pattern_name and pattern_name!="pointwise_conv2d" and False:
            inp_tensor: MatchTensor = match_node.var_tensors[match_node.var_names[0]]
            conv: MatchOpConv2D = match_node.ops["conv2d"]
            padding = conv.padding
            filter_shape = conv.kernel_size
            tile_inp_chs = schedule.tensor_tiles[inp_tensor.name][0].tiled_dims[3].size
            im2col_size_l1 = 0
            # im2col size only for std convs
            if pattern_name=="conv2d":
                # 2 * CORES * np.prod(ks) * tile_n_in
                im2col_size_l1 = 2 * self.NUM_CORES * math.prod(filter_shape) * tile_inp_chs
            elif pattern_name=="depthwise_conv2d":
                # CORES * (ks[0] * (tile_n_in + p[0] + p[2]) + ks[0])
                im2col_size_l1 = self.NUM_CORES * (filter_shape[0] * (tile_inp_chs + padding[0] + padding[2]) + filter_shape[0])
            elif pattern_name=="conv2d_fp16":
                im2col_size_l1 = 2 * self.NUM_CORES * math.prod(filter_shape) * tile_inp_chs
            if im2col_size_l1:
                schedule.buffers.append(MatchMemBuffer(name="im2col", mem_name="MEM_L1_SPATZ",
                                                   num_bytes=im2col_size_l1))
            # I searched in the pulp_nn lib but also for DW convs the pwt buffer(bufferB in pulp_nn_depthwise_generic declaration)
            # doesnt seem to be used anywhere...

    def platform_apis_def(self, platform_apis: PlatformApis=None, pattern_name: str="conv2d"):
        platform_apis.startup_fn = "spatz_startup"
        platform_apis.init_platform = "spatz_offload_async"
        platform_apis.init_module = "spatz_init"
        platform_apis.free_module = "spatz_free"
        
        platform_apis.smp_configured_core_guard = "spatz_check_should_run"
        platform_apis.smp_primary_core_guard = "spatz_check_main_core"
        
        platform_apis.print_fn = "mini_printf"
        
        platform_apis.wait_for_task_fn = "spatz_wait_for_task_mbox"
        platform_apis.end_of_task_fn = "spatz_end_of_task_mbox"
        
        return platform_apis
    
    def mem_apis_def(self, memory_apis: MemoryApis=None, pattern_name="conv2d"):
        memory_apis.mem_transfer = "handle_dma_transfer"
        memory_apis.alloc_buffer = "spatz_alloc_buffer"
        memory_apis.init_memory["MEM_L1_SPATZ"] = "spatz_l1_init"
        memory_apis.free_memory["MEM_L1_SPATZ"] = "spatz_l1_free"
        
        return memory_apis
    
    def sync_apis_def(self, sync_apis: SyncApis=None, pattern_name: str="conv2d"):
        sync_apis.wait_load = "wait_l1_dma_transfers"
        sync_apis.wait_store = "wait_l1_dma_transfers"
        sync_apis.must_sync_after_store = True
        sync_apis.must_sync_after_computation = False
        
        sync_apis.smp_barrier = "spatz_sync_cores"
        
        return sync_apis
    
    def comp_apis_def(self, computational_apis: ComputationalApis=None, pattern_name: str="conv2d"):
        computational_apis.compute_tile = "kernel_wrapper"
        return computational_apis
    
    
    def partitioning_patterns(self):
        
        def check_fp16_out(node):
            #return False
            is_fp16 = (node.args[0].checked_type.dtype == "float16")
            return is_fp16
        
        def check_conv2d_even_channels(node):
            conv = add_checks_get_first_op(node, "nn.conv2d")
            has_even_in_channels = (conv.args[0].checked_type.shape[3] % 2 == 0)
            has_even_out_channels = (conv.args[1].checked_type.shape[3] % 2 == 0)
            return has_even_in_channels and has_even_out_channels
        
        def check_conv2d_even_output_channels(node):
            conv = add_checks_get_first_op(node, "nn.conv2d")
            has_even_out_channels = (conv.args[1].checked_type.shape[3] % 2 == 0)
            return has_even_out_channels
        
        def check_conv2d_even_input_group_channels(node):
            conv = add_checks_get_first_op(node, "nn.conv2d")
            has_even_in_group_channels = ((conv.args[0].checked_type.shape[3] // conv.attrs.groups) % 2 == 0)
            return has_even_in_group_channels

        def check_linear_even_out_channels(node):
            dense = add_checks_get_first_op(node, "nn.dense")
            has_even_out_channels = (dense.args[1].checked_type.shape[0] % 4 == 0)
            return has_even_out_channels
        
        def spatz_conv2d_check(node):
            valid = check_fp16_out(node)
            valid = valid and check_conv2d_even_channels(node)
            
            conv = add_checks_get_first_op(node, "nn.conv2d")
            valid = valid and conv.attrs.data_layout == "NHWC"
            valid = valid and conv.attrs.groups == 1 # Grouped conv currently not supported
            valid = valid and conv.attrs.dilation[0] == 1 and conv.attrs.dilation[1] == 1 # Dilation currently not supported
            
            return valid
        
        def spatz_conv2d_grouped_check(node):
            valid = check_fp16_out(node)
            valid = valid and check_conv2d_even_output_channels(node)
            valid = valid and check_conv2d_even_input_group_channels(node)
            
            conv = add_checks_get_first_op(node, "nn.conv2d")
            valid = valid and conv.attrs.data_layout == "NHWC"
            valid = valid and conv.attrs.groups == conv.args[1].checked_type.shape[3] # Only depthwise conv supported in match
            valid = valid and conv.attrs.dilation[0] == 1 and conv.attrs.dilation[1] == 1 # Dilation currently not supported
            return valid
        
        def spatz_dense_check(node):
            return check_fp16_out(node) and check_linear_even_out_channels(node)
        
        def conv2d():
            conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
            return conv2d
        
        def conv2d_bias():
            conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
            conv2d_add = is_op("add")(conv2d, is_constant()) | is_op("add")(is_constant(), conv2d)
            return conv2d_add
        
        def conv2d_bnorm():
            conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
            conv2d_bias = is_op("add")(conv2d, is_constant()) | is_op("add")(is_constant(), conv2d)
            conv2d_batch_mul = is_op("multiply")(conv2d_bias, is_constant()) | is_op("multiply")(is_constant(), conv2d_bias)
            conv2d_batch_add = is_op("add")(conv2d_batch_mul, is_constant()) | is_op("add")(is_constant(), conv2d_batch_mul)
            return conv2d_batch_add
        
        def dense():
            dense = is_op("nn.dense")(wildcard(), wildcard())
            return dense
        
        def dense_bias():
            dense = is_op("nn.dense")(wildcard(), wildcard())
            dense_add = is_op("add")(dense, wildcard()) | is_op("add")(wildcard(), dense)
            return dense_add
        
        return [
            PartitioningPattern(name="spatz_conv2d_fp16", pattern=conv2d, additional_checks=spatz_conv2d_check),
            PartitioningPattern(name="spatz_conv2d_bias_fp16", pattern=conv2d_bias, additional_checks=spatz_conv2d_check),
            PartitioningPattern(name="spatz_conv2d_bnorm_fp16", pattern=conv2d_bnorm, additional_checks=spatz_conv2d_check),
            
            PartitioningPattern(name="spatz_conv2d_grouped_fp16", pattern=conv2d, additional_checks=spatz_conv2d_grouped_check),
            PartitioningPattern(name="spatz_conv2d_grouped_bias_fp16", pattern=conv2d_bias, additional_checks=spatz_conv2d_grouped_check),
            PartitioningPattern(name="spatz_conv2d_grouped_bnorm_fp16", pattern=conv2d_bnorm, additional_checks=spatz_conv2d_grouped_check),
            
            PartitioningPattern(name="spatz_dense_fp16", pattern=dense, additional_checks=spatz_dense_check),
            PartitioningPattern(name="spatz_dense_bias_fp16", pattern=dense_bias, additional_checks=spatz_dense_check),
        ]