import math
import os
from match.node.node import MatchNode
from match.ops.conv2d import MatchOpConv2D
from match.partition.utils import add_checks_get_first_op
from match.schedule.buffer import MatchMemBuffer
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ComputationalApis, ExecModule, MemoryApis, ModuleLib, PlatformApis, SyncApis
from match.cost_model.examples.pulp_cluster import PulpClusterCostModel
from match.target.memory_inst import MemoryInst
from match.tensor.tensor import MatchTensor
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, has_dtype
from match.partition.partitioning_pattern import PartitioningPattern

class PulpCluster(ExecModule):
    def __init__(self, num_cores: int=8, l1_kb_size: int=64, l2_kb_size: int=512,
                 l3_kb_size: int=8912, async_dma: bool=False):
        super(PulpCluster, self).__init__(name="pulp_cluster",
                                          libs_required={
                                              "carfield_lib": ModuleLib(name="carfield_lib", base_path=os.path.dirname(__file__)+"/libs/carfield_lib"),
                                              "pulp_nn": ModuleLib(name="pulp_nn", base_path=os.path.dirname(__file__)+"/libs/pulp_nn"),
                                              "pulp_nn_fp16": ModuleLib(name="pulp_nn", base_path=os.path.dirname(__file__)+"/libs/pulp_nn_fp16"),
                                          })
        self.NUM_CORES = num_cores
        self.L1_SCRATCHPAD_KB_SIZE = l1_kb_size
        self.L2_SHARED_MEM_KB_SIZE = l2_kb_size
        self.L3_FLASH_KB_SIZE = l3_kb_size
        self.ASYNC_DMA = async_dma
        # self.schedule_engine = "basic"
        self.separate_build = True # Requires separate compilation
        self.is_smp = True # Execution model is Symmetric Multiprocessing
        self.shared_memory_extern_addr = "offload_args"
        # Host functions specific to this exec module
        self.host_send_task_fn = "pulp_cluster_send_task_mbox"
        self.host_wait_end_of_task_fn = "pulp_cluster_wait_end_of_task_mbox"
        self.timer_start_fn = "cluster_timer_start"
        self.timer_stop_fn = "cluster_timer_stop"

    def include_list(self):
        return ["carfield_lib/cluster"]

    def module_memories(self):
        return [
            # from lower level to higher level memories
            MemoryInst(name="L1_SCRATCHPAD",k_bytes=self.L1_SCRATCHPAD_KB_SIZE,sw_controlled=True),
        ]

    def zigzag_optimal_spatial_mapping_def(self, match_node: MatchNode=None, pattern_name = "conv2d"):
        if pattern_name == "pointwise_conv2d":
            return [
                ("OY",8),("OX",2),("K",4)
            ]
        elif pattern_name == "depthwise_conv2d":
            return [
                ("K",8),("OX",8),("OY",self.FULL_DIM)
            ]
        elif pattern_name == "conv2d":
            return [
                ("OY",8),("OX",2),("K",4)
            ]
        elif pattern_name == "add_requant":
            return [
                ("OY",8),("OX",2)
            ]
        elif "dense" in pattern_name:
            # TODO: K 8 C 4
            return [
                ("K",8)
            ]
        else:
            # DEFAULT LIKE CONV2D
            return [
                ("OY",8),("OX",2),("K",4)
            ]

    def zigzag_cost_model(self):
        return PulpClusterCostModel

    def update_constants(self, match_node: MatchNode=None, pattern_name: str="conv2d"):
        for w_tensor in match_node.const_tensors.values():
            if "dense" in w_tensor.name and pattern_name=="flatten_dense_out":
                if w_tensor.layout!="CN":
                    w_tensor.data = w_tensor.data.transpose(1,0)
                    w_tensor.dims = [w_tensor.dims[1], w_tensor.dims[0]]
                w_tensor.layout = "CN"
            elif "conv2d" in w_tensor.name:
                if w_tensor.layout=="HWIO":
                    w_tensor.data = w_tensor.data.transpose(3,0,1,2)
                    w_tensor.dims = [w_tensor.dims[3], w_tensor.dims[0], w_tensor.dims[1], w_tensor.dims[2]]
                elif w_tensor.layout=="OIHW":
                    w_tensor.data = w_tensor.data.transpose(0,2,3,1)
                    w_tensor.dims = [w_tensor.dims[0], w_tensor.dims[2], w_tensor.dims[3], w_tensor.dims[1]]
                w_tensor.layout = "OHWI"

    def set_buffers_for_schedule(self, match_node: MatchNode=None, schedule: MatchSchedule=None,
                                 pattern_name: str="conv2d", engine: str="ZigZag"):
        if engine=="ZigZag" and "conv2d" in pattern_name and pattern_name!="pointwise_conv2d":
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
            if im2col_size_l1:
                schedule.buffers.append(MatchMemBuffer(name="im2col", mem_name="L1_SCRATCHPAD",
                                                   num_bytes=im2col_size_l1))
            # I searched in the pulp_nn lib but also for DW convs the pwt buffer(bufferB in pulp_nn_depthwise_generic declaration)
            # doesnt seem to be used anywhere...

    def platform_apis_def(self, platform_apis: PlatformApis=None, pattern_name: str="conv2d"):
        platform_apis.init_platform = "pulp_cluster_offload_async"
        platform_apis.init_module = "cluster_lib_init"
        platform_apis.free_module = "cluster_lib_cleanup"
        
        platform_apis.smp_configured_core_guard = "cluster_check_should_run"
        platform_apis.smp_primary_core_guard = "cluster_check_main_core"
        
        platform_apis.print_fn = "mini_printf"
        
        platform_apis.wait_for_task_fn = "cluster_wait_for_task_mbox"
        platform_apis.end_of_task_fn = "cluster_end_of_task_mbox"
        
        return platform_apis
    
    def mem_apis_def(self, memory_apis: MemoryApis=None, pattern_name="conv2d"):
        memory_apis.mem_transfer = "handle_dma_transfer"
        memory_apis.alloc_buffer = "cluster_alloc_buffer"
        memory_apis.init_memory["L1_SCRATCHPAD"] = "init_l1_scratchpad_memory"
        memory_apis.free_memory["L1_SCRATCHPAD"] = "free_l1_scrachpad_memory"
        
        return memory_apis
    
    def sync_apis_def(self, sync_apis: SyncApis=None, pattern_name: str="conv2d"):
        sync_apis.wait_load = "wait_l1_dma_transfers"
        sync_apis.wait_store = "wait_l1_dma_transfers"
        sync_apis.must_sync_after_store = True
        sync_apis.must_sync_after_computation = False
        
        sync_apis.smp_barrier = "cluster_sync_cores"
        
        return sync_apis
    
    def comp_apis_def(self, computational_apis: ComputationalApis=None, pattern_name: str="conv2d"):
        computational_apis.compute_tile = "pulp_nn_wrapper"
        return computational_apis
    
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

        
        def dense_pt_requant():
            """Create pattern for conv2D with optional fused relu."""
            dense = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            dense = is_op("cast")(dense) | dense
            bias_add = is_op("nn.bias_add")(dense, wildcard()) | is_op("add")(dense, wildcard())
            scale = is_op("multiply")(dense, wildcard()) | is_op("multiply")(wildcard(), dense)
            bias = is_op("add")(scale, wildcard()) | is_op("add")(wildcard(), scale)
            right_shift = is_op("right_shift")(bias_add | bias, is_constant())
            clip = is_op("clip")(right_shift)
            cast = is_op("cast")(clip)
            return cast

        def dense_pt_out():
            dense = is_op("nn.dense")(
                wildcard(), wildcard()
            )
            add = is_op("add")(dense, is_constant()) | is_op("add")(is_op("cast")(dense),is_constant())
            return add
        
        def dense_fp16():
            dense = is_op("nn.dense")(wildcard(), wildcard()) 
            dense_add = is_op("add")(dense, is_constant())
            return dense_add
        
        def add_pt_requant():
            cast_a = is_op("cast")(wildcard())
            cast_b = is_op("cast")(wildcard())
            add = is_op("add")(cast_a, cast_b)
            # pattern cast cast add clip cast cast multiply right shift cast
            mul = is_op("multiply")(is_constant(), add) | is_op("multiply")(add, is_constant())
            rshift = is_op("right_shift")(mul, is_constant())
            # cast for both paths
            pt = is_op("cast")(is_op("clip")(rshift))
            return pt

        def only_out_uint8(node):
            return add_checks_get_first_op(node, "cast").attrs.dtype=="uint8"
        
        def only_out_fp16(node):
            is_fp16 = add_checks_get_first_op(node, "nn.dense").attrs.out_dtype == "float16"
            is_fp16 |= getattr(node.attrs, "out_dtype", None) == "float16"
            return is_fp16

        def only_std_convs(node):
            conv = add_checks_get_first_op(node, "nn.conv2d")
            if not only_out_uint8(node):
                return False
            # theres a pointwise specific pattern
            if tuple([int(i) for i in conv.attrs.kernel_size]) == (1,1):
                return False
            if conv.attrs.groups!=1:
                return False
            if conv.attrs.data_layout!="NHWC":
                return False
            return True

        def only_pw_convs(node):
            conv = add_checks_get_first_op(node, "nn.conv2d")
            if not only_out_uint8(node):
                return False
            if tuple([int(i) for i in conv.attrs.kernel_size]) != (1,1):
                return False
            if conv.attrs.groups!=1:
                return False
            if conv.attrs.data_layout!="NHWC":
                return False
            return True
        
        def only_dw_convs(node):
            conv = add_checks_get_first_op(node, "nn.conv2d")
            out_chs = conv.args[1].checked_type.shape[3]
            if not only_out_uint8(node):
                return False
            if conv.attrs.groups!=out_chs:
                return False
            if conv.attrs.data_layout!="NHWC":
                return False
            return True

        return [
            #PartitioningPattern(name="dense_out",pattern=dense_pt_out),
            PartitioningPattern(name="dense_fp16",pattern=dense_fp16,additional_checks=only_out_fp16),
            PartitioningPattern(name="dense",pattern=dense_pt_requant,additional_checks=only_out_uint8),
            PartitioningPattern(name="conv2d",pattern=conv_pt_requant,additional_checks=only_std_convs),
            PartitioningPattern(name="depthwise_conv2d",pattern=conv_pt_requant,additional_checks=only_dw_convs),
            PartitioningPattern(name="pointwise_conv2d",pattern=conv_pt_requant,additional_checks=only_pw_convs),
            PartitioningPattern(name="add_requant",pattern=add_pt_requant,additional_checks=only_out_uint8),
        ]