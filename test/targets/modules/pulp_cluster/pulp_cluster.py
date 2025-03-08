import math
import os
from match.partition.utils import add_checks_get_first_op
from match.schedule.buffer import MatchMemBuffer
from match.target.exec_module import ExecModule
from match.cost_model.examples.pulp_cluster import PulpClusterCostModel
from match.target.memory_inst import MemoryInst
from match.transform.layout import MatchLayoutNCHWtoNHWC
from match.transform.requant import MatchRequantRewriter
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant
from match.partition.partitioning_pattern import PartitioningPattern

class PulpCluster(ExecModule):
    def __init__(self, num_cores: int=8, l1_kb_size: int=64, l2_kb_size: int=512,
                 l3_kb_size: int=8912, async_dma: bool=False):
        super(PulpCluster, self).__init__(name="pulp_cluster",
                                          src_path=os.path.dirname(__file__)+"/src",
                                          inc_path=os.path.dirname(__file__)+"/include")
        self.host_memory = "L2_SHARED_MEM"
        self.NUM_CORES = num_cores
        self.L1_SCRATCHPAD_KB_SIZE = l1_kb_size
        self.L2_SHARED_MEM_KB_SIZE = l2_kb_size
        self.L3_FLASH_KB_SIZE = l3_kb_size
        self.ASYNC_DMA = async_dma

    def memories_def(self, pattern_name, operands):
        return [
            # from lower level to higher level memories
            MemoryInst(name="L1_SCRATCHPAD",operands=["I","W","O"],k_bytes=self.L1_SCRATCHPAD_KB_SIZE,sw_controlled=True),
            MemoryInst(name="L2_SHARED_MEM",operands=["I","W","O"],k_bytes=self.L2_SHARED_MEM_KB_SIZE),
            # MemoryInst(name="L3_RAM",k_bytes=self.L3_FLASH_KB_SIZE,sw_controlled=True),
        ]

    def zigzag_optimal_spatial_mapping_def(self, match_node=None, pattern_name = "conv_2d"):
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

    def network_transformations(self, opts):
        return [
            ("requant",MatchRequantRewriter()),
            ("layout",MatchLayoutNCHWtoNHWC()),
        ]

    def update_constants(self, match_node, pattern_name):
        for w_tensor in match_node.const_tensors.values():
            if "dense" in w_tensor.name:
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

    def set_buffers_for_schedule(self, match_node, schedule, pattern_name, engine):
        if engine=="zigzag" and "conv2d" in pattern_name and pattern_name!="pointwise_conv2d":
            inp_tensor = match_node.var_tensors[match_node.var_names[0]]
            padding = match_node.ops["conv2d"].padding
            filter_shape = match_node.ops["conv2d"].kernel_size
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

    def platform_apis_def(self, platform_apis = ...):
        platform_apis.init_platform = "offload_to_pulp_cluster"
        platform_apis.init_module = "cluster_lib_init"
        return platform_apis
    
    def mem_apis_def(self, memory_apis = ...):
        memory_apis.mem_transfer = "handle_dma_transfer"
        memory_apis.alloc_buffer = "cluster_alloc_buffer"
        memory_apis.init_memory["L1_SCRATCHPAD"] = "init_l1_scratchpad_memory"
        memory_apis.free_memory["L1_SCRATCHPAD"] = "free_l1_scrachpad_memory"
        return memory_apis
    
    def sync_apis_def(self, sync_apis = ...):
        sync_apis.wait_load = "wait_l1_dma_transfers"
        sync_apis.wait_store = "wait_l1_dma_transfers"
        sync_apis.wait_tile_computation = "wait_pulp_nn_computation"
        sync_apis.must_sync_after_store = True
        sync_apis.must_sync_after_computation = True
        return sync_apis
    
    def comp_apis_def(self, computational_apis = ...):
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
            bias_add = is_op("nn.bias_add")(dense, wildcard())
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
        
        def add_pt_requant():
            cast_a = is_op("cast")(wildcard())
            cast_b = is_op("cast")(wildcard())
            add = is_op("add")(cast_a, cast_b)
            # pattern cast cast add clip cast cast multiply right shift cast
            clip = is_op("clip")(add)
            cast_c = is_op("cast")(clip)
            cast_d = is_op("cast")(cast_c)
            mul = is_op("multiply")(is_constant(),cast_d)
            rshift = is_op("right_shift")(mul, is_constant())
            # pattern cast cast add right shif clip cast
            rshift_clip = is_op("clip")(is_op("right_shift")(add,is_constant()))
            # cast for both paths
            pt = is_op("cast")(rshift | rshift_clip)
            return pt

        def only_out_uint8(node):
            return add_checks_get_first_op(node, "cast").attrs.dtype=="uint8"

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
            out_chs = conv.args[1].checked_type.shape[0]
            if not only_out_uint8(node):
                return False
            if conv.attrs.groups!=out_chs:
                return False
            if conv.attrs.data_layout!="NHWC":
                return False
            return True

        return [
            PartitioningPattern(name="dense_out",pattern=dense_pt_out),
            PartitioningPattern(name="dense",pattern=dense_pt_requant,additional_checks=only_out_uint8),
            PartitioningPattern(name="conv2d",pattern=conv_pt_requant,additional_checks=only_std_convs),
            PartitioningPattern(name="depthwise_conv2d",pattern=conv_pt_requant,additional_checks=only_dw_convs),
            PartitioningPattern(name="pointwise_conv2d",pattern=conv_pt_requant,additional_checks=only_pw_convs),
            PartitioningPattern(name="add_requant",pattern=add_pt_requant,additional_checks=only_out_uint8),
        ]