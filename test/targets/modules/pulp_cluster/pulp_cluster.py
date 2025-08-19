import math
import os
from match.node.node import MatchNode
from match.ops.conv2d import MatchOpConv2D
from match.ops.conv2d_transpose import MatchOpConv2DTranspose
from match.ops.conv3d import MatchOpConv3D
from match.schedule.buffer import MatchMemBuffer
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ComputationalApis, ExecModule, MemoryApis, ModuleLib, PlatformApis, SyncApis
from match.cost_model.examples.pulp_cluster import PulpClusterCostModel
from match.target.memory_inst import MemoryInst
from match.tensor.tensor import MatchTensor
from match.partition.partitioning_pattern import PartitioningPattern
from .patterns import add_pt_requant, bw_instance_norm_tail_pt, conv2d_bw, conv2d_bw_check, conv2d_fw, conv2d_transpose_ptrain_check, conv2d_transpose_ptrain_pt, conv3d_pt_requant, conv_pt_requant, dense_pt_out, dense_pt_requant, dw_convs_fp32_pulp, fw_instance_norm_tail_pt, only_dw_convs, only_out_int32, only_out_uint8, only_pw_convs, only_std_convs, std_convs_fp32

class PulpCluster(ExecModule):
    def __init__(self, num_cores: int=8, l1_kb_size: int=64, l2_kb_size: int=512,
                 l3_kb_size: int=8912, async_dma: bool=False):
        super(PulpCluster, self).__init__(
            name="pulp_cluster",
            libs_required={
                "pulp_nn": ModuleLib(name="pulp_nn", base_path=os.path.dirname(__file__)+"/../libs/pulp_nn"),
                "pulp_cluster": ModuleLib(name="pulp_cluster", base_path=os.path.dirname(__file__)+"/../libs/pulp_cluster"),
                "pulp_mem": ModuleLib(name="pulp_mem", base_path=os.path.dirname(__file__)+"/../libs/pulp_mem"),
                "pulp_utils": ModuleLib(name="pulp_utils", base_path=os.path.dirname(__file__)+"/../libs/pulp_utils"),
                "pulp_utils": ModuleLib(name="pulp_utils", base_path=os.path.dirname(__file__)+"/../libs/pulp_utils"),
                "pulp_train": ModuleLib(name="pulp_train", base_path=os.path.dirname(__file__)+"/../libs/pulp_train"),
            }
        )
        self.NUM_CORES = num_cores
        self.L1_SCRATCHPAD_KB_SIZE = l1_kb_size
        self.L2_SHARED_MEM_KB_SIZE = l2_kb_size
        self.L3_FLASH_KB_SIZE = l3_kb_size
        self.ASYNC_DMA = async_dma
        self.timer_start_fn = "start_perf_counter"
        self.timer_stop_fn = "stop_perf_counter"

    def get_schedule_engine_for_pt(self, pattern_name = ""):
        return "EasyTile" if pattern_name in ["fw_instance_norm_tail", "bw_instance_norm_tail"] else self.schedule_engine

    def include_list(self):
        return ["pulp_cluster/cluster", "pmsis"]

    def module_memories(self):
        return [
            # from lower level to higher level memories
            MemoryInst(name="L1_SCRATCHPAD",k_bytes=self.L1_SCRATCHPAD_KB_SIZE,sw_controlled=True),
        ]

    def zigzag_optimal_spatial_mapping_def(self, match_node: MatchNode=None, pattern_name = "conv2d"):
        if pattern_name == "conv2d_grad_params":
            return [
                # ("OY",2),("OX",2),("K",8)
                ("K",self.FULL_DIM), ("B",self.FULL_DIM)
            ]
        elif pattern_name == "conv2d_transpose":
            return [
                ("OY",8),("OX",2),("K",4)
            ]
        elif pattern_name == "pointwise_conv2d":
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
        if pattern_name in ["conv2d_train", "conv2ddw_train", "conv2d_grad_params", "conv2d_transpose", "bw_instance_norm_tail", "fw_instance_norm_tail"]:
            return
        for w_tensor in match_node.const_tensors.values():
            if "dense" in w_tensor.name and pattern_name=="flatten_dense_out":
                if w_tensor.layout!="CN":
                    w_tensor.data = w_tensor.data.transpose(1,0)
                    w_tensor.dims = [w_tensor.dims[1], w_tensor.dims[0]]
                w_tensor.layout = "CN"
            elif "conv3d" in w_tensor.name:
                if w_tensor.layout=="DHWIO":
                    w_tensor.data = w_tensor.data.transpose(4,0,1,2,3)
                    w_tensor.dims = [w_tensor.dims[4], w_tensor.dims[0], w_tensor.dims[1], w_tensor.dims[2], w_tensor.dims[3]]
                w_tensor.layout = "ODHWI"
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
        if engine=="ZigZag" and "conv2d" in pattern_name and pattern_name not in "pointwise_conv2d":
            is_pulp_train_conv = pattern_name in ["conv2d_train", "conv2ddw_train", "conv2d_transpose", "conv2d_grad_params"]
            inp_tensor: MatchTensor = match_node.var_tensors[match_node.var_names[0]]
            out_tensor: MatchTensor = match_node.output_tensors[match_node.output_names[0]]
            if pattern_name == "conv2d_transpose":
                conv: MatchOpConv2DTranspose = match_node.ops["conv2d_transpose"]
            else:
                conv: MatchOpConv2D = match_node.ops["conv2d"]
            padding = conv.padding
            filter_shape = conv.kernel_size
            stride = conv.strides
            tile_inp_chs = schedule.tensor_tiles[inp_tensor.name][1].tiled_dims[3].size
            im2col_size_l1 = 0
            bt_buffer_size_l1 = 0
            # im2col size only for std convs
            if pattern_name=="conv2d":
                # 2 * CORES * np.prod(ks) * tile_n_in
                im2col_size_l1 = 2 * self.NUM_CORES * math.prod(filter_shape) * tile_inp_chs
            elif pattern_name=="depthwise_conv2d":
                # CORES * (ks[0] * (tile_n_in + p[0] + p[2]) + ks[0])
                im2col_size_l1 = self.NUM_CORES * (filter_shape[0] * (tile_inp_chs + padding[0] + padding[2]) + filter_shape[0])
            elif pattern_name=="conv2d_train":
                # size in bytes
                tile_inp_c = schedule.tensor_tiles[inp_tensor.name][1].tiled_dims[1].size
                tile_inp_h = schedule.tensor_tiles[inp_tensor.name][1].tiled_dims[2].size
                tile_inp_w = schedule.tensor_tiles[inp_tensor.name][1].tiled_dims[3].size
                # (Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0
                # if (tile_inp_h - filter_shape[0] + padding[0] + padding[2] + conv.strides[0]) % conv.strides[0] > 0:
                #     im2col_size_l1 = 0
                if  filter_shape[0] == 1 and filter_shape[1] == 1 and \
                    padding[0] == 0 and padding[1] == 0 and padding[2] == 0 and padding[3] == 0 and \
                    stride[0] == 1 and stride[1] == 1:
                    # special pointwise acceleration do not need IM2COL buffer
                    im2col_size_l1 = 0
                else: # standard conv2d
                    im2col_size_l1 = (
                        filter_shape[0] * filter_shape[1] * tile_inp_c *
                        ((tile_inp_h - filter_shape[0] + padding[0] + padding[2] + stride[0]) // stride[0]) *
                        ((tile_inp_w - filter_shape[1] + padding[1] + padding[3] + stride[1]) // stride[1])
                    ) * 4
                # print(f"IM2COL SIZE L1: {im2col_size_l1/1024} KB")

            elif pattern_name=="conv2d_grad_params" and not conv.depthwise and filter_shape!= (1,1):
                # int im2col_rows = kernel_h * kernel_w * inp_ch;
                # int im2col_cols = out_height * out_width;
                tile_out_h = schedule.tensor_tiles[out_tensor.name][1].tiled_dims[2].size
                tile_out_w = schedule.tensor_tiles[out_tensor.name][1].tiled_dims[3].size
                im2col_size_l1 = filter_shape[0] * filter_shape[1] * tile_inp_chs * \
                                    (tile_out_h * tile_out_w) * 4

            if im2col_size_l1:
                schedule.buffers.append(
                    MatchMemBuffer(
                        name="im2col",
                        mem_name="L1_SCRATCHPAD",
                        num_bytes=im2col_size_l1,
                        required= not is_pulp_train_conv
                    )
                )
            if bt_buffer_size_l1:
                schedule.buffers.append(
                    MatchMemBuffer(
                        name="bt_buffer",
                        mem_name="L1_SCRATCHPAD",
                        num_bytes=bt_buffer_size_l1,
                        required=True
                    )
                )
            # I searched in the pulp_nn lib but also for DW convs the pwt buffer(bufferB in pulp_nn_depthwise_generic declaration)
            # doesnt seem to be used anywhere...
        elif engine=="ZigZag" and "conv3d" in pattern_name:
            inp_tensor: MatchTensor = match_node.var_tensors[match_node.var_names[0]]
            conv: MatchOpConv3D = match_node.ops["conv3d"]
            filter_shape = conv.kernel_size
            tile_inp_chs = schedule.tensor_tiles[inp_tensor.name][0].tiled_dims[4].size
            im2col_size_l1 = 2 * self.NUM_CORES * math.prod(filter_shape) * tile_inp_chs
            if im2col_size_l1:
                schedule.buffers.append(MatchMemBuffer(name="im2col", mem_name="L1_SCRATCHPAD",
                                                   num_bytes=im2col_size_l1))
                

    def platform_apis_def(self, platform_apis: PlatformApis=None, pattern_name: str="conv2d"):
        platform_apis.init_platform = "offload_to_pulp_cluster"
        platform_apis.init_module = "cluster_lib_init"
        platform_apis.free_module = "cluster_lib_cleanup"
        return platform_apis
    
    def mem_apis_def(self, memory_apis: MemoryApis=None, pattern_name="conv2d"):
        memory_apis.mem_transfer = "handle_dma_transfer"
        memory_apis.alloc_buffer = "cluster_alloc_buffer"
        memory_apis.init_memory["L1_SCRATCHPAD"] = "init_l1_scratchpad_memory"
        memory_apis.free_memory["L1_SCRATCHPAD"] = "free_l1_scratchpad_memory"
        return memory_apis
    
    def sync_apis_def(self, sync_apis: SyncApis=None, pattern_name: str="conv2d"):
        sync_apis.wait_load = "wait_l1_dma_transfers"
        sync_apis.wait_store = "wait_l1_dma_transfers"
        if pattern_name not in ["conv2d_train", "conv2ddw_train", "conv2d_transpose", "conv2d_grad_params", "bw_instance_norm_tail", "fw_instance_norm_tail"]:
            sync_apis.wait_tile_computation = "wait_pulp_nn_computation"
        else:
            sync_apis.wait_tile_computation = ""
        sync_apis.must_sync_after_store = True
        sync_apis.must_sync_after_computation = True
        return sync_apis
    
    def comp_apis_def(self, computational_apis: ComputationalApis=None, pattern_name: str="conv2d"):
        computational_apis.compute_tile = "pulp_nn_wrapper"
        return computational_apis
    
    def partitioning_patterns(self):
        
        def disabled_pattern(node):
            return False

        return [
            PartitioningPattern(name="conv3d",pattern=conv3d_pt_requant,additional_checks=only_out_uint8),
            PartitioningPattern(name="dense_out",pattern=dense_pt_out,additional_checks=only_out_int32),
            PartitioningPattern(name="dense",pattern=dense_pt_requant,additional_checks=only_out_uint8),
            PartitioningPattern(name="conv2d",pattern=conv_pt_requant,additional_checks=only_std_convs),
            PartitioningPattern(name="depthwise_conv2d",pattern=conv_pt_requant,additional_checks=only_dw_convs),
            PartitioningPattern(name="pointwise_conv2d",pattern=conv_pt_requant,additional_checks=only_pw_convs),
            PartitioningPattern(name="add_requant",pattern=add_pt_requant,additional_checks=only_out_uint8),
        ] + [
            # add training layers
            PartitioningPattern(name="conv2d_train", pattern=conv2d_fw, additional_checks=std_convs_fp32),
            PartitioningPattern(name="conv2ddw_train", pattern=conv2d_fw, additional_checks=dw_convs_fp32_pulp),
            PartitioningPattern(name="conv2d_transpose", pattern=conv2d_transpose_ptrain_pt, additional_checks=conv2d_transpose_ptrain_check),
            PartitioningPattern(name="conv2d_grad_params", pattern=conv2d_bw, additional_checks=conv2d_bw_check),
            PartitioningPattern(name="bw_instance_norm_tail", pattern=bw_instance_norm_tail_pt, additional_checks=disabled_pattern),
            PartitioningPattern(name="fw_instance_norm_tail", pattern=fw_instance_norm_tail_pt, additional_checks=disabled_pattern),
        ]