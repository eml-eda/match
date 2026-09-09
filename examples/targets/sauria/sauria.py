import os

from match.partition.utils import add_checks_get_first_op
from match.target.exec_module import ComputationalApis, ExecModule, MemoryApis, ModuleLib, PlatformApis, SyncApis
from match.target.memory_inst import MemoryInst

from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, has_dtype
from match.partition.partitioning_pattern import PartitioningPattern

__SAURIA_OUTPUT_BANK_SIZE__ = 32
__SAURIA_WEIGHTS_BANK_SIZE__ = 32
__SAURIA_IFMAP_BANK_SIZE__ = 32
__SAURIA_SCRATCHPAD_SIZE__ = __SAURIA_OUTPUT_BANK_SIZE__ \
    + __SAURIA_WEIGHTS_BANK_SIZE__ + __SAURIA_IFMAP_BANK_SIZE__

class Sauria(ExecModule):
    def __init__(self):
        curr_path = os.path.dirname(__file__)
        print(f"[Sauria] Initializing Sauria target with base path: {curr_path}")
        print(f"[Sauria] Sauriatch lib base path {curr_path + '/libs/sauriatch'}")
        print(f"[Sauria] Carfield lib base path {curr_path + '/../astral_libs/carfield_lib'}")
        super().__init__(
            libs_required = {
                "sauriatch" : ModuleLib(
                    name="sauriatch", base_path=curr_path + "/libs/sauriatch"
                ),
                "carfield_lib": ModuleLib(
                    name="carfield_lib", base_path=curr_path + "/../astral_libs/carfield_lib"
                ),
            }
        )
        self.is_smp = False
        self.schedule_engine = "CP"
    
    def include_lists(self):
        self.include_list = [
            "sauriatch/sauria.h",
        ]

    def module_memories(self):
        return [
            MemoryInst(
                name = "__sauria_scratchpad_output_bank__",
                k_bytes = __SAURIA_OUTPUT_BANK_SIZE__,
                sw_controlled = True,
                tensor_types = ["output", "intermediate"],
            ),
            MemoryInst(
                name = "__sauria_scratchpad_weights_bank__",
                k_bytes = __SAURIA_WEIGHTS_BANK_SIZE__,
                sw_controlled = True,
                tensor_types = ["const"],
            ),
            MemoryInst(
                name = "__sauria_scratchpad_ifmap_bank__",
                k_bytes = __SAURIA_IFMAP_BANK_SIZE__,
                sw_controlled = True,
                tensor_types = ["var"],
            ),
        ]

    def mem_apis_def(self, memory_apis = MemoryApis(), pattern_name = "conv2d"):
        memory_apis.mem_transfer = "sauria_dma_transfer"
        memory_apis.init_memory["__sauria_scratchpad_output_bank__"] = "__sauria_scratchpad_get_output_bank__"
        memory_apis.init_memory["__sauria_scratchpad_weights_bank__"] = "__sauria_scratchpad_get_weights_bank__"
        memory_apis.init_memory["__sauria_scratchpad_ifmap_bank__"] = "__sauria_scratchpad_get_ifmap_bank__"

        return memory_apis

    def sync_apis_def(self, sync_apis = SyncApis(), pattern_name = "conv2d"):
        sync_apis.must_sync_after_store = True
        sync_apis.must_sync_after_computation = False

        sync_apis.wait_tile_computation = "__sauria_wait_compute__"
        sync_apis.wait_load = "__sauria_wait_dma_transfer__"
        sync_apis.wait_store = "__sauria_wait_dma_transfer__"

        return sync_apis

    def comp_apis_def(self, computational_apis = ComputationalApis(), pattern_name: str="conv2d"):
        computational_apis.compute_tile = "__sauria_compute__"
        return computational_apis

    def platform_apis_def(self, platform_apis = PlatformApis(), pattern_name: str="conv2d"):
        platform_apis.init_platform = "__init_sauria__"
        return platform_apis

    def partitioning_patterns(self):
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

        def conv3d_pt_requant():
            """Create pattern for conv2D with optional fused relu."""
            conv3d = is_op("nn.conv3d")(
                wildcard(), wildcard()
            )
            conv3d = is_op("cast")(conv3d) | conv3d
            bias_add = is_op("nn.bias_add")(conv3d, wildcard()) | is_op("add")(conv3d, wildcard())
            scale = is_op("multiply")(conv3d, wildcard()) | is_op("multiply")(wildcard(), conv3d)
            bias = is_op("add")(scale, wildcard()) | is_op("add")(wildcard(), scale)
            right_shift = is_op("right_shift")(bias_add | bias, is_constant())
            clip = is_op("clip")(right_shift)
            cast = is_op("cast")(clip)
            return cast
        
        def only_out_uint8(node):
            return add_checks_get_first_op(node, "cast").attrs.dtype=="uint8"

        return [
            # PartitioningPattern(
            #     name = "sauria_dense_requant",
            #     pattern = dense_pt_requant,
            #     additional_checks = only_out_uint8,
            # ),
            PartitioningPattern(
                name = "sauria_conv3d_requant",
                pattern = conv3d_pt_requant,
                additional_checks = only_out_uint8,
            ),
        ]