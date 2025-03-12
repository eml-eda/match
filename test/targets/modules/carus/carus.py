
import os
from match.target.exec_module import ExecModule, ModuleLib
from match.partition.partitioning_pattern import PartitioningPattern
from match.target.exec_module import ComputationalApis, MemoryApis
from match_arcane.match.target.memory_inst import MemoryInst
from tvm.relay.dataflow_pattern import wildcard, is_op

class Carus(ExecModule):
    def __init__(self):
        super().__init__(
            name = "carus",
            libs_required = {
                "carus_helper": ModuleLib(name="carus_helper", base_path=os.path.dirname(__file__)+"/../libs/carus_helper")
            },
        )
    
    def zigzag_optimal_spatial_mapping_def(self, match_node = None, pattern_name = "conv2d"):
        return [("K", 128)]
    
    def partitioning_patterns(self):
        def dense_int32_pt():
            return is_op("nn.dense")(wildcard(), wildcard())
        
        def check_int32_only(node):
            dense = node
            return dense.args[1].checked_type.dtype=="int32"

        return [
            PartitioningPattern(name="dense", pattern=dense_int32_pt, additional_checks=check_int32_only)
        ]
    
    def module_memories(self):
        return [
            MemoryInst(name="ARCANE_L1_MEM", size=self.ARCANE_L1_MEMORY_SIZE),
        ]

    def comp_apis_def(self, computational_apis: ComputationalApis=None, pattern_name = "conv2d"):
        computational_apis.compute_tile = "carus_compute_wrapper"
        return computational_apis
    
    def mem_apis_def(self, memory_apis: MemoryApis=None, pattern_name = "conv2d"):
        memory_apis.mem_transfer = "carus_mem_transfer"
        memory_apis.init_memory["L1_MEM"] = "carus_l1_mem_init"
        return memory_apis