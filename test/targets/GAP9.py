import os
from match.target.memory_inst import MemoryInst
from match.target.target import MatchTarget
from match.transform.layout import MatchLayoutNCHWtoNHWC, MatchLayoutNCHWtoNHWCTVM
from match.transform.requant import MatchRequantRewriter
from .modules.ne16_accelerator.accelerator import NE16Accelerator
from .modules.pulp_cluster.pulp_cluster import PulpCluster
from tvm import relay

# pulp config
PULP_CORES = 8
L1_SCRATCHPAD_KB_SIZE = 90
L2_SHARED_MEM_KB_SIZE = 1496
L3_FLASH_KB_SIZE = 89128
ASYNC_DMA = False

class GAP9(MatchTarget):
    def __init__(self):
        super(GAP9,self).__init__([
            PulpCluster(
                num_cores=PULP_CORES,
                l1_kb_size=L1_SCRATCHPAD_KB_SIZE,
                l2_kb_size=L2_SHARED_MEM_KB_SIZE,
                l3_kb_size=L3_FLASH_KB_SIZE,
                async_dma=ASYNC_DMA
            ),
            NE16Accelerator(
                num_cores=PULP_CORES,
                l1_kb_size=L1_SCRATCHPAD_KB_SIZE,
                l2_kb_size=L2_SHARED_MEM_KB_SIZE,
                l3_kb_size=L3_FLASH_KB_SIZE,
                async_dma=ASYNC_DMA
            )
        ],name="GAP9")
        self.set_target_host()
        self.set_paths()
        self.set_apis()
        self.soc_memory_bytes = L2_SHARED_MEM_KB_SIZE*1024

    def set_target_host(self):
        self.cpu_type = "riscv_cpu"

    def set_paths(self):
        self.makefile_path = os.path.dirname(__file__)+"/pulp_config_lib/Makefile.GAP9"
        self.tvm_runtime_include_path = os.path.dirname(__file__)+"/pulp_config_lib/tvm_runtime.h"
        self.tvm_runtime_src_path = os.path.dirname(__file__)+"/pulp_config_lib/tvm_runtime.c"
        self.crt_config_path = os.path.dirname(__file__)+"/pulp_config_lib/crt_config.h"
        self.include_list = [
            "pulp_utils/pulp_rt_profiler_wrapper",
            "pmsis",
            "pulp_cluster/cluster_dev",
            "pulp_mem/dma",
            "pulp_cluster/cluster",
            "pulp_mem/ram",
        ]

    def set_apis(self):
        # profiling ones
        self.start_get_timestamp_api = "start_match_perf_counter"
        self.end_get_timestamp_api = "stop_match_perf_counter"
        self.timestamp_to_ms = ""
        self.timestamp_type = "int"
        # initialization and cleaning
        self.init_funcs = ["pulp_cluster_init"]
        self.clean_funcs = ["pulp_cluster_close"]
        # memory management ones
        self.alloc_fn = "malloc_wrapper"
        self.free_fn = "free_wrapper"
        # external memory management
        self.allocate_ext_mem = "pulp_init_ram"
        self.load_file_to_ext_mem_fn = "pulp_load_file"
        self.load_to_ext_mem_fn = "pulp_memcpy_to_ram"
        self.load_from_ext_mem_fn = "pulp_memcpy_from_ram"
        self.free_external_mem = "pulp_shutdown_ram"

    def network_transformations(self, opts):
        return [
            ("requant", MatchRequantRewriter()),
            ("layout", MatchLayoutNCHWtoNHWCTVM),
        ]
    
    def host_memories(self):
        return [
            MemoryInst(name="L2_SHARED_MEM",k_bytes=L2_SHARED_MEM_KB_SIZE),
        ]