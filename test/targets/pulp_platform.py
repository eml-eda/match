import os
from match.target.target import MatchTarget
from .modules.pulp_cluster.pulp_cluster import PulpCluster

# pulp config
PULP_CORES = 8
L1_SCRATCHPAD_KB_SIZE = 64
L2_SHARED_MEM_KB_SIZE = 512
L3_FLASH_KB_SIZE = 8912
ASYNC_DMA = False

class PulpPlatform(MatchTarget):
    def __init__(self):
        super(PulpPlatform,self).__init__([
            PulpCluster(
                num_cores=PULP_CORES,
                l1_kb_size=L1_SCRATCHPAD_KB_SIZE,
                l2_kb_size=L2_SHARED_MEM_KB_SIZE,
                l3_kb_size=L3_FLASH_KB_SIZE,
                async_dma=ASYNC_DMA
            ),
        ],name="pulp_platform")
        self.cpu_type = "riscv_cpu"
        self.static_mem_plan = False
        self.static_mem_plan_algorithm = "hill_climb"
        self.makefile_path = os.path.dirname(__file__)+"/pulp_config_lib/Makefile"
        self.tvm_runtime_include_path = os.path.dirname(__file__)+"/pulp_config_lib/tvm_runtime.h"
        self.tvm_runtime_src_path = os.path.dirname(__file__)+"/pulp_config_lib/tvm_runtime.c"
        self.crt_config_path = os.path.dirname(__file__)+"/pulp_config_lib/crt_config.h"
        self.timestamp_to_ms = ""
        self.timestamp_type = "int"
        self.start_get_timestamp_api = "start_match_perf_counter"
        self.end_get_timestamp_api = "stop_match_perf_counter"
        self.init_funcs = ["pulp_cluster_init"]
        self.clean_funcs = ["pulp_cluster_close"]
        self.include_list = ["pulp_cluster/pulp_rt_profiler_wrapper","pmsis",
                             "pulp_cluster/cluster_dev","pulp_cluster/dory_dma",
                             "pulp_cluster/cluster_lib"]
        self.alloc_fn = "malloc_wrapper"
        self.free_fn = "free_wrapper"
        self.allocate_ext_mem = "pulp_init_ram"
        self.load_file_to_ext_mem_fn = "pulp_load_file"
        self.load_to_ext_mem_fn = "pulp_memcpy_to_ram"
        self.load_from_ext_mem_fn = "pulp_memcpy_from_ram"
        self.free_external_mem = "pulp_shutdown_ram"
        self.soc_memory_bytes = L2_SHARED_MEM_KB_SIZE*1024