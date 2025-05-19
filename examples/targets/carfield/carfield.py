import os
from match.target.memory_inst import MemoryInst
from match.target.target import MatchTarget
from match.transform.layout import MatchLayoutNCHWtoNHWC, MatchLayoutNCHWtoNHWCTVM
from match.transform.requant import MatchRequantRewriter
from pulp_cluster import PulpCluster
from tvm import relay

# pulp config
PULP_CORES = 8
#L1_SCRATCHPAD_KB_SIZE = 128*1024
#L2_SHARED_MEM_KB_SIZE = 256*1024

L1_SCRATCHPAD_KB_SIZE = 32
L2_SHARED_MEM_KB_SIZE = 256*1024
L3_FLASH_KB_SIZE = 8*1024*1024

ASYNC_DMA = False

class Carfield(MatchTarget):
    def __init__(self):
        super(Carfield,self).__init__([
            PulpCluster(
                num_cores=PULP_CORES,
                l1_kb_size=L1_SCRATCHPAD_KB_SIZE,
                l2_kb_size=L2_SHARED_MEM_KB_SIZE,
                l3_kb_size=L3_FLASH_KB_SIZE,
                async_dma=ASYNC_DMA
            )
        ],name="carfield")
        self.set_target_host()
        self.set_paths()
        self.set_apis()

    def set_target_host(self):
        self.cpu_type = "riscv_cpu -march=riscv64"

    def set_paths(self):
        self.makefile_path = os.path.dirname(__file__)+"/config/Makefile"
        self.other_files_to_copy = [
            os.path.dirname(__file__)+"/config/link.ld",
            os.path.dirname(__file__)+"/config/link_pulpd.ld",
            os.path.dirname(__file__)+"/config/elf2payload.py"
        ]
        self.tvm_runtime_include_path = os.path.dirname(__file__)+"/config/tvm_runtime.h"
        self.tvm_runtime_src_path = os.path.dirname(__file__)+"/config/tvm_runtime.c"
        self.crt_config_path = os.path.dirname(__file__)+"/config/crt_config.h"
        self.include_list = [
            "carfield_lib/carfield"
        ]

    def set_apis(self):
        # profiling ones
        self.start_get_timestamp_api = "start_match_perf_counter"
        self.end_get_timestamp_api = "stop_match_perf_counter"
        self.timestamp_to_ms = ""
        self.timestamp_type = "int"
        # initialization and cleaning
        self.init_funcs = ["carfield_init"]
        self.clean_funcs = ["carfield_shutdown"]
        # memory management ones
        self.alloc_fn = "" # should use stack
        self.free_fn = ""
        # external memory management
        self.allocate_ext_mem = "pulp_init_ram"
        self.load_file_to_ext_mem_fn = "pulp_load_file"
        self.load_to_ext_mem_fn = "pulp_memcpy_to_ram"
        self.load_from_ext_mem_fn = "pulp_memcpy_from_ram"
        self.free_external_mem = "pulp_shutdown_ram"
        # offload dma
        self.offload_dma_fn = "handle_host_dma_transfer"
        self.print_fn = "mini_printf"

    def network_transformations(self, opts):
        return [
            ("requant", MatchRequantRewriter()),
            ("layout", MatchLayoutNCHWtoNHWCTVM),
            ("folded", relay.transform.FoldConstant()),
        ]
    
    def host_memories(self):
        return [
            MemoryInst(name="L2_SHARED_MEM",k_bytes=L2_SHARED_MEM_KB_SIZE),
        ]