import os
from sauria import Sauria
from match.target.memory_inst import MemoryInst
from match.target.target import MatchTarget
from match.transform.layout import MatchLayoutNCHWtoNHWC, MatchLayoutNCHWtoNHWCTVM, RemoveLayoutTransformPass
from match.transform.requant import MatchRequantRewriter
from tvm import relay

__ASTRAL_DRAM_KB__ = 16*1024

class Astral(MatchTarget):
    def __init__(self):
        super(Astral,self).__init__(
            [
                Sauria()
            ],
            name="astral",
        )
        self.set_target_host()
        self.set_paths()
        self.set_apis()
        self.enable_device_parallelism = False
        self.async_off_chip_to_on_chip_dm = True

    def set_target_host(self):
        self.cpu_type = "riscv_cpu -march=riscv64"
        # self.params_data_on_chip_flag = "__attribute__((section(\".l2_data\"), ))"
        # self.params_data_off_chip_flag = "__attribute__((section(\".rodata.tvm\"), ))"
        self.params_off_chip_file = False

    def set_paths(self):
        self.makefile_path = os.path.dirname(__file__)+"/../isolde/config/Makefile"
        self.other_files_to_copy = [
            os.path.dirname(__file__)+"/../isolde/config/link_pulpd.ld",
            os.path.dirname(__file__)+"/../isolde/config/elf2payload.py",
            os.path.dirname(__file__)+"/../isolde/config/link.ld",
            os.path.dirname(__file__)+"/../isolde/config/correct_link_pulpd.ld",
        ]
        self.tvm_runtime_include_path = os.path.dirname(__file__)+"/../isolde/config/tvm_runtime.h"
        self.tvm_runtime_src_path = os.path.dirname(__file__)+"/../isolde/config/tvm_runtime.c"
        self.crt_config_path = os.path.dirname(__file__)+"/../isolde/config/crt_config.h"
        self.include_list = [
            "carfield_lib/carfield"
        ]
        self.include_list_quotes = [
            "util"
        ]

    def set_apis(self):
        # profiling ones
        self.start_get_timestamp_api = "astral_get_cycleno"
        self.end_get_timestamp_api = "astral_get_cycleno"
        self.timestamp_to_ms = ""
        self.timestamp_type = "uint64_t"
        # initialization and cleaning
        self.init_funcs = ["carfield_init"]
        self.clean_funcs = ["carfield_shutdown"]
        # memory management ones
        self.alloc_fn = "" # should use stack
        self.free_fn = ""
        # external memory management
        self.allocate_ext_mem = "carfield_init_ram"
        self.load_file_to_ext_mem_fn = "carfield_load_file_to_ram"
        self.load_to_ext_mem_fn = "carfield_memcpy_to_ram"
        self.load_from_ext_mem_fn = "carfield_memcpy_from_ram"
        self.free_external_mem = "carfield_free_ram"
        # offload dma
        self.offload_dma_fn = "handle_host_dma_transfer"
        self.print_fn = "mini_printf"
        self.timer_start_fn = "carfield_timer_start"
        self.timer_stop_fn = "carfield_timer_stop"
        self.fix_io_tensors_in_ext_mem = False
        # wait
        self.wait_eoc = "carfield_wait_eoc"
        self.wait_async_off_chip_transfer_fn = "carfield_wait_dma"

    def network_transformations(self, opts):
        return [
            ("requant", MatchRequantRewriter()),
            # ("layout", MatchLayoutNCHWtoNHWCTVM),
            # ("laybatchdense", RemoveLayoutTransformPass),
            ("folded", relay.transform.FoldConstant()),
        ]
    
    def host_memories(self):
        return [
            MemoryInst(name="__astral_dram__",k_bytes=__ASTRAL_DRAM_KB__),
        ]