from match.target.target import MatchTarget
from match.target.memory_inst import MemoryInst
import os

from match.transform.requant import MatchRequantRewriter
from .modules.carus.carus import Carus

ARCANE_L2_MEMORY_SIZE = 8*32*1024

class Arcane(MatchTarget):
    def __init__(self):
        super().__init__(
            exec_modules = [
                Carus()
            ]
            , name = "arcane",
        )
        self.makefile_path = os.path.dirname(__file__)+"/arcane_lib/Makefile"
        self.tvm_runtime_include_path = os.path.dirname(__file__)+"/arcane_lib/tvm_runtime.h"
        self.tvm_runtime_src_path = os.path.dirname(__file__)+"/arcane_lib/tvm_runtime.c"
        self.init_funcs = ["carus_helper_init_l1_mem"]
        self.include_list = ["carus_helper/carus_helper"]

    def network_transformations(self, opts):
        return [
            ("requant", MatchRequantRewriter()),
        ]

    def host_memories(self):
        return [
            MemoryInst(name="ARCANE_L2_MEM", k_bytes=ARCANE_L2_MEMORY_SIZE),
        ]