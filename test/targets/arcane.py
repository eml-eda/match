from match.target.target import MatchTarget
from match.target.memory_inst import MemoryInst
from modules.carus.carus import Carus

ARCANE_L2_MEMORY_SIZE = 1024

class Arcane(MatchTarget):
    def __init__(self):
        super().__init__(
            exec_modules = [
                Carus()
            ]
            , name = "arcane",
        )
    
    def host_memories(self):
        return [
            MemoryInst(name="ARCANE_L2_MEM", size=ARCANE_L2_MEMORY_SIZE),
        ]