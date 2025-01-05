

from typing import List
from match.dim.dim import MatchDim
from match.schedule.instr import MatchInstr
from match.schedule.mem_transfer import MatchMemTransfer

class MatchLoop:
    def __init__(self, dim: MatchDim, size: int=0, step: int=1, name: str="width",
                 mem_transfers: List[MatchMemTransfer]=[], init_instrs: List[MatchInstr]=[],
                 instrs: List[MatchInstr]=[],) -> None:
        self.name = ""
        self.size = size
        self.name = name
        self.dim = dim
        self.mem_transfers = mem_transfers
        # TODO: fix the step calculation
        self.step = step
        self.init_instrs = init_instrs
        self.instrs = instrs