from typing import List
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop


class MatchBlock:
    def __init__(self,loops: List[MatchLoop]=[],
                 init_instrs: List[MatchInstr]=[],
                 instrs: List[MatchInstr]=[],) -> None:
        self.loops = loops
        self.init_instrs = init_instrs
        self.instrs = instrs

    @property
    def loop_idx_end_sw_controlled_loads(self):
        return 0