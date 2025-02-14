from typing import List
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop


class MatchBlock:
    def __init__(self,loops: List[MatchLoop]=[],
                 init_instrs: List[MatchInstr]=[],
                 instrs: List[MatchInstr]=[],
                 backend: str="MATCH",
                 num_buffers_for_computation: int=1,
                 parallel_execution: bool=False,
                 num_tasks: int=1
                ) -> None:
        self.loops = loops
        self.init_instrs = init_instrs
        self.instrs = instrs
        self.backend = backend
        self.num_buffers_for_computation = num_buffers_for_computation
        self.parallel_execution = parallel_execution
        self.num_tasks = num_tasks

    @property
    def loop_idx_end_sw_controlled_loads(self):
        idx_end = -1
        for idx,lp in enumerate(self.loops):
            if any([mt.sw_controlled for mt in lp.mem_transfers]):
                idx_end = idx
        return idx_end