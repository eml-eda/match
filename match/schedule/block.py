from typing import List
from match.schedule.loop import MatchLoop


class MatchBlock:
    def __init__(self,loops: List[MatchLoop]=[]) -> None:
        self.loops = loops

    @property
    def loop_idx_end_sw_controlled_loads(self):
        return 0