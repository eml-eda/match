from match.ops.op import MatchOp
from typing import List

class MatchOpReshape(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., newshape: List[int]=[], **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Reshape", **kwargs)
        self.newshape = newshape
        self.op_code = 15

    def basic_schedules(self):
        # TODO: Implement basic scheduling for reshape operation
        return []
