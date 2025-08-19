from match.ops.op import MatchOp
from typing import Union, List

class MatchOpSum(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., axis: Union[int, List[int], None]=None, keepdims: bool=False, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Sum", **kwargs)
        self.axis = axis
        self.keepdims = keepdims
        self.op_code = 16

    def basic_schedules(self):
        # TODO: Implement basic scheduling for sum operation
        return []
