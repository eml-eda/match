from match.ops.op import MatchOp

class MatchOpRepeat(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., repeats: int=1, axis: int=0, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Repeat", **kwargs)
        self.repeats = repeats
        self.axis = axis
        self.op_code = 14

    def basic_schedules(self):
        # TODO: Implement basic scheduling for repeat operation
        return []
