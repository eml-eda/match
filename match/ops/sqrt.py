from match.ops.op import MatchOp

class MatchOpSqrt(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Sqrt", **kwargs)
        self.op_code = 12

    def basic_schedules(self):
        # TODO: Implement basic scheduling for sqrt operation
        return []
