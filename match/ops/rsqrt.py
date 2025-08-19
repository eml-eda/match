from match.ops.op import MatchOp

class MatchOpRsqrt(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Rsqrt", **kwargs)
        self.op_code = 19

    def basic_schedules(self):
        # TODO: Implement scheduling for rsqrt
        return []
