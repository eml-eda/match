from match.ops.op import MatchOp

class MatchOpAdd(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Add", **kwargs)
        self.op_code = 5