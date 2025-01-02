from match.ops.op import MatchOp

class MatchOpBiasAdd(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., axis: int=-1, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="BiasAdd", **kwargs)
        self.axis = axis
        self.op_code = 2