
from match.ops.op import MatchOp

class MatchOpDense(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., inp_features:int=1, out_features: int=1, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Dense", **kwargs)
        self.inp_features = inp_features
        self.out_features = out_features
        self.op_code = 1