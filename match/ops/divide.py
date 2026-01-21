from match.ops.op import MatchOp

class MatchOpDivide(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., axis: int=-1, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Divide", **kwargs)
        self.axis = axis
        self.op_code = 13

    def basic_schedules(self):
        # TODO: Implement basic scheduling for divide operation
        return []
