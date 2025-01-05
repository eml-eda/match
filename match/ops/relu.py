from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAssignExpr, MatchGtExpr, MatchPrimitiveExpr, MatchTensorExpr, MatchTernaryExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpReLU(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="ReLU", **kwargs)
        self.op_code = 0

    def basic_schedules(self):
        output, activations = self.outs[0], self.vars[0]
        loops = []
        for dim in output.dims:
            loops.append(MatchLoop(dim=dim, size=dim.size, name=dim.name))
        loops[-1].instrs.append(MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),
                                           rhs_expr=MatchTernaryExpr(
                                               if_expr=MatchInstr(lhs_expr=MatchTensorExpr(tensor=activations),
                                                                  eq_expr=MatchGtExpr(),rhs_expr=MatchPrimitiveExpr(name="zero",dtype=output.dtype,const=True,val=0)),
                                                  else_expr=MatchTensorExpr(tensor=activations),
                                                  then_expr=MatchPrimitiveExpr(name="zero",dtype=output.dtype,const=True,val=0),
                                           )))
        basic_relu_schedule = MatchSchedule(
            blocks=[
                MatchBlock(
                    loops=loops
                )
            ]
        )
        return [basic_relu_schedule]