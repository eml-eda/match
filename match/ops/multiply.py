from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAssignExpr, MatchMulExpr, MatchPrimitiveExpr, MatchTensorExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpMultiply(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., axis: int=-1, multiplier: int=1, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Multiply", **kwargs)
        self.axis = axis
        self.multiplier = multiplier
        self.op_code = 8

    def basic_schedules(self):
        output, activations, biases = self.outs[0], self.vars[0], self.consts[0]
        loops = []
        for dim in output.dims:
            loops.append(MatchLoop(dim=dim, size=dim.size, name=dim.name, init_instrs=[], instrs=[]))
        scale_val = MatchPrimitiveExpr(name="mul_val",dtype=output.dtype, const=False, init_expr=MatchTensorExpr(tensor=biases))
        axis = -1
        if self.axis >= 0:
            axis = self.axis
        loops[axis].init_instrs.append(MatchInstr(lhs_expr=scale_val))
        loops[-1].instrs.append(MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),
                                           rhs_expr=MatchInstr(lhs_expr=MatchTensorExpr(tensor=activations),eq_expr=MatchMulExpr(),
                                                                rhs_expr=scale_val)))
        basic_scale_schedule = MatchSchedule(
            blocks=[
                MatchBlock(
                    loops=loops,
                    init_instrs=[],
                    instrs=[],
                )
            ],
            init_instrs=[],
            instrs=[],
        )
        return [basic_scale_schedule]