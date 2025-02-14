import numpy as np
from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAddExpr, MatchAssignExpr, MatchExpr, MatchPrimitiveExpr, MatchTensorExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpRightShift(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., right_shift: int=1, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="RightShift", **kwargs)
        self.right_shift = right_shift
        self.op_code = 5

    def basic_schedules(self):
        output, activations = self.outs[0], self.vars[0]
        loops = []
        for dim in output.dims:
            loops.append(MatchLoop(dim=dim, size=dim.size, name=dim.name, init_instrs=[], instrs=[]))
        loops[-1].instrs.append(MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),
                                           rhs_expr=MatchInstr(lhs_expr=MatchTensorExpr(tensor=activations),eq_expr=MatchExpr(name=">>"),
                                                                rhs_expr=MatchPrimitiveExpr(name="shift_val",const=True,dtype=np.dtype("int32"),val=self.right_shift))))
        basic_right_shift_schedule = MatchSchedule(
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
        return [basic_right_shift_schedule]