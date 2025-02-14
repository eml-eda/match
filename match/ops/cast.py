import numpy as np
from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAddExpr, MatchAssignExpr, MatchEmptyExpr, MatchExpr, MatchLtExpr, MatchPrimitiveExpr, MatchTensorExpr, MatchTernaryExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule
from match.utils.utils import numpy_dtype_to_c_type
from numpy import typing as npt

class MatchOpCast(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., cast_dtype: npt.DTypeLike=np.dtype("int8"), **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Cast", **kwargs)
        self.cast_dtype = cast_dtype
        self.op_code = 7

    def basic_schedules(self):
        output, activations = self.outs[0], self.vars[0]
        loops = []
        for dim in output.dims:
            loops.append(MatchLoop(dim=dim, size=dim.size, name=dim.name, init_instrs=[], instrs=[]))
        loops[-1].instrs.append(MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),
                                           rhs_expr=MatchInstr(lhs_expr=MatchExpr(name=f"({numpy_dtype_to_c_type(self.cast_dtype)})"),eq_expr=MatchEmptyExpr(),rhs_expr=MatchTensorExpr(tensor=activations))))
        basic_cast_schedule = MatchSchedule(
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
        return [basic_cast_schedule]