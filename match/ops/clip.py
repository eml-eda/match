import numpy as np
from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAddExpr, MatchAssignExpr, MatchExpr, MatchGtExpr, MatchLtExpr, MatchPrimitiveExpr, MatchTensorExpr, MatchTernaryExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpClip(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., clip_min: int=0, clip_max: int=256, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Clip", **kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.op_code = 6

    def basic_schedules(self):
        output, activations = self.outs[0], self.vars[0]
        loops = []
        for dim in output.dims:
            loops.append(MatchLoop(dim=dim, size=dim.size, name=dim.name, init_instrs=[], instrs=[]))
        val = MatchPrimitiveExpr(name="val_to_clip",dtype=activations.dtype,init_expr=MatchTensorExpr(tensor=activations))
        loops[-1].init_instrs.append(MatchInstr(lhs_expr=val))
        loops[-1].instrs.append(MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),
                                           rhs_expr=MatchTernaryExpr(if_expr=MatchInstr(lhs_expr=val,eq_expr=MatchLtExpr(),rhs_expr=MatchPrimitiveExpr(name="clip_min",dtype=np.dtype("int32"),const=True,val=self.clip_min)),
                                                                     then_expr=MatchPrimitiveExpr(name="clip_min",dtype=np.dtype("int32"),const=True,val=self.clip_min),
                                                                     else_expr=MatchTernaryExpr(if_expr=MatchInstr(lhs_expr=val,eq_expr=MatchGtExpr(),rhs_expr=MatchPrimitiveExpr(name="clip_max",dtype=np.dtype("int32"),const=True,val=self.clip_max)),
                                                                                                then_expr=MatchPrimitiveExpr(name="clip_max",dtype=np.dtype("int32"),const=True,val=self.clip_max),
                                                                                                else_expr=val))))
        basic_clip_schedule = MatchSchedule(
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
        return [basic_clip_schedule]