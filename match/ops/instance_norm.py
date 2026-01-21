from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import (
    MatchAssignExpr, MatchSubExpr, MatchMulExpr, MatchDivExpr,
    MatchAddExpr, MatchPrimitiveExpr, MatchTensorExpr
)
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpInstanceNorm(MatchOp):
    """Instance Normalization.

    Expected tensors ordering in parser:
      vars: [input]
      consts (optional): [gamma, beta]
      outs: [output]
    Attributes:
      epsilon: small float to avoid div by zero
      momentum: (kept for API consistency, not used in pure inference norm)
    """
    def __init__(self, out_arr=..., var_arr=..., const_arr=..., epsilon: float=1e-5, momentum: float=0.9, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="InstanceNorm", **kwargs)
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)
        self.op_code = 17

    def basic_schedules(self):
        # Simple reference schedule (not optimized):
        # For each instance (N*C*H*W) compute mean/var across spatial+channel dims or channel only?
        # Standard InstanceNorm: normalize each channel per sample over spatial dims (H,W)
        # Assuming layout NCHW.
        inp = self.vars[0]
        out = self.outs[0]
        gamma = self.consts[0] if len(self.consts) > 0 else None
        beta = self.consts[1] if len(self.consts) > 1 else None

        # For now just copy (identity) since proper scheduling primitives for reduction not yet implemented here.
        loops = []
        for dim in out.dims:
            loops.append(MatchLoop(dim=dim, size=dim.size, name=dim.name, init_instrs=[], instrs=[]))
        # identity fallback (should be replaced by actual normalization schedule once reduction ops available)
        loops[-1].instrs.append(MatchInstr(
            lhs_expr=MatchTensorExpr(tensor=out),
            eq_expr=MatchAssignExpr(),
            rhs_expr=MatchTensorExpr(tensor=inp)
        ))
        basic_schedule = MatchSchedule(
            blocks=[MatchBlock(loops=loops, init_instrs=[], instrs=[])],
            init_instrs=[],
            instrs=[],
        )
        return [basic_schedule]
