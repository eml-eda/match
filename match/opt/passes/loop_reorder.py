from match.opt.passes.schedule_pass import MatchScheduleOptPass
from match.schedule.block import MatchBlock
from match.schedule.expr import MATCH_ASSIGN_EXPRS_CLS, MatchTensorExpr
from match.schedule.schedule import MatchSchedule


def _iter_block_instrs(block: MatchBlock):
    for instr in block.init_instrs:
        yield instr
    for instr in block.instrs:
        yield instr
    for lp in block.loops:
        for instr in lp.init_instrs:
            yield instr
        for instr in lp.instrs:
            yield instr


def _has_accumulation(block: MatchBlock) -> bool:
    for instr in _iter_block_instrs(block):
        eq = getattr(instr, "eq_expr", None)
        if isinstance(eq, MATCH_ASSIGN_EXPRS_CLS):
            en = type(eq).__name__
            if en in ("MatchPlusEqExpr", "MatchMinusEqExpr"):
                return True
    return False


def _has_sw_controlled_mem(block: MatchBlock) -> bool:
    for lp in block.loops:
        if any(getattr(mt, "sw_controlled", False) for mt in getattr(lp, "mem_transfers", []) or []):
            return True
    return False


class MatchLoopReorderPass(MatchScheduleOptPass):
    """Heuristic loop reordering to improve locality for element-wise blocks.

    Sort loops by dimension size ascending (smaller inner) when safe.
    """

    def __call__(self, schedule: MatchSchedule) -> MatchSchedule:
        for blk in schedule.blocks:
            if _has_accumulation(blk) or _has_sw_controlled_mem(blk):
                continue
            blk.loops = sorted(list(blk.loops), key=lambda lp: getattr(lp.dim, "size", lp.size))
        return schedule
