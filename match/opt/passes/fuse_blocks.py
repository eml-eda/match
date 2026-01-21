from match.opt.passes.schedule_pass import MatchScheduleOptPass
from match.schedule.block import MatchBlock
from match.schedule.expr import MATCH_ASSIGN_EXPRS_CLS, MatchTensorExpr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

# --- Helpers ---

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


def _collect_reads_writes(block: MatchBlock):
    reads = set()
    writes = set()

    def walk_expr(expr):
        if isinstance(expr, MatchTensorExpr):
            reads.add(expr.name)
        for v in getattr(expr, "__dict__", {}).values():
            if hasattr(v, "__dict__"):
                walk_expr(v)

    for instr in _iter_block_instrs(block):
        lhs = getattr(instr, "lhs_expr", None)
        eq = getattr(instr, "eq_expr", None)
        rhs = getattr(instr, "rhs_expr", None)
        if isinstance(lhs, MatchTensorExpr) and isinstance(eq, MATCH_ASSIGN_EXPRS_CLS):
            writes.add(lhs.name)
        if rhs is not None:
            walk_expr(rhs)
    return reads, writes


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


def _loops_identical(a: MatchBlock, b: MatchBlock) -> bool:
    if len(a.loops) != len(b.loops):
        return False
    for la, lb in zip(a.loops, b.loops):
        if la.dim.name != lb.dim.name or la.size != lb.size:
            return False
    return True


def _merge_blocks_with_identical_loops(a: MatchBlock, b: MatchBlock) -> MatchBlock:
    # Merge init/instrs at block level
    init_instrs = a.init_instrs + b.init_instrs
    instrs = a.instrs + b.instrs
    # Merge per-loop init/instrs
    loops = []
    for la, lb in zip(a.loops, b.loops):
        loops.append(
            MatchLoop(
                dim=la.dim,
                size=la.size,
                step=la.step,
                name=la.name,
                mem_transfers=(la.mem_transfers or []) + (lb.mem_transfers or []),
                init_instrs=(la.init_instrs or []) + (lb.init_instrs or []),
                instrs=(la.instrs or []) + (lb.instrs or []),
            )
        )
    return MatchBlock(
        loops=loops,
        init_instrs=init_instrs,
        instrs=instrs,
        backend=a.backend,
        num_buffers_for_computation=max(a.num_buffers_for_computation, b.num_buffers_for_computation),
        parallel_execution=a.parallel_execution or b.parallel_execution,
        num_tasks=max(a.num_tasks, b.num_tasks),
    )


class MatchFuseBlocksScheduleOptPass(MatchScheduleOptPass):
    """Safely fuse adjacent blocks when possible.

    We fuse two consecutive blocks A,B if:
      - Both are element-wise: no accumulations, no SW-controlled mem transfers
      - Their loop nests are identical (same dims order and sizes)
      - Their write sets are disjoint (no clobbering)
    Fusion merges instructions at block and loop levels.
    """

    def __call__(self, schedule: MatchSchedule) -> MatchSchedule:
        i = 0
        new_blocks = []
        while i < len(schedule.blocks):
            if i < len(schedule.blocks) - 1:
                a = schedule.blocks[i]
                b = schedule.blocks[i + 1]
                if not _has_accumulation(a) and not _has_accumulation(b) \
                   and not _has_sw_controlled_mem(a) and not _has_sw_controlled_mem(b) \
                   and _loops_identical(a, b):
                    _, a_w = _collect_reads_writes(a)
                    _, b_w = _collect_reads_writes(b)
                    if a_w.isdisjoint(b_w):
                        fused = _merge_blocks_with_identical_loops(a, b)
                        new_blocks.append(fused)
                        i += 2
                        continue
            new_blocks.append(schedule.blocks[i])
            i += 1
        schedule.blocks = new_blocks
        return schedule
