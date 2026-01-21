from typing import Set

from match.opt.passes.schedule_pass import MatchScheduleOptPass
from match.schedule.mem_transfer import MatchMemTransfer
from match.schedule.schedule import MatchSchedule
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchTensorExpr, MATCH_ASSIGN_EXPRS_CLS


def _collect_reads(block: MatchBlock) -> Set[str]:
    reads = set()

    def walk_expr(expr):
        if isinstance(expr, MatchTensorExpr):
            reads.add(expr.name)
        for v in getattr(expr, "__dict__", {}).values():
            if hasattr(v, "__dict__"):
                walk_expr(v)

    # Traverse all instructions
    for instr in block.init_instrs + block.instrs:
        walk_expr(getattr(instr, "rhs_expr", None))
    for lp in block.loops:
        for instr in lp.init_instrs + lp.instrs:
            walk_expr(getattr(instr, "rhs_expr", None))
    return reads


def _existing_transfers(block: MatchBlock) -> Set[str]:
    names = set()
    for lp in block.loops:
        for mt in getattr(lp, "mem_transfers", []) or []:
            tensor = getattr(mt, "tensor", None)
            if tensor is not None:
                names.add(tensor.name)
    return names


def _choose_mem_names(schedule: MatchSchedule):
    em = getattr(schedule, "exec_module", None)
    if em is None:
        return ("L2_MEM", "L1_SCRATCHPAD")
    try:
        mems = em.module_memories() or []
        if not mems:
            return ("L2_MEM", "L1_SCRATCHPAD")
        low = mems[0].name
        top = None
        for m in reversed(mems):
            if getattr(m, "external", False):
                top = m.name
                break
        if top is None:
            top = mems[-1].name
        return (top, low)
    except Exception:
        return ("L2_MEM", "L1_SCRATCHPAD")


class MatchSetMemTransfersPass(MatchScheduleOptPass):
    """Add memory transfers for tensors read in each block.

    Inserts a MatchMemTransfer for each read tensor at the outermost loop
    of the block, if not already present. This is conservative and
    correctness-preserving; further scheduling may move/split transfers.
    """

    def __call__(self, schedule: MatchSchedule) -> MatchSchedule:
        top_mem, low_mem = _choose_mem_names(schedule)
        for blk in schedule.blocks:
            if not blk.loops:
                continue
            reads = _collect_reads(blk)
            existing = _existing_transfers(blk)
            target_loop = blk.loops[0]
            for tname, tensor in schedule.tensors.items():
                if tname in reads and tname not in existing:
                    target_loop.mem_transfers = list(getattr(target_loop, "mem_transfers", []) or [])
                    target_loop.mem_transfers.append(
                        MatchMemTransfer(tensor=tensor, top_mem=top_mem, mem=low_mem, sw_controlled=True)
                    )
        return schedule
