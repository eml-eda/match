

from match.opt.passes.schedule_pass import MatchScheduleOptPass
from match.schedule.block import MatchBlock
from match.schedule.expr import MATCH_ASSIGN_EXPRS_CLS, MatchTensorExpr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

def blocks_have_same_loops(block: MatchBlock, other_block: MatchBlock):
    for o_lp in other_block.loops:
        for instr in o_lp.init_instrs:
            if isinstance(instr.lhs_expr,MatchTensorExpr) and isinstance(instr.eq_expr,MATCH_ASSIGN_EXPRS_CLS):
                written_tensor_name = instr.lhs_expr.name


class MatchFuseBlocksScheduleOptPass(MatchScheduleOptPass):
    
    def __call__(self, schedule: MatchSchedule) -> MatchSchedule:
        block_idx = 0
        while block_idx<len(schedule.blocks)-1:
            block, other_block = schedule.blocks[block_idx], schedule.blocks[block_idx+1]
            block_loops_name_set = set([lp.name for lp in block.loops])
            other_block_loops_name_set = set([lp.name for lp in other_block])
            intersec = block_loops_name_set.intersection(other_block_loops_name_set)
            # simple fusion case block is superset
            if block_loops_name_set.issuperset(other_block_loops_name_set):
                order_of_loops = {lp.name:idx for idx,lp in enumerate(block.loops)}
                other_block_reordered_loops = []
                for lp in other_block.loops:
                    other_block_reordered_loops.insert(order_of_loops[lp.name],lp)
                
                # if block_war_data_dependencies(schedule.blocks[block_idx],schedule.blocks[block_idx+1]):
                #     loops = [lp for lp in schedule.blocks[block_idx].loops]
                #     loops += [lp for lp in schedule.blocks[block_idx+1] if lp.name not in set([b_lp.name for b_lp in schedule.blocks[block_idx]])]
            # simple fusion case block is subset
            elif block_loops_name_set.issuperset(other_block_loops_name_set):
                common_lps_no_instrs = [MatchLoop(dim=lp.dim,size=lp.size,step=lp.step,name=lp.name,
                                                  mem_transfers=lp.mem_transfers,init_instrs=[],instrs=[]) for lp in schedule.blocks[block_idx].loops if lp.name in intersec]
