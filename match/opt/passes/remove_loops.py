

from match.opt.passes.schedule_pass import MatchScheduleOptPass
from match.schedule.schedule import MatchSchedule


class MatchRemoveLoops(MatchScheduleOptPass):
    def __call__(self, schedule: MatchSchedule) -> MatchSchedule:
        new_blocks = []
        blocks_to_del = []
        for block_idx in range(len(schedule.blocks)):
            for loop_idx in range(len(schedule.blocks[block_idx].loops)-1,-1,-1):
                if schedule.blocks[block_idx].loops[loop_idx].size==1:
                    if loop_idx == 0:
                        # move instrs to block
                        schedule.blocks[block_idx].init_instrs+=schedule.blocks[block_idx].loops[loop_idx].init_instrs
                        schedule.blocks[block_idx].instrs=schedule.blocks[block_idx].loops[loop_idx].instrs+schedule.blocks[block_idx].instrs
                    else:
                        # move instrs to prev loop
                        schedule.blocks[block_idx].loops[loop_idx-1].init_instrs+=schedule.blocks[block_idx].loops[loop_idx].init_instrs
                        schedule.blocks[block_idx].loops[loop_idx-1].instrs=schedule.blocks[block_idx].loops[loop_idx].instrs+schedule.blocks[block_idx].loops[loop_idx-1].instrs
                    # delete loop
                    del schedule.blocks[block_idx].loops[loop_idx]
            if len(schedule.blocks[block_idx].loops)==0:
                # move all instr to schedule and remove block
                schedule.init_instrs+=schedule.blocks[block_idx].init_instrs
                schedule.instrs+=schedule.blocks[block_idx].instrs
                blocks_to_del.append(schedule.blocks[block_idx])
            else:
                new_blocks.append(schedule.blocks[block_idx])
        for block_idx in range(len(blocks_to_del)-1,-1,-1):
            del blocks_to_del[block_idx]
        schedule.blocks = new_blocks
        return schedule
                