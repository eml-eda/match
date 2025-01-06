

from match.opt.basic import BasicEngine
from match.opt.passes.remove_loops import MatchRemoveLoops
from match.opt.passes.schedule_pass import MatchScheduleOptPassContext, MatchScheduleOptSequentialPasses
import traceback as tb

class BasicPlusEngine(BasicEngine):
    def generate_schedule(self):
        super().generate_schedule()
        pipeline = []
        pipeline.append(MatchRemoveLoops())
        seq = MatchScheduleOptSequentialPasses(pipeline)
        with MatchScheduleOptPassContext():
            try:
                self.schedule = seq(self.schedule)
            except Exception as exc:
                print(f"[BASIC PLUS ENGINE] Exception occurred {tb.format_exception(exc)}")