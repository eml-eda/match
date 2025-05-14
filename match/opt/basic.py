


from match.node.node import MatchNode
from match.opt.engine import ScheduleEngine
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ExecModule


class BasicEngine(ScheduleEngine):
    def __init__(self, target=None, exec_module: ExecModule=None,
                 pattern_name: str="", match_node: MatchNode=None
                 ):
        super(BasicEngine, self).__init__(target=target, exec_module=exec_module,
                                           pattern_name=pattern_name, match_node=match_node)
    
    def generate_schedule(self):
        # import pdb; pdb.set_trace()
        # breakpoint()
        schedule_blocks = []
        for op in self.match_node.ops.values():
            schedule_blocks += op.basic_schedules()[0].blocks
        # breakpoint()
        self.schedule = MatchSchedule(blocks=schedule_blocks,tensors=self.match_node.tensors,init_instrs=[],instrs=[])

    def transform_schedule_for_engine(self):
        pass

    def transform_schedule(self):
        pass
    