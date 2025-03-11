from match.node.node import MatchNode
from match.opt.basic import BasicEngine
from match.opt.basic_plus import BasicPlusEngine
from match.parser.relay import MatchRelayParser
# TVM imports
from match.schedule.schedule import MatchSchedule
import tvm
from typing import List

from match.target.exec_module import ExecModule

from match.opt.zigzag.zigzag import ZigZagEngine
from match.opt.engine import ScheduleEngine
    
SCHEDULE_ENGINE_MAP={
    "ZigZag":ZigZagEngine,
    "basic":BasicEngine,
    "basic_plus":BasicPlusEngine,
}

def get_schedule_engine(engine_name: str="ZigZag"):
    if engine_name not in SCHEDULE_ENGINE_MAP:
        return ScheduleEngine
    else:
        assert issubclass(SCHEDULE_ENGINE_MAP[engine_name],ScheduleEngine)
        return SCHEDULE_ENGINE_MAP[engine_name]


class ScheduleGenerator:
    def __init__(self, node: tvm.ir.IRModule, args_list: List=[],
                 target=None, exec_module: ExecModule=None,
                 pattern_name: str="conv2d", partitioned: bool=False, pattern_inst=None):
        self.node = node
        self.args_list = args_list
        self.target = target
        self.exec_module = exec_module
        self.pattern_name = pattern_name
        self.partitioned = partitioned
        self.pattern_inst = pattern_inst
        self.match_node = MatchNode()
        self.parser = MatchRelayParser(
            node=self.node, args_list=self.args_list, exec_module=self.exec_module,
            pattern_name=self.pattern_name, partitioned = self.partitioned,
            pattern_inst=self.pattern_inst, match_node=self.match_node
        )
        self.schedule_engine_classname = self.exec_module.schedule_engine
        self.schedule_engine_class = get_schedule_engine(self.exec_module.schedule_engine)
        self.schedule: MatchSchedule = None
        
    def parse(self):
        try:
            self.parser.visit()
        except Exception as exc:
            raise exc

    def generate(self):
        schedule_engine = self.schedule_engine_class(self.target, self.exec_module,self.pattern_name,self.match_node)
        schedule_engine.transform_schedule_for_engine()
        try:
            schedule_engine.generate_schedule()
        except Exception as exc:
            raise Exception(f"[SCHEDULER] No valid schedule found {exc}")
        schedule_engine.transform_schedule()
        self.schedule = schedule_engine.get_schedule()
        self.exec_module.set_buffers_for_schedule(self.match_node, self.schedule,
                                                  self.pattern_name, self.schedule_engine_classname)
        self.latency = schedule_engine.get_latency()
        self.energy = schedule_engine.get_energy()
    
    def apply_constraints(self):
        self.schedule = self.exec_module.constrain_schedule(self.schedule,self.match_node)
    
    def get_match_node(self):
        return self.match_node