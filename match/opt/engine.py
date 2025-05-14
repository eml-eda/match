from abc import ABC,abstractmethod
from match.node.node import MatchNode
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ExecModule

class ScheduleEngine(ABC):
    """
    Abstract base class for a temporal engine
    """
    def __init__(self, target=None, exec_module: ExecModule=None,
                 pattern_name: str="", match_node: MatchNode=None):
        self.target = target
        self.exec_module = exec_module
        self.pattern_name = pattern_name
        self.match_node = match_node
        self.schedule: MatchSchedule = None
        self.energy = 0
        self.latency = 0

    @abstractmethod
    def transform_schedule_for_engine(self):
        raise NotImplementedError()
    
    @abstractmethod
    def generate_schedule(self):
        raise NotImplementedError()
    
    @abstractmethod
    def transform_schedule(self):
        raise NotImplementedError()
    
    def get_schedule(self):
        if self.schedule is None:
            raise ValueError("Schedule not generated yet.")
        if self.schedule.exec_module is None:
            self.schedule.exec_module = self.exec_module
        return self.schedule

    def get_latency(self):
        return self.latency

    def get_energy(self):
        return self.energy