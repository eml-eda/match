from abc import ABC,abstractmethod
from typing import Any, Dict,List,Type
from match.parser.parser import MatchParser
from match.target.exec_module import ExecModule

class ScheduleEngine(ABC):
    """
    Abstract base class for a temporal engine
    """
    def __init__(self,exec_module:ExecModule=None,pattern_name:str="",pattern_inst=None,match_node=None):
        self.exec_module = exec_module
        self.pattern_name = pattern_name
        self.pattern_inst = pattern_inst
        self.match_node = match_node
        self.schedule=None
        self.energy=0
        self.latency=0

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
        return self.schedule

    def get_latency(self):
        return self.latency

    def get_energy(self):
        return self.energy