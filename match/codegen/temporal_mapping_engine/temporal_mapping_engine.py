from abc import ABC,abstractmethod
from typing import Any, Dict,List,Type

from match.codegen.layer_data import LayerData
from match.target.exec_module import ExecModule

class TemporalMappingEngine(ABC):
    """
    Abstract base class for a temporal engine
    """
    def __init__(self,exec_module:ExecModule=None,pattern_name:str="",layer_data:LayerData=None):
        self.exec_module=exec_module
        self.pattern_name=pattern_name
        self.temporal_mapping=[]
        self.layer_data=layer_data
        self.energy=0
        self.latency=0

    @abstractmethod
    def transform_workload_for_engine(self):
        raise NotImplementedError()
    
    @abstractmethod
    def generate_temporal_mapping(self):
        raise NotImplementedError()
    
    @abstractmethod
    def transform_temporal_mapping(self):
        raise NotImplementedError()
    
    def get_temporal_mapping(self):
        return self.temporal_mapping

    def get_latency(self):
        return self.latency

    def get_energy(self):
        return self.energy