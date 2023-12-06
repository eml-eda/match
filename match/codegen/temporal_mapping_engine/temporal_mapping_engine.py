from abc import ABC,abstractmethod
from typing import Any, Dict,List,Type

from match.codegen.layer_data import LayerData
from match.hwmodel.hwmodel import HwModel

class TemporalMappingEngine(ABC):
    """
    Abstract base class for a temporal engine
    """
    def __init__(self,workload:Dict[str,Any]={},hwmodel:HwModel=None,pattern_name:str="",layer_data:LayerData=None):
        self.workload=workload
        self.hwmodel=hwmodel
        self.pattern_name=pattern_name
        self.temporal_mapping=[]
        self.layer_data=layer_data

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