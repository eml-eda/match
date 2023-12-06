import numpy as np
from match.codegen.temporal_mapping_engine import get_temporal_mapping_engine
from match.codegen.workload_parser import WorkloadParser

# TVM imports
import tvm
from math import ceil
from collections import OrderedDict
import copy
from typing import Any, Dict,List,Type

from match.hwmodel.hwmodel import HwModel




class TemporalMappingGenerator:
    def __init__(self,mod:tvm.ir.IRModule,hwmodel:HwModel=None,pattern_name:str="",temporal_mapping_engine:str="zigzag"):
        self.mod=mod
        self.hwmodel=hwmodel
        self.pattern_name=pattern_name
        self.workload_parser=WorkloadParser(mod=self.mod,hwmodel=self.hwmodel,pattern_name=self.pattern_name)
        self.temporal_mapping_engine_class=get_temporal_mapping_engine(temporal_mapping_engine)
        self.workload=dict()
        self.temporal_mapping=[]
        self.layer_data=None

    def generate_workload(self):
        self.workload_parser.visit()
        self.workload=self.workload_parser.get_workload()
        self.layer_data=self.workload_parser.get_layer_data()
        #breakpoint()

        
    def generate_temporal_mapping(self):
        temporal_mapping_engine=self.temporal_mapping_engine_class(self.workload,self.hwmodel,self.pattern_name,layer_data=self.layer_data)
        temporal_mapping_engine.transform_workload_for_engine()
        self.hwmodel.optimal_spatial_mapping_def(workload_name=self.layer_data.workload_name,dim_sizes=
                                                 self.layer_data.layer_attrs["loop_sizes"])
        self.hwmodel.memories_def(self.layer_data.operands)
        self.optimal_spatial_mapping=self.hwmodel.optimal_spatial_mapping
        self.platform_memories=self.hwmodel.platform_memories
        self.spatial_mapping=self.mapping = self.hwmodel.spatial_mapping(dim_sizes=self.layer_data.layer_attrs["loop_sizes"],
                                                                         workload_name=self.layer_data.workload_name,
                                                  operands=self.layer_data.operands)
        self.cost_model=self.hwmodel.cost_model()
        temporal_mapping_engine.generate_temporal_mapping(spatial_mapping=self.spatial_mapping,platform_memories=self.platform_memories,
                                                          optimal_spatial_mapping=self.optimal_spatial_mapping,cost_model=self.cost_model)
        temporal_mapping_engine.transform_temporal_mapping()
        self.temporal_mapping=temporal_mapping_engine.get_temporal_mapping()
    
    def constraint_temporal_mapping(self):
        self.temporal_mapping=self.hwmodel.adjust_temporal_mapping(self.temporal_mapping,self.layer_data)

    def get_temporal_mapping(self):
        return self.temporal_mapping
    
    def get_layer_data(self):
        return self.layer_data