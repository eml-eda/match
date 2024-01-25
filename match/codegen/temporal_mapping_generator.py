import numpy as np
from match.codegen.temporal_mapping_engine import get_temporal_mapping_engine
from match.codegen.workload_parser import WorkloadParser
# TVM imports
import tvm
from math import ceil
from collections import OrderedDict
import copy
from typing import Any, Dict,List,Type

from match.target.exec_module import ExecModule

class TemporalMappingGenerator:
    def __init__(self,node:tvm.ir.IRModule,args_list:List=[],exec_module:ExecModule=None,pattern_name:str="",partitioned:bool=False,pattern_inst=None,temporal_mapping_engine:str="zigzag"):
        self.node=node
        self.args_list=args_list
        self.exec_module=exec_module
        self.pattern_name=pattern_name
        self.partitioned=partitioned
        self.workload_parser=WorkloadParser(node=self.node,args_list=self.args_list,exec_module=self.exec_module,pattern_name=self.pattern_name,partitioned=self.partitioned,pattern_inst=pattern_inst)
        self.temporal_mapping_engine_class=get_temporal_mapping_engine(temporal_mapping_engine)
        self.workload=dict()
        self.temporal_mapping=[]
        self.layer_data=None

    def generate_workload(self):
        self.workload_parser.visit()
        self.layer_data=self.workload_parser.get_layer_data()

    def set_exec_module_for_layer(self):
        self.exec_module.match_optimal_spatial_mapping(pattern_name=self.pattern_name,dim_sizes=
                                                 self.layer_data.layer_attrs["loop_sizes"],layer_attrs=self.layer_data.layer_attrs)
        self.exec_module.match_memories(self.pattern_name,self.layer_data.operands)
        self.platform_memories=self.exec_module.platform_memories
        self.spatial_mapping=self.exec_module.spatial_mapping(dim_sizes=self.layer_data.layer_attrs["loop_sizes"],
                                                                         pattern_name=self.pattern_name,
                                                  operands=self.layer_data.operands,layer_attrs=self.layer_data.layer_attrs)
        self.cost_model=self.exec_module.cost_model()
        self.exec_module.match_specific_pattern(pattern_name=self.pattern_name,dim_sizes=
                                                self.layer_data.layer_attrs["loop_sizes"],layer_attrs=self.layer_data.layer_attrs)
        self.layer_data.specific_pattern=self.exec_module.specific_pattern
        self.exec_module.match_layout_operand(pattern_name=self.pattern_name,specific_pattern=self.layer_data.specific_pattern)
        self.optimal_spatial_mapping=self.exec_module.optimal_spatial_mapping
        
    def generate_temporal_mapping(self):
        temporal_mapping_engine=self.temporal_mapping_engine_class(self.exec_module,self.pattern_name,layer_data=self.layer_data)
        temporal_mapping_engine.transform_workload_for_engine()
        try:
            temporal_mapping_engine.generate_temporal_mapping(spatial_mapping=self.spatial_mapping,platform_memories=self.platform_memories,
                                                          optimal_spatial_mapping=self.optimal_spatial_mapping,cost_model=self.cost_model)
        except Exception as exc:
            raise Exception("No valid loop ordering found")
        temporal_mapping_engine.transform_temporal_mapping()
        self.temporal_mapping=temporal_mapping_engine.get_temporal_mapping()
        self.latency=temporal_mapping_engine.get_latency()
        self.energy=temporal_mapping_engine.get_energy()
    
    def constraint_temporal_mapping(self):
        self.temporal_mapping=self.exec_module.adjust_temporal_mapping(self.temporal_mapping,self.layer_data)

    def get_temporal_mapping(self):
        return self.temporal_mapping
    
    def get_layer_data(self):
        return self.layer_data

    def get_latency(self):
        return self.latency

    def get_energy(self):
        return self.energy