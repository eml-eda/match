from match.codegen.template import TemplateDataGenerator,TemplateEngine
from match.codegen.temporal_mapping_generator import TemporalMappingGenerator
from match.codegen.temporal_mapping_engine import get_temporal_mapping_engine

# TVM imports
import tvm
# ZigZag imports
from math import ceil
from mako.template import Template
from mako import exceptions
import functools
import operator
from collections import OrderedDict
import copy
from typing import Dict,List,Type
from match.matchutils import mock_func
from match.target import get_target

def get_code(mod: tvm.ir.IRModule,target_name:str="",pattern_name:str=""):
    target=get_target.get_target(target_name=target_name)
    temporal_mapping,layer_data,exec_module=target.get_layer_from_module(mod=mod,pattern_name=pattern_name)
    tempgen = TemplateDataGenerator(mod,temporal_mapping=temporal_mapping,
                                    layer_data=layer_data,exec_module=exec_module,pattern_name=pattern_name)
    tempgen.generate_hw_dependent_template_data()
    tempgen.generate_general_template_data()
    template_data=tempgen.get_template_data()
    tempengine = TemplateEngine(mod=mod,exec_module=exec_module,pattern_name=pattern_name,template_data=template_data)
    return tempengine.get_code()

def codegen(mod: tvm.ir.IRModule):
    target_name,pattern_name = mod.body.op.attrs["Composite"].split(".")[1:]
    code, error_codegen = get_code(mod=mod,target_name=target_name,pattern_name=pattern_name)
    if error_codegen:
        raise Exception("Couldn't generate output")
    return code
