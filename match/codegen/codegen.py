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
from match.matchutils import get_hw_model,mock_func

def generate_code(mod: tvm.ir.IRModule,device_name:str="",pattern_name:str=""):
    hwmodel=get_hw_model(device_name=device_name)
    tmapgen = TemporalMappingGenerator(mod=mod,hwmodel=hwmodel,pattern_name=pattern_name)
    tmapgen.generate_workload()
    tmapgen.generate_temporal_mapping()
    tmapgen.constraint_temporal_mapping()
    temporal_mapping=tmapgen.get_temporal_mapping()
    layer_data=tmapgen.get_layer_data()
    tempgen = TemplateDataGenerator(mod,temporal_mapping=temporal_mapping,
                                    layer_data=layer_data,hwmodel=hwmodel,pattern_name=pattern_name)
    tempgen.generate_hw_dependent_template_data()
    tempgen.generate_general_template_data()
    template_data=tempgen.get_template_data()
    breakpoint()
    tempengine = TemplateEngine(mod=mod,hwmodel=hwmodel,pattern_name=pattern_name,template_data=template_data)
    return tempengine.get_code()


def codegen(mod: tvm.ir.IRModule):
    device_name,pattern_name = mod.body.op.attrs["Composite"].split(".")[1].split("_")
    code, error_codegen = generate_code(mod=mod,device_name=device_name,pattern_name=pattern_name)
    if error_codegen:
        raise Exception("Couldn't generate output")
    return code
