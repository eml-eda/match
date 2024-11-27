from match.codegen.template import TemplateDataGenerator,TemplateEngine

# TVM imports
import tvm
from match.target import get_target

def get_code(mod: tvm.ir.IRModule,exec_module_name:str="",pattern_name:str=""):
    target=get_target()
    temporal_mapping,layer_data,exec_module,latency,energy=target.get_layer_from_module(mod=mod,exec_module_name=exec_module_name,pattern_name=pattern_name)
    tempgen = TemplateDataGenerator(mod,temporal_mapping=temporal_mapping,
                                    layer_data=layer_data,exec_module=exec_module,pattern_name=pattern_name,
                                    latency=latency,energy=energy)
    tempgen.generate_hw_dependent_template_data()
    tempgen.generate_general_template_data()
    template_data=tempgen.get_template_data()
    tempengine = TemplateEngine(mod=mod,exec_module=exec_module,pattern_name=pattern_name,template_data=template_data)
    return tempengine.get_code()

def codegen(mod: tvm.ir.IRModule):
    _,exec_module_name,pattern_name = mod.body.op.attrs["Composite"].split(".")[1:]
    try:
        code, error_codegen = get_code(mod=mod,exec_module_name=exec_module_name,pattern_name=pattern_name)
    except Exception as exc:
        #breakpoint()
        raise exc
    if error_codegen:
        #breakpoint()
        raise Exception("Couldn't generate output")
    return code
