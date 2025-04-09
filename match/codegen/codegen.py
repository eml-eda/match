# TVM imports
from tvm import relay
from match.codegen.template_writer import TemplateWriter
from match.utils.utils import add_fname_node_schedule
import tvm
from match.target import get_target
import tvm.relay

def schedule_to_code(mod: tvm.ir.IRModule,exec_module_name:str="",pattern_name:str=""):
    target=get_target()
    schedule,match_node,exec_module,latency,energy=target.get_layer_from_module(mod=mod,exec_module_name=exec_module_name,pattern_name=pattern_name)
    tempengine = TemplateWriter(mod=mod,target=target,exec_module=exec_module,
                                pattern_name=pattern_name,schedule=schedule,match_node=match_node,
                                latency=latency,energy=energy)
    return tempengine.get_code()

def codegen(mod: tvm.ir.IRModule):
    _,exec_module_name,pattern_name = mod.body.op.attrs["Composite"].split(".")[1:]
    try:
        code = schedule_to_code(mod=mod,exec_module_name=exec_module_name,pattern_name=pattern_name)
        error_codegen = False
    except Exception as exc:
        print(f"[CODEGEN]: Couldn't generate the output, {exc}")
        error_codegen = True
    if error_codegen:
        raise Exception("[CODEGEN]: Couldn't generate the output")
    return code
