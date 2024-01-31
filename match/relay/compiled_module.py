
import tvm

class CompiledModuleResult:
    def __init__(self,mod: tvm.ir.IRModule,match_inputs,match_output):
        self.mod=mod
        self.match_inputs=match_inputs
        self.match_output=match_output

class CompiledModule:
    def __init__(self):
        return
    @classmethod
    def define_compiled_module(cls,mod: tvm.ir.IRModule,match_inputs,match_output):
        cls.result=CompiledModuleResult(mod=mod,match_inputs=match_inputs,match_output=match_output)