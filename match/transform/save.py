from match.relay.compiled_module import CompiledModule
from match.utils.utils import add_save_relay
import tvm


@tvm.ir.transform.module_pass(opt_level=0)
class MatchSaveModule:
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        match_inputs = None
        match_output = None
        CompiledModule.define_compiled_module(mod=mod,match_inputs=match_inputs,match_output=match_output)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)
    
@tvm.ir.transform.module_pass(opt_level=0)
class MatchSaveRelay:

    def __init__(self,prefix_relay:str=""):
        super().__init__()
        self.prefix_relay=prefix_relay

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        add_save_relay(prefix=self.prefix_relay,mod=mod)
        return mod
    
    def __call__(self, mod):
        return self.transform_module(mod)