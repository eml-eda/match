import tvm
from tvm.relay.dataflow_pattern import DFPatternCallback, is_op, rewrite, wildcard, is_constant
from tvm import relay

class AddCastInMainPatternCallback(DFPatternCallback):
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.conv2d = is_op("nn.conv2d")(wildcard(),wildcard())
        self.multiply = wildcard()
        self.pattern = is_op("multiply")(self.conv2d,self.multiply)

    def callback(self, pre, post, node_map):
        conv2d = node_map[self.conv2d][0]
        mult = node_map[self.multiply][0]
        return relay.op.multiply(relay.op.cast(conv2d, "int32"),mult)


@tvm.ir.transform.module_pass(opt_level=0)
class MatchAddCastInMain:
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        global_var=mod.get_global_var("main")
        func=mod.functions[global_var]
        func = rewrite(AddCastInMainPatternCallback(), func)
        mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)