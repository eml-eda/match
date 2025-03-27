from tvm.relay.dataflow_pattern import DFPatternCallback, is_op, rewrite, wildcard, is_constant
from tvm import relay
import tvm
import numpy as np
    
class FloorDivToRightShiftRewriter(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.floor = is_op("floor")(wildcard())
        self.div = is_op("divide")(self.floor,is_constant())
        self.pattern = self.div

    def callback(self, pre, post, node_map):
        div = node_map[self.div][0]
        floor = node_map[self.floor][0]
        shift_factor = int(np.log2(abs(int(div.args[1].data.numpy()))))

        return relay.op.right_shift(floor.args[0], relay.const(shift_factor))
    
class DivFloorToRightShiftRewriter(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.div = is_op("divide")(wildcard(),is_constant())
        self.floor = is_op("floor")(self.div)
        self.pattern = self.floor

    def callback(self, pre, post, node_map):
        div = node_map[self.div][0]
        shift_factor = int(np.log2(abs(int(div.args[1].data.numpy()))))

        return relay.op.right_shift(div.args[0], relay.const(shift_factor))

@tvm.ir.transform.module_pass(opt_level=0)
class MatchRequantRewriter:
    """ Find and rewrite MATCH requant to requant for internal use:
        div->div->floor->max->min to
        right_shift->clip->cast
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(DivFloorToRightShiftRewriter(), func)
            func = rewrite(FloorDivToRightShiftRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)