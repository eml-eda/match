from tvm.relay.dataflow_pattern import DFPatternCallback, is_op, rewrite, wildcard, is_constant
from tvm import relay
import tvm
import numpy as np

class DivFloorPlinioOnnx(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)

        self.div = is_op("divide")(wildcard(),is_constant())
        self.floor = is_op("floor")(self.div)
        self.clip = is_op("clip")(self.floor)
        self.cast = is_op("cast")(self.clip)
        self.pattern = self.cast

    def callback(self, pre, post, node_map):
        div = node_map[self.div][0]
        cast = node_map[self.cast][0]
        clip = node_map[self.clip][0]

        shift_factor = int(np.log2(abs(int(div.args[1].data.numpy()))))

        x = relay.op.right_shift(div.args[0], relay.const(shift_factor))
        x = relay.op.clip(x, a_min=int(clip.attrs.a_min), a_max=int(clip.attrs.a_max))
        return relay.op.cast(x, cast.attrs["dtype"])
    
class DivReqPlinioOnnx(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)

        self.div = is_op("divide")(wildcard(),is_constant())
        self.clip = is_op("clip")(self.div)
        self.cast = is_op("cast")(self.clip)
        self.pattern = self.cast

    def callback(self, pre, post, node_map):
        div = node_map[self.div][0]
        cast = node_map[self.cast][0]
        clip = node_map[self.clip][0]

        shift_factor = int(np.log2(abs(int(div.args[1].data.numpy()))))

        x = relay.op.right_shift(div.args[0], relay.const(shift_factor))
        x = relay.op.clip(x, a_min=int(clip.attrs.a_min), a_max=int(clip.attrs.a_max))
        return relay.op.cast(x, cast.attrs["dtype"])
    
class FloorDivCastOnnx(DFPatternCallback):
    """Rewriter for digital requant pattern
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.floor = is_op("floor")(wildcard())
        self.div = is_op("divide")(self.floor,is_constant())
        self.pattern = self.div

    def callback(self, pre, post, node_map):
        div = node_map[self.div][0]

        shift_factor = int(np.log2(abs(int(div.args[1].data.numpy()))))

        x = relay.op.right_shift(div.args[0].args[0], relay.const(shift_factor))
        return x

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
            # TODO: why they are all used, whats different? fuse into a single rewriter!
            func = rewrite(DivFloorPlinioOnnx(), func)
            func = rewrite(DivReqPlinioOnnx(), func)
            func = rewrite(FloorDivCastOnnx(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)