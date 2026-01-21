import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback, is_op, wildcard, rewrite

@tvm.relay.transform.function_pass(opt_level=0)
class MatchDecomposeInstanceNorm(relay.ExprMutator):
    """Decompose nn.instance_norm into 3 logical stages matching TVM naming:
       1) tvmgen_default_fused_mean
       2) tvmgen_default_fused_variance  (compute variance via relay.op.variance)
       3) tvmgen_default_fused_subtract_add_rsqrt_multiply[_multiply_add]
          Only two final fused names are produced per user spec:
            - tvmgen_default_fused_subtract_add_rsqrt_multiply
            - tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add
    """
    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        new_op = self.visit(call.op)
        new_args = [self.visit(a) for a in call.args]

        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.instance_norm":
            data = new_args[0]
            gamma = new_args[1] if len(new_args) > 1 else None
            beta = new_args[2] if len(new_args) > 2 else None
            attrs = call.attrs
            epsilon = float(getattr(attrs, "epsilon", 1e-5))
            axis = int(getattr(attrs, "axis", 1))
            center = bool(getattr(attrs, "center", True))
            scale = bool(getattr(attrs, "scale", True))

            # Determine reduction axes (exclude batch + channel)
            red_axes = 2

            # 1) Mean stage: mean(x)
            mean = relay.op.mean(data, axis=red_axes, keepdims=True)
            mean = relay.frontend.common.set_span(mean, "tvmgen_default_fused_mean")

            # 2) Variance stage: variance(x)
            var = relay.op.variance(data, axis=red_axes, keepdims=True, with_mean=mean)
            var = relay.frontend.common.set_span(var, "tvmgen_default_fused_variance")

            # 3) Fused tail: (x - mean) * rsqrt(var + eps) * gamma + beta
            centered = relay.op.subtract(data, mean)
            var_eps = relay.op.add(var, relay.const(epsilon))
            inv_std = relay.op.rsqrt(var_eps)
            normed = relay.op.multiply(centered, inv_std)

            fused_name = "tvmgen_default_fused_subtract_add_rsqrt_multiply"
            if scale and center and (gamma is not None) and (beta is not None) and any(gamma.data.numpy()!=1.0) and any(beta.data.numpy()!=0.0):
                gamma = relay.const(gamma.data.numpy().reshape(1, gamma.data.shape[0]))
                normed = relay.op.multiply(normed, gamma)
                beta = relay.const(beta.data.numpy().reshape(1, beta.data.shape[0]))
                normed = relay.op.add(normed, beta)
                fused_name = "tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add"
            else:
                if scale and (gamma is not None) and any(gamma.data.numpy()!=1.0):
                    gamma = relay.const(gamma.data.numpy().reshape(1, gamma.data.shape[0]))
                    normed = relay.op.multiply(normed, gamma)
                if center and (beta is not None) and any(beta.data.numpy()!=0.0):
                    beta = relay.const(beta.data.numpy().reshape(1, beta.data.shape[0]))
                    normed = relay.op.add(normed, beta)

            normed = relay.frontend.common.set_span(normed, fused_name)
            return normed

        return relay.Call(new_op, new_args, call.attrs, call.type_args, call.span)

# class _InstanceNormDecomposeCallback(DFPatternCallback):
#     def __init__(self, require_type=False):
#         super().__init__(require_type=require_type)
#         self.data = wildcard()
#         self.gamma = wildcard()
#         self.beta = wildcard()
#         self.pattern = is_op("nn.instance_norm")(self.data, self.gamma, self.beta)

#     def callback(self, pre, post, node_map):
#         call = post  # nn.instance_norm call
#         data = node_map[self.data][0]
#         gamma = node_map[self.gamma][0]
#         beta = node_map[self.beta][0]
#         attrs = call.attrs
#         epsilon = float(getattr(attrs, "epsilon", 1e-5))
#         center = bool(getattr(attrs, "center", True))
#         scale = bool(getattr(attrs, "scale", True))
#         # We assume shape (1, C, S) -> reduce over spatial axis 2
#         red_axis = 2
#         mean = relay.op.mean(data, axis=red_axis, keepdims=True)
#         mean = relay.frontend.common.set_span(mean, "tvmgen_default_fused_mean")
#         var = relay.op.variance(data, axis=red_axis, keepdims=True)
#         var = relay.frontend.common.set_span(var, "tvmgen_default_fused_variance")
#         centered = relay.op.subtract(data, mean)
#         inv_std = relay.op.rsqrt(relay.op.add(var, relay.const(epsilon)))
#         normed = relay.op.multiply(centered, inv_std)
#         fused_name = "tvmgen_default_fused_subtract_add_rsqrt_multiply"
#         if scale and center:
#             normed = relay.op.multiply(normed, gamma)
#             normed = relay.op.add(normed, beta)
#             fused_name = "tvmgen_default_fused_subtract_add_rsqrt_multiply_multiply_add"
#         else:
#             if scale:
#                 normed = relay.op.multiply(normed, gamma)
#             if center:
#                 normed = relay.op.add(normed, beta)
#         normed = relay.frontend.common.set_span(normed, fused_name)
#         return normed

# @tvm.ir.transform.module_pass(opt_level=0)
# class MatchDecomposeInstanceNorm:
#     def transform_module(self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext) -> tvm.ir.IRModule:
#         # Apply pattern rewrite to every function
#         new_funcs = {}
#         for gvar, func in mod.functions.items():
#             if isinstance(func, relay.Function):
#                 func = rewrite(_InstanceNormDecomposeCallback(), func)
#             new_funcs[gvar] = func
#         for gvar, func in new_funcs.items():
#             mod.update_func(gvar, func)
#         return mod

#     def __call__(self, mod):
#         return self.transform_module(mod)
