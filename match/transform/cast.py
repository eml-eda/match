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

NODE_WITH_OUT_DTYPE_ATTR = ("nn.dense", "nn.conv2d", "nn.conv1d")
CAN_REMOVE_FAKE_CAST = False
@tvm.relay.transform.function_pass(opt_level=0)
class MatchRemoveFakeOutDtypeCasts(tvm.relay.expr_functor.ExprMutator):
    """Cast linear layers in graph to integers and insert the necessary cast operations (from MATCH ONNX file)
    """
    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        """Rewrite ops
        """
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        is_out_dtype_node = False
        is_reshape_of_out_dtype_node = False
        is_fake_cast_node = False
        if len(new_args) > 0:
            span_name = "" if not hasattr(call.span, "source_name") else call.span.source_name.name
            is_out_dtype_node = isinstance(new_args[0], relay.Call) and new_args[0].op.name in NODE_WITH_OUT_DTYPE_ATTR
            is_reshape_of_out_dtype_node = isinstance(new_args[0], relay.Call) and new_args[0].op.name=="reshape" and new_args[0].args[0]
            is_fake_cast_node = span_name=="FAKE_CAST_TVM_OUT_DTYPE" and call.op.name=="cast" \
                and isinstance(new_args[0], relay.Call) and (is_out_dtype_node or is_reshape_of_out_dtype_node)
        if not CAN_REMOVE_FAKE_CAST and is_fake_cast_node:
            if is_reshape_of_out_dtype_node:
                new_args[0]=new_args[0].args[0]
            new_args_attrs=dict(new_args[0].attrs)
            new_args_attrs["out_dtype"]=call.attrs.dtype
            new_args[0]=relay.frontend.common.get_relay_op(new_args[0].op.name)(*new_args[0].args, **new_args_attrs)
        # Default case
        new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        if CAN_REMOVE_FAKE_CAST and is_fake_cast_node:
            if is_reshape_of_out_dtype_node:
                new_args[0]=new_args[0].args[0]
            new_args_attrs=dict(new_args[0].attrs)
            new_args_attrs["out_dtype"]=call.attrs.dtype
            new_args[0]=relay.frontend.common.get_relay_op(new_args[0].op.name)(*new_args[0].args, **new_args_attrs)
            new_call=new_args[0]
        return new_call