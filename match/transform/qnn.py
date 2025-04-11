from tvm import relay
from tvm.relay import transform

# An opt pass that transforms all qnn.conv2d ops to a nn.conv2d op + cast
@transform.function_pass(opt_level=0)
class MatchQNNtoNNRewriter(relay.ExprMutator):
    # TODO: This may be broken, it may work only a subset of the instructions and exit

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if call.op.name == "qnn.conv2d":
            data = self.visit(call.args[0])
            kernel = self.visit(call.args[1])
            bias = self.visit(call.args[2])
            data_layout = call.attrs.data_layout
            kernel_layout = call.attrs.kernel_layout
            out_dtype = call.attrs.out_dtype
            out = relay.nn.conv2d(
                data,
                kernel,
                strides=call.attrs.strides,
                padding=call.attrs.padding,
                dilation=call.attrs.dilation,
                groups=call.attrs.groups,
                data_layout=data_layout,
                kernel_layout=kernel_layout,
                out_dtype = out_dtype,
            )
            #out = relay.cast(out, out_dtype)
            if bias is not None and bias.data.numpy() != 0:
                out = relay.add(out, bias)
            return out
        elif call.op.name == "qnn.dense":
            data = self.visit(call.args[0])
            kernel = self.visit(call.args[1])
            bias = self.visit(call.args[2])
            out_dtype = call.attrs.out_dtype
            out = relay.nn.dense(data, kernel, out_dtype="int32")
            #out = relay.cast(out, out_dtype)
            if bias is not None and bias.data.numpy() != 0:
                out = relay.add(out, bias)
            return out
        return super().visit_call(call)