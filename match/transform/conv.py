import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    rewrite,
    wildcard,
    is_op,
    is_constant,
)

# An opt pass that transforms all qnn.conv2d ops to a nn.conv2d op + cast
@transform.function_pass(opt_level=0)
class MatchDenseToConvRewriter(relay.ExprMutator):

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):

        if call.op.name == "nn.dense":
            data = self.visit(call.args[0])
            if len(call.args) > 2:
                bias = self.visit(call.args[2])
            else:
                bias = None
            kernel = self.visit(call.args[1])
            kernel_data = kernel.data.numpy()
            f_in = kernel_data.shape[1]
            f_out = kernel_data.shape[0]
            # Get the kernel data
            # Convert it to (F_OUT, F_IN)
            kernel_data = kernel_data.transpose(1, 0)
            # Convert it to HWCK
            kernel_data = kernel_data.reshape((1, 1, f_in, f_out))
            kernel_size = (1, 1)
            out_dtype = call.attrs.out_dtype
            # Create a new constant for the kernel
            kernel = relay.const(kernel_data, dtype=kernel.checked_type.dtype)
            # Swap with a new node
            # If the input is on 2 dimensions, we need to add manually 2 dimensions at the end!
            out = data
            if len(call.args[0].checked_type.shape) == 2:
                shape = call.args[0].checked_type.shape
                out = relay.reshape(out, newshape = [shape[0], 1, 1, shape[1]])

            out = relay.nn.conv2d(
                out,  # The original input to pad
                kernel,  
                strides=(1,1),
                padding=(0, 0, 0, 0),  # Merge padding directly into conv2d
                dilation=(1, 1),
                groups=1,
                channels=f_out,
                kernel_size=(1,1),
                data_layout="NHWC",
                kernel_layout="HWIO",
                out_dtype="int32",
            )

            # Squash back the output, for easier compatibility
            out = relay.reshape(out, newshape = [shape[0], f_out])

            # Then add the bias with the original dimension
            if bias is not None and bias.data.numpy() != 0:
                out = relay.add(out, bias)
            return out
        return super().visit_call(call)

class MatchMergePadConv2dRewriter(DFPatternCallback):
    """Rewriter for merging pad and conv2d pattern"""

    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.x = wildcard()
        pad = is_op("nn.pad")(self.x, is_constant())
        self.pattern = is_op("nn.conv2d")(pad, wildcard())

    def callback(self, pre, post, node_map):
        pad_node = node_map[self.pattern.args[0]][0]
        conv_node = post

        # Extract padding values from the pad node
        pad_attr = pad_node.attrs
        pad_width = pad_attr.pad_width

        # Ensure padding is for NHWC or NCHW format
        if len(pad_width) == 4 and all(len(p) == 2 for p in pad_width):
            if conv_node.attrs.data_layout == "NHWC":
                pad_top, pad_bottom = pad_width[1]
                pad_left, pad_right = pad_width[2]
            elif conv_node.attrs.data_layout == "NCHW":
                pad_top, pad_bottom = pad_width[2]
                pad_left, pad_right = pad_width[3]
            new_padding = (pad_top, pad_left, pad_bottom, pad_right)

            # Create a new conv2d node with updated padding
            new_conv = relay.nn.conv2d(
                pad_node.args[0],  # The original input to pad
                conv_node.args[1],  # The weights remain unchanged
                strides=conv_node.attrs.strides,
                padding=new_padding,  # Merge padding directly into conv2d
                dilation=conv_node.attrs.dilation,
                groups=conv_node.attrs.groups,
                channels=conv_node.attrs.channels,
                kernel_size=conv_node.attrs.kernel_size,
                data_layout=conv_node.attrs.data_layout,
                kernel_layout=conv_node.attrs.kernel_layout,
                out_dtype=conv_node.attrs.out_dtype,
            )
            return new_conv

        return post


@tvm.ir.transform.module_pass(opt_level=0)
class MatchMergePadConv2d:
    """Find and rewrite MATCH ONNX requant to requant for internal use:
    """

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(MatchMergePadConv2dRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)