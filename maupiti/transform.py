import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    rewrite,
    wildcard,
    is_op,
    is_constant,
)


class MaupitiFlattenRewriter(DFPatternCallback):
    """Rewriter for digital requant pattern"""

    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.x = wildcard()
        transpose = is_op("transpose")(self.x)
        reshape = is_op("reshape")(transpose)
        comp = is_op("annotation.compiler_begin")(reshape)
        self.pattern = is_op("annotation.compiler_end")(is_op("nn.batch_flatten")(comp))

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        return relay.nn.batch_flatten(x)


@tvm.ir.transform.module_pass(opt_level=0)
class MaupitiBatchFlattenTransform:
    """Find and rewrite MATCH ONNX requant to requant for internal use:
    div->div->floor->max->min to
    right_shift->clip->cast
    """

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(MaupitiFlattenRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)


@transform.function_pass(opt_level=0)
class ONNXNHWCRewriter(relay.ExprMutator):
    NCHW_TO_NHWC_OPERATORS_SET = (
        "qnn.conv2d",
        "nn.pad",
        "nn.conv2d",
        "nn.max_pool2d",
        "nn.avg_pool2d",
        "nn.global_max_pool2d",
        "nn.global_avg_pool2d",
        "nn.batch_norm",
        "nn.instance_norm",
        "nn.layer_norm",
    )
    LAYOUT_FROM_TO = {
        "data_layout": {"from": "NCHW", "to": "NHWC"},
        "pad_width": {"from": "NCHW", "to": "NHWC"},
        "kernel_layout": {"from": "OIHW", "to": "HWIO"},
    }

    def __init__(self):
        super().__init__()
        self.new_vars = {}

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_function(self, fn):
        """Rewrite function arguments"""
        new_params = []
        binds = {}

        for param in fn.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation

            # bias params are int32
            if param.name_hint.endswith("bias"):
                dtype = "int32"
            else:
                dtype = var_type.dtype

            # Generate new variable.
            var_shape_out = var_type.shape
            if len(var_type.shape) == 4:
                var_shape_out = (
                    var_shape_out[0],
                    var_shape_out[2],
                    var_shape_out[3],
                    var_shape_out[1],
                )

            new_param = relay.var(param.name_hint, shape=var_shape_out, dtype=dtype)

            new_params.append(new_param)
            self.new_vars[param.name_hint] = new_param
            binds[param] = new_param

        new_body = self.visit(fn.body)
        # Rewrite the body to use new parameters.
        new_body = relay.bind(new_body, binds)

        # Construct the updated function and return.
        new_func = relay.Function(
            new_params,
            new_body,
            # You could change the return type, if you use None it will re-infer.
            None,
            type_params=fn.type_params,
            attrs=fn.attrs,
        )
        return new_func

    def visit_constant(self, const):
        # Get the shape of the constant
        shape = np.array(const.data.numpy().shape, dtype=int)
        # Broadcasting
        # If the shape is ND tensor (N>1) and all but one dimension is 1
        data = const.data.numpy()

        if len(shape) > 1 and np.sum(shape == 1) == len(shape) - 1:
            # Move at the end the != 1 dimension
            new_axis = [i for i in range(len(shape)) if shape[i] == 1]
            new_axis.extend([i for i in range(len(shape)) if shape[i] != 1])
            return relay.const(
                data.transpose(new_axis).astype(const.checked_type.dtype)
            )
        elif len(shape) == 4:
            # KCHW -> HWCK
            # NOTE: Performs the permute on all constants, it may be wrong
            return relay.const(
                data.transpose(2, 3, 1, 0).astype(const.checked_type.dtype)
            )
        return super().visit_constant(const)

    def visit_var(self, var):
        if var.name_hint in self.new_vars:
            return self.new_vars[var.name_hint]
        else:
            return super.visit_var(var)

    def visit_call(self, call):
        # Recurse into arguments
        new_args = [self.visit(arg) for arg in call.args]
        new_call = None
        # Modify layout-sensitive operators
        if (
            isinstance(call.op, tvm.ir.Op)
            and call.op.name in self.NCHW_TO_NHWC_OPERATORS_SET
        ):
            updated_attrs = {key: getattr(call.attrs, key) for key in call.attrs.keys()}
            if "pad" in call.op.name:
                updated_attrs["pad_value"] = new_args.pop(-1)

            for key in updated_attrs.keys():
                if key in self.LAYOUT_FROM_TO:
                    if updated_attrs[key] != "":
                        if updated_attrs[key] == self.LAYOUT_FROM_TO[key]["from"]:
                            updated_attrs[key] = self.LAYOUT_FROM_TO[key]["to"]
                        elif "pad" in key and len(updated_attrs[key]) == 4:
                            data = np.array(
                                updated_attrs[key], dtype=int
                            )  # A 4,2 array
                            # Change row  order
                            data = data[[0, 2, 3, 1], :].tolist()
                            data = tuple(tuple(x) for x in data)
                            # Convert to TVM
                            updated_attrs[key] = data

            # TODO: Make this better
            if "qnn.conv2d" in call.op.name:
                op_func = relay.qnn.conv2d
            elif "nn" in call.op.name and hasattr(
                relay.op.nn, ".".join(call.op.name.split(".")[1:])
            ):
                op_func = getattr(relay.op.nn, ".".join(call.op.name.split(".")[1:]))
            else:
                if hasattr(relay.op.nn, call.op.name):
                    op_func = getattr(relay.op.nn, call.op.name)
                elif hasattr(relay.op, call.op.name):
                    op_func = getattr(relay.op, call.op.name)
                elif hasattr(relay.nn, call.op.name):
                    op_func = getattr(relay.nn, call.op.name)

            if op_func:
                new_call = op_func(*new_args, **updated_attrs)
            else:

                new_call = relay.Call(call.op, new_args, call.attrs)

        # Default behavior for other operators
        if new_call is None:
            new_call = relay.Call(call.op, new_args, call.attrs)
        return new_call


# An opt pass that transforms all qnn.conv2d ops to a nn.conv2d op + cast
@transform.function_pass(opt_level=0)
class MaupitiQOpRewriter(relay.ExprMutator):
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

class MaupitiMergePadConv2dRewriter(DFPatternCallback):
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
class MaupitiMergePadConv2d:
    """Find and rewrite MATCH ONNX requant to requant for internal use:
    """

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(MaupitiMergePadConv2dRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)





def maupiti_network_transformations(opts=None):
    pipeline = []
    pipeline.append(transform.InferType())
    pipeline.append(ONNXNHWCRewriter())
    pipeline.append(transform.InferType())
    pipeline.append(MaupitiBatchFlattenTransform())
    pipeline.append(transform.InferType())
    pipeline.append(MaupitiQOpRewriter())
    pipeline.append(transform.InferType())
    pipeline.append(MaupitiMergePadConv2d())
    pipeline.append(transform.InferType())
    return pipeline


def maupiti_adjust_network(opts=None):
    pipeline = []

    return pipeline
