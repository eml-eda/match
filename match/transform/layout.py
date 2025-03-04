import tvm
import tvm.relay
from tvm import relay

NCHW_TO_NHWC_OPERATORS_SET = (
    "nn.conv2d", "nn.max_pool2d", "nn.avg_pool2d",
    "nn.global_max_pool2d", "nn.global_avg_pool2d",
    "nn.batch_norm", "nn.instance_norm", "nn.layer_norm"
)
LAYOUT_FROM_TO = {"data_layout":{"from":"NCHW","to":"NHWC"},"kernel_layout":{"from":"OIHW","to":"HWIO"},"layout":{"from":"NCHW","to":"NHWC"}}

@tvm.relay.transform.function_pass(opt_level=0)
class MatchLayoutNCHWtoNHWC(relay.ExprMutator):
    
    def __init__(self):
        super().__init__()
        self.new_vars = {}

    def transform_function(self, func, mod, ctx):
        return self.visit(func)
    
    def visit_function(self, fn):
        """Rewrite function arguments
        """
        new_params = []
        binds = {}

        for param in fn.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation

            # bias params are int32
            if param.name_hint.endswith('bias'):
                dtype = 'int32'
            else:
                dtype = var_type.dtype

            # Generate new variable.
            new_param = relay.var(param.name_hint, shape=var_type.shape if len(var_type.shape)!=4 else [var_type.shape[0],var_type.shape[2],var_type.shape[3],var_type.shape[1]], dtype=dtype)

            new_params.append(new_param)
            self.new_vars[param.name_hint]=new_param
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
        shape = [int(sz) for sz in const.checked_type.shape]
        len_shape = len(shape)
        dim_sized_one = sum([sz==1 for sz in shape])
        dtype = const.checked_type.dtype
        if len_shape>1 and dim_sized_one==len_shape-1:
            size_dim_not_at_one = int([sz for sz in shape if sz>1][0])
            #breakpoint()
            # broadcasting
            return relay.const(const.data.numpy().reshape(tuple([1 for _ in range(int(3))]+[size_dim_not_at_one])).astype(dtype))
        elif len_shape==4:
            return relay.const(const.data.numpy().reshape((shape[2],shape[3],shape[1],shape[0])).astype(dtype))
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
        if isinstance(call.op, tvm.ir.Op) and call.op.name=="nn.bias_add":
            updated_attrs = {key: getattr(call.attrs, key) for key in call.attrs.keys()}
            updated_attrs["axis"] = -1
            if "nn" in call.op.name and hasattr(relay.op.nn,".".join(call.op.name.split(".")[1:])):
                op_func = getattr(relay.op.nn, ".".join(call.op.name.split(".")[1:]))
            else:
                if hasattr(relay.op.nn,call.op.name):
                    op_func = getattr(relay.op.nn,call.op.name)
                elif hasattr(relay.op,call.op.name):
                    op_func = getattr(relay.op,call.op.name)
                elif hasattr(relay.nn,call.op.name):
                    op_func = getattr(relay.nn,call.op.name)

            if op_func:
                new_call = op_func(*new_args,**updated_attrs)
            else:

                new_call = relay.Call(call.op, new_args, call.attrs)
        if isinstance(call.op, tvm.ir.Op) and call.op.name in NCHW_TO_NHWC_OPERATORS_SET:
            updated_attrs = {key: getattr(call.attrs, key) for key in call.attrs.keys()}
            for layout_key in [key for key in updated_attrs.keys() if key in LAYOUT_FROM_TO and updated_attrs[key]!="" and updated_attrs[key]==LAYOUT_FROM_TO[key]["from"]]:
                updated_attrs[layout_key] = LAYOUT_FROM_TO[layout_key]["to"]

            if "nn" in call.op.name and hasattr(relay.op.nn,".".join(call.op.name.split(".")[1:])):
                op_func = getattr(relay.op.nn, ".".join(call.op.name.split(".")[1:]))
            else:
                if hasattr(relay.op.nn,call.op.name):
                    op_func = getattr(relay.op.nn,call.op.name)
                elif hasattr(relay.op,call.op.name):
                    op_func = getattr(relay.op,call.op.name)
                elif hasattr(relay.nn,call.op.name):
                    op_func = getattr(relay.nn,call.op.name)

            if op_func:
                new_call = op_func(*new_args,**updated_attrs)
            else:

                new_call = relay.Call(call.op, new_args, call.attrs)
        
        # Default behavior for other operators
        if new_call is None:
            new_call = relay.Call(call.op, new_args, call.attrs)
        return new_call