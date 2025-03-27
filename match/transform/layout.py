import tvm
import tvm.relay
from tvm import relay

NCHW_TO_NHWC_OPERATORS_SET = (
    "nn.conv2d", "nn.max_pool2d", "nn.avg_pool2d",
    "nn.global_max_pool2d", "nn.global_avg_pool2d",
    "nn.batch_norm", "nn.instance_norm", "nn.layer_norm"
)
LAYOUT_FROM_TO = {"data_layout":{"from":"NCHW","to":"NHWC"},"kernel_layout":{"from":"OIHW","to":"HWIO"},"layout":{"from":"NCHW","to":"NHWC"}}

desired_layouts = {"nn.conv2d": ["NHWC", "HWIO"], "nn.max_pool2d": ["NHWC"], "nn.avg_pool2d": ["NHWC"], "nn.global_max_pool2d": ["NHWC"], "nn.global_avg_pool2d": ["NHWC"], "nn.batch_norm": ["NHWC"], "nn.instance_norm": ["NHWC"], "nn.layer_norm": ["NHWC"]}
MatchLayoutNCHWtoNHWCTVM = relay.transform.ConvertLayout(desired_layouts)


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

        new_body = self.visit(fn.body)
        
        for param in fn.params:
            new_param = param
            if hasattr(param, "name_hint") and isinstance(param, relay.Var) and param.name_hint in self.new_vars:
                new_param = self.new_vars[param.name_hint]
            new_params.append(new_param)
            binds[param] = new_param
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
            # broadcasting
            return relay.const(const.data.numpy().reshape(tuple([1 for _ in range(int(3))]+[size_dim_not_at_one])).astype(dtype))
        return super().visit_constant(const)

    def visit_var(self, var):
        if var.name_hint in self.new_vars:
            return self.new_vars[var.name_hint]
        else:
            return super().visit_var(var)

    def modify_axis_to_bias_add(self, call, new_args):
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
        return new_call

    def visit_call(self, call):
        new_call = None
        # Modify layout-sensitive operators
        if isinstance(call.op, tvm.ir.Op) and call.op.name=="nn.bias_add":
            # Recurse into arguments
            new_args = [self.visit(arg) for arg in call.args]
            new_call = self.modify_axis_to_bias_add(call, new_args)
        elif isinstance(call.op, tvm.ir.Op) and call.op.name in NCHW_TO_NHWC_OPERATORS_SET:
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

            new_args_layout = []
            # Recurse into arguments
            for arg in call.args:
                arg_to_add = arg
                if isinstance(arg, relay.Constant):
                    shape = [int(sz) for sz in arg.checked_type.shape]
                    if len(shape)==4 and (call.attrs.kernel_layout=="" or call.attrs.kernel_layout==LAYOUT_FROM_TO["kernel_layout"]["from"]):
                        arg_to_add = relay.const(arg.data.numpy().transpose(2,3,1,0).astype(arg.checked_type.dtype))
                elif isinstance(arg, relay.Var):
                    shape = arg.type_annotation.shape
                    if len(shape)==4 and (call.attrs.kernel_layout=="" or call.attrs.kernel_layout==LAYOUT_FROM_TO["kernel_layout"]["from"]):
                        new_var = relay.var(arg.name_hint, shape=(shape[0],shape[2],shape[3],shape[1]), dtype=arg.type_annotation.dtype)  
                        arg_to_add = new_var
                        self.new_vars[arg.name_hint] = new_var
                else:
                    arg_to_add = self.visit(arg)    
                new_args_layout.append(arg_to_add)

            if op_func:
                new_call = op_func(*new_args_layout,**updated_attrs)
            else:

                new_call = relay.Call(call.op, new_args_layout, call.attrs)
        
        # Default behavior for other operators
        elif new_call is None:
            # Recurse into arguments
            new_args = [self.visit(arg) for arg in call.args]
            new_call = relay.Call(call.op, new_args, call.attrs)
        return new_call