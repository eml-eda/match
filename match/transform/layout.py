import tvm
import tvm.relay
from tvm import relay
import numpy as np

NCHW_TO_NHWC_OPERATORS_SET = (
    "nn.conv2d", "nn.max_pool2d", "nn.avg_pool2d",
    "nn.global_max_pool2d", "nn.global_avg_pool2d",
    "nn.batch_norm", "nn.instance_norm", "nn.layer_norm"
)
LAYOUT_FROM_TO = {"data_layout":{"from":"NCHW","to":"NHWC"},"kernel_layout":{"from":"OIHW","to":"HWIO"},"layout":{"from":"NCHW","to":"NHWC"}}

desired_layouts = {
    "nn.conv1d": ["NWC", "OWI"],
    "nn.conv2d": ["NHWC", "HWIO"],
    "nn.conv3d": ["NDHWC", "DHWIO"],
    "nn.max_pool2d": ["NHWC"],
    "nn.avg_pool2d": ["NHWC"],
    "nn.global_max_pool2d": ["NHWC"],
    "nn.global_avg_pool2d": ["NHWC"],
    "nn.batch_norm": ["NHWC"],
    "nn.instance_norm": ["NHWC"],
    "nn.layer_norm": ["NHWC"],
    "nn.dense": ["NDHWC"]
}
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

class RemoveLayoutTransformMutator(relay.ExprMutator):
    def visit_call(self, call):
        # 1. Look for the nn.dense operator
        if call.op == tvm.ir.Op.get("nn.dense"):
            inputs = call.args
            data_expr = inputs[0]
            weight_expr = inputs[1]

            # 2. Check if the input to dense comes from nn.batch_flatten
            # We recurse/visit the data expression first to ensure upstream graph is processed
            data_expr = self.visit(data_expr)
            
            if isinstance(data_expr, relay.Call) and data_expr.op == tvm.ir.Op.get("nn.batch_flatten"):
                flatten_input = data_expr.args[0]
                flatten_input = self.visit(flatten_input)

                # 3. Check if the input to batch_flatten comes from a layout_transform
                if isinstance(flatten_input, relay.Call) and flatten_input.op == tvm.ir.Op.get("layout_transform"):
                    lt_attrs = flatten_input.attrs
                    
                    # Verify it's exactly the NDHWC -> NCDHW transform you want to eliminate
                    if lt_attrs.src_layout == "NDHWC" and lt_attrs.dst_layout == "NCDHW":
                        # We also need to make sure the weights are a Constant we can manipulate
                        if isinstance(weight_expr, relay.Constant):
                            
                            # --- NumPy Weight Permutation Magic ---
                            # Extract raw data from TVM Constant to a NumPy array
                            old_weights = weight_expr.data.asnumpy() # Shape: (Units, 1260) -> (1, 1260)
                            units = old_weights.shape[0]
                            
                            # Hardcoded mapping based on your graph dimensions:
                            # Total features 1260 = C(70) * D(2) * H(3) * W(3)
                            C, D, H, W = 70, 2, 3, 3
                            
                            # Reshape to 5D logical NCDHW structure (including units axis)
                            weights_5d = old_weights.reshape(units, C, D, H, W)
                            
                            # Permute axes to match NDHWC layout flattening order:
                            # Original axes: 0=Units, 1=C, 2=D, 3=H, 4=W
                            # Target order:  Units, D, H, W, C -> axes (0, 2, 3, 4, 1)
                            permuted_weights_5d = np.transpose(weights_5d, (0, 2, 3, 4, 1))
                            
                            # Flatten back to 2D matrix (Units, 1260)
                            new_weights_np = permuted_weights_5d.reshape(units, -1)
                            
                            # Create a fresh TVM Constant with the updated layout data
                            new_weight_const = relay.const(new_weights_np, dtype=weight_expr.data.dtype)
                            
                            # --- Graph Rewriting ---
                            # Bypass layout_transform: feed the layout_transform's source (%40) 
                            # directly into a new batch_flatten node
                            new_flatten = relay.nn.batch_flatten(flatten_input.args[0])
                            
                            # Construct the new dense layer using the bypassed data and updated weights
                            new_dense = relay.nn.dense(new_flatten, new_weight_const, units=call.attrs.units)
                            return new_dense

        return super().visit_call(call)

# --- How to wrap and execute it as a standard TVM Pass ---
@tvm.relay.transform.function_pass(opt_level=1)
def RemoveLayoutTransformPass(function, module, context):
    mutator = RemoveLayoutTransformMutator()
    return mutator.visit(function)