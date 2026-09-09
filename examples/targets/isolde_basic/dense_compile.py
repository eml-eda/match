import tvm
from tvm import relay
import match
from match.relay.utils.utils import create_random_array
from match.utils.utils import get_default_inputs
from match.model.model import MatchModel as Model
import random
from astral import Astral

OUTPUT_DIR_PATH = "output"

def create_dense_ex(
    inp_features:int=64, out_features:int=10,
    right_shift:int=1,**kwargs
):
    """Generate a small network in TVM Relay IR that performs a requantized convolution
    """
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType((1,inp_features), "float16"))
    # Get or generate weight_values
    weights = create_random_array((out_features,inp_features),"float16", min_val=-1.0, max_val=1.0)
    # Get or generate bias values
    bias = create_random_array((out_features,), "float16", min_val=-1000.0, max_val=1000.0)
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "dense_weights"
    bias_name = "dense_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, bias_name: bias}

    # define operations
    x = relay.op.nn.dense(x, w, out_dtype=bias.dtype)
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=-1)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

random.seed(1341211)
relay_mod, relay_params = create_dense_ex(inp_features=64, out_features=10, right_shift=1)

oenne_model = Model(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[]),
    # handle_out_fn="handle_fp16_classifier",
    debug = True,
    debug_fallback = True,
    profile = False,
    profile_fallback = False,
)
target = Astral()
# target.disable_exec_module("pulp_cluster")
match.match(
    model = oenne_model,
    target = target,
    output_path = OUTPUT_DIR_PATH,
)   
