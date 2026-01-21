import sys

MATCH_PATH = "../../../.."
sys.path.append(f"{MATCH_PATH}/match/match-tvm/python")
sys.path.append(f"{MATCH_PATH}/match/zigzag")
sys.path.append(f"{MATCH_PATH}/match")
sys.path.append(".")

import match
from match.utils.utils import get_default_inputs
from match.relay.utils.utils import create_random_array
import tvm
from tvm import relay
from match.model.model import MatchModel
from carfield import Carfield

# INPUT_FILE_PATH = "/home/moyne/phd/yolo/yolov5relu-tristan-industrial/exp/qat21/golden/input_quantizer.txt"
ONNX_FILE_PATH = "models/tenk_temponet.onnx"
OUTPUT_DIR = "tenk_temponet"

# python3 test.py --executor graph --target pulp_open --model yolo_sanitized_uint8_fix --min_input_val 0 --max_input_val 0 --handle_out_fn handle_yolo_output --input_files /home/moyne/phd/yolo/yolov5relu-tristan-industrial/exp/qat21/golden/input_quantizer.txt

def create_dense_ex(
    inp_features:int=256, out_features:int=128,
    activation:bool=True, requant_pattern:bool=False,
    right_shift:int=1,**kwargs
):
    """Generate a small network in TVM Relay IR that performs a requantized convolution
    """
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType((1,inp_features), "uint8"))
    # Get or generate weight_values
    weights = create_random_array((out_features,inp_features),"int8", )
    # Get or generate bias values
    bias = create_random_array((out_features,), "int32", min_val=-2500, max_val=2500)
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
    if activation:
        if requant_pattern:
            x = relay.op.right_shift(x, relay.const(right_shift))
            x = relay.op.clip(x, a_min=0, a_max=255)
            x = relay.op.cast(x, "uint8")
        else:
            x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

# relay_mod, relay_params = match.get_relay_network(filename=ONNX_FILE_PATH)
relay_mod, relay_params = create_dense_ex(
    inp_features=8, out_features=8,
    activation=True, requant_pattern=True,
    right_shift=1
)

oenne_model = MatchModel(
    # relay_mod = relay_mod,
    # relay_params = relay_params,
    # model_name="densedemmerda",
    # executor="aot",
    filename=ONNX_FILE_PATH,
    model_type="onnx",
    model_name = OUTPUT_DIR,
    executor="graph",
    # default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[INPUT_FILE_PATH]),
    # handle_out_fn="handle_yolo_output",
    debug=True
)
target = Carfield()
# target.disable_exec_module("pulp_cluster")
match.match(
    model = oenne_model,
    target = target,
    output_path = OUTPUT_DIR,
)