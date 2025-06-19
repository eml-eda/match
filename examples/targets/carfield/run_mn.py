import sys

MATCH_PATH = "../../../.."
sys.path.append(f"{MATCH_PATH}/match/match-tvm/python")
sys.path.append(f"{MATCH_PATH}/match/zigzag")
sys.path.append(f"{MATCH_PATH}/match")
sys.path.append(".")

import match
from match.utils.utils import get_default_inputs
from match.model.model import MatchModel
from carfield import Carfield

INPUT_FILE_PATH = "model_fp16/cifar10_resnet8_fp16/input.txt"
ONNX_FILE_PATH = "model_fp16/cifar10_resnet8_fp16/model_fp16_nchw.onnx"
OUTPUT_DIR = "output_mn"

relay_mod, relay_params = match.get_relay_network(filename=ONNX_FILE_PATH)

oenne_model = MatchModel(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[INPUT_FILE_PATH]),
    handle_out_fn="handle_fp16_classifier",
    debug = True,
    debug_fallback = True
)
match.match(
    model = oenne_model,
    target = Carfield(),
    output_path = OUTPUT_DIR,
)