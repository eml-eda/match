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

INPUT_FILE_PATH = "model/input.txt"
ONNX_FILE_PATH = "model/model.onnx"
OUTPUT_DIR = "output"

relay_mod, relay_params = match.get_relay_network(filename=ONNX_FILE_PATH)

oenne_model = MatchModel(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[INPUT_FILE_PATH]),
    handle_out_fn="handle_int_classifier",
    debug=True
)
match.match(
    model = oenne_model,
    target = Carfield(),
    output_path = OUTPUT_DIR,
)