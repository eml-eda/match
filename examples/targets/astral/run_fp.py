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

INPUT_FILE_PATH = "model_fp/input.txt"
RELAY_FILE_PATH = "model_fp16/model_graph.relay"
RELAY_PARAMS_PATH = "model_fp16/model_params.txt"
OUTPUT_DIR = "output_fp"

relay_mod, relay_params = match.get_relay_network(input_type="relay", filename=RELAY_FILE_PATH, params_filename=RELAY_PARAMS_PATH)

oenne_model = MatchModel(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[INPUT_FILE_PATH]),
    #handle_out_fn="handle_int_classifier",
    debug=True
)
match.match(
    model = oenne_model,
    target = Carfield(),
    output_path = OUTPUT_DIR,
)