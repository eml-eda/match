#!/usr/bin/env python3

import sys

MATCH_PATH = "../../../.."
sys.path.append(f"{MATCH_PATH}/match/match-tvm/python")
sys.path.append(f"{MATCH_PATH}/match/zigzag")
sys.path.append(f"{MATCH_PATH}/match")
sys.path.append(".")

import argparse

import match
from match.utils.utils import get_default_inputs
from match.model.model import MatchModel as Model

from astral import Astral

NET = "model_fp16/model_graph.relay"
PARAMS = "model_fp16/model_params.txt"
OUTPUT_DIR_PATH = "output"

relay_mod, relay_params = match.get_relay_network(input_type="relay", filename=NET, params_filename=PARAMS)

oenne_model = Model(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[]),
    handle_out_fn="handle_fp16_classifier",
    debug = False,
    debug_fallback = False,
    profile = True,
    profile_fallback = True,
)
target = Astral()
# target.disable_exec_module("pulp_cluster")
match.match(
    model = oenne_model,
    target = target,
    output_path = OUTPUT_DIR_PATH,
)   
