#!/usr/bin/env python3

import sys
import argparse
import tvm
from tvm import relay
import match
from match.relay.utils.utils import create_random_array
from match.utils.utils import get_default_inputs
from match.model.model import MatchModel as Model
import random
from astral import Astral

__HOST_ONLY__ = False

# ONNX_FILE_PATH = "/home/moyne/phd/aicomp/match/test/models/anomaly_detection.onnx"
ONNX_FILE_PATH = "./isolde_net/isolde.onnx"
INPUT_FILE_PATH ="./isolde_net/features/input_quantizer.txt"
OUTPUT_DIR_PATH = "output" + ("_host" if __HOST_ONLY__ else "_sauria")

relay_mod, relay_params = match.get_relay_network(
    filename = ONNX_FILE_PATH
)

match_model = Model(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[
        INPUT_FILE_PATH
    ]),
    debug = False,
    debug_fallback = False,
    profile = False,
    profile_fallback = False,
)

target = Astral()
if __HOST_ONLY__:
    target.disable_exec_module("sauria")

match.match(
    model = match_model,
    target = target,
    output_path = OUTPUT_DIR_PATH,
)   
