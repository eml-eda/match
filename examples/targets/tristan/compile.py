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

ONNX_FILE_PATH = "isolde.onnx"
INPUT_FILE_PATH ="input.txt"
OUTPUT_DIR_PATH = "output"

relay_mod, relay_params = match.get_relay_network(
    filename = ONNX_FILE_PATH
)

oenne_model = Model(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[INPUT_FILE_PATH]),
    debug = False,
    debug_fallback = False,
    profile = False,
    profile_fallback = False,
)
target = Astral()
match.match(
    model = oenne_model,
    target = target,
    output_path = OUTPUT_DIR_PATH,
)   
