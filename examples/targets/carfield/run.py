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
from match.matcha.model import MatchaModel as Model

from carfield import Carfield

INPUT_FILE_PATH = "model_fp16/cifar10_resnet8_fp16/input.txt"
ONNX_FILE_PATH = "model_fp16/cifar10_resnet8_fp16/model_fp16_nchw.onnx"
OUTPUT_DIR_PATH = "output"

argparser = argparse.ArgumentParser()
argparser.add_argument("-o", "--output_dir", type=str, default=OUTPUT_DIR_PATH, help="Directory to save the output files")
argparser.add_argument("-i", "--input_file", type=str, default=INPUT_FILE_PATH, help="Input file path")
argparser.add_argument("-m", "--model_file", type=str, default=ONNX_FILE_PATH, help="ONNX model file path")

args = argparser.parse_args()

print(f"Using model file: '{args.model_file}'")
print(f"Using input file: '{args.input_file}'")
print(f"Using output dir: '{args.output_dir}'")

relay_mod, relay_params = match.get_relay_network(filename=args.model_file)

oenne_model = Model(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, input_files=[args.input_file]),
    handle_out_fn="handle_fp16_classifier",
    debug = False,
    debug_fallback = False,
    profile = True,
    profile_fallback = True,
)
match.match(
    model = oenne_model,
    target = Carfield(),
    output_path = args.output_dir,
)   