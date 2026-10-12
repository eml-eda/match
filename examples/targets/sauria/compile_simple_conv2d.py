#!/usr/bin/env python3
"""Small chain of int8 CONV2Ds written directly in Relay, to test the single SAURIA
conv2d kernel (sauria_conv2d_wrapper) without the rest of ISOLDE.

Same structure as isolde.onnx: every conv2d is int8 x int8 with out_dtype int32,
followed by a cast to int32 (matched by the sauria_conv2d pattern); the requantization
back to int8 between layers runs on the host. No padding: SAURIA does not support it."""

import numpy as np
import tvm
from tvm import relay
import match
from match.relay.utils.utils import create_random_array
from match.utils.utils import get_default_inputs
from match.model.model import MatchModel as Model
from astral import Astral

__HOST_ONLY__ = False
OUTPUT_DIR_PATH = "output_simple_conv2d" + ("_host" if __HOST_ONLY__ else "_sauria")

INPUT_SHAPE = (1, 4, 16, 16)  # NCHW
# (out_channels, kernel, stride, requant right shift)
LAYERS = [
    (8,  3, 1, 9),   # 4x16x16  -> 8x14x14
    (16, 3, 1, 10),  # 8x14x14  -> 16x12x12
    (16, 1, 1, 9),   # 16x12x12 -> 16x12x12
    (8,  3, 2, 10),  # 16x12x12 -> 8x5x5
]


def requant(x, shift):
    """int32 -> int8 in [0, 127], like the activations between ISOLDE layers"""
    x = relay.right_shift(x, relay.const(shift, "int32"))
    x = relay.clip(x, a_min=0, a_max=127)
    return relay.cast(x, "int8")


def create_conv2d_chain():
    x = relay.var("input_0", relay.TensorType(INPUT_SHAPE, "int8"))
    in_c = INPUT_SHAPE[1]
    for idx, (out_c, k, s, shift) in enumerate(LAYERS):
        w = relay.const(create_random_array((out_c, in_c, k, k), "int8", min_val=-8, max_val=8))
        x = relay.nn.conv2d(
            x, w,
            strides=(s, s), padding=(0, 0),
            channels=out_c, kernel_size=(k, k),
            data_layout="NCHW", kernel_layout="OIHW",
            out_dtype="int32",
        )
        x = relay.cast(x, "int32")
        if idx < len(LAYERS) - 1:
            x = requant(x, shift)
        in_c = out_c
    mod = tvm.ir.IRModule.from_expr(relay.Function(relay.analysis.free_vars(x), x))
    return relay.transform.InferType()(mod), {}


relay_mod, relay_params = create_conv2d_chain()
print(relay_mod)

match_model = Model(
    relay_mod = relay_mod,
    relay_params = relay_params,
    model_name = "model",
    default_inputs = get_default_inputs(mod=relay_mod, params=relay_params, min_input_val=0, max_input_val=127),
    debug = True,
    debug_fallback = True,
    profile = True,
    profile_fallback = True,
)

target = Astral()
if __HOST_ONLY__:
    target.disable_exec_module("sauria")

match.match(
    model = match_model,
    target = target,
    output_path = OUTPUT_DIR_PATH,
)
