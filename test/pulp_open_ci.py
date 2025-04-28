import subprocess
import pytest
import match
from match.model.model import MatchModel
from targets.pulp_open import PulpOpen
from match.utils.utils import get_default_inputs
import numpy as np
from quants import create_conv_ex, create_dense_ex

CHECK_STR = "output differs from checksum by"
EXPECTED_STR = "output differs from checksum by 0"
MIN_INPUT_VAL = 0
MAX_INPUT_VAL = 2

def utils_pulp_open_run_and_get_checksum_diffs(output_path):
    subprocess.run(
        ["make", "all"],
        cwd=output_path,
    )
    # Open the output file for writing
    with open(f"{output_path}/output.txt", "w") as output_file:
        # Run the command and capture both stdout and stderr
        process = subprocess.Popen(
            ["make", "run"],
            cwd=output_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Stream output to both the file and the screen
        for line in process.stdout:
            print(line, end="")  # Print to screen
            output_file.write(line)  # Write to file
        
        process.wait()  # Wait for the process to complete
    with open(f"{output_path}/output.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if CHECK_STR in line:
                assert EXPECTED_STR in line, f"Checksum mismatch found! {line}"

def test_big_conv2d(tmp_path):
    # Example test for big_conv
    inp_width = 64
    inp_height = 64
    # set random to 0 for weights and constants generation...
    np.random.seed(0)
    mod, params = create_conv_ex(
        input_shape=(inp_height, inp_width),
        inp_ch=3,
        out_ch=64,
        fil_shape=(3,3),
        stride=(1,1),
        padding=(1,1,1,1),
        requant_pattern=True,
        nhwc=True,
        groups=1,
        right_shift=15,
    )
    match.match(
        model=MatchModel(
            relay_mod=mod, relay_params=params,
            model_name="big_conv2d", executor="graph",
            default_inputs=get_default_inputs(
                mod=mod, params=params,
                min_input_val=MIN_INPUT_VAL, max_input_val=MAX_INPUT_VAL
            ),
            debug=True
        ),
        target=PulpOpen(),
        output_path=tmp_path,
    )
    utils_pulp_open_run_and_get_checksum_diffs(tmp_path)

def test_big_strided_conv2d(tmp_path):
    # Example test for big_conv
    inp_width = 64
    inp_height = 64
    # set random to 0 for weights and constants generation...
    np.random.seed(0)
    mod, params = create_conv_ex(
        input_shape=(inp_height, inp_width),
        inp_ch=3,
        out_ch=64,
        fil_shape=(3,3),
        stride=(2,2),
        padding=(1,1,1,1),
        requant_pattern=True,
        nhwc=True,
        groups=1,
        right_shift=15,
    )
    match.match(
        model=MatchModel(
            relay_mod=mod, relay_params=params,
            model_name="big_strided_conv2d", executor="graph",
            default_inputs=get_default_inputs(
                mod=mod, params=params,
                min_input_val=MIN_INPUT_VAL, max_input_val=MAX_INPUT_VAL
            ),
            debug=True
        ),
        target=PulpOpen(),
        output_path=tmp_path,
    )
    utils_pulp_open_run_and_get_checksum_diffs(tmp_path)

def test_big_dense(tmp_path):
    # set random to 0 for weights and constants generation...
    np.random.seed(0)
    mod, params = create_dense_ex(
        inp_features=256,
        out_features=128,
        requant_pattern=True,
        right_shift=15,
    )
    match.match(
        model=MatchModel(
            relay_mod=mod, relay_params=params,
            model_name="big_dense", executor="graph",
            default_inputs=get_default_inputs(
                mod=mod, params=params,
                min_input_val=MIN_INPUT_VAL, max_input_val=MAX_INPUT_VAL
            ),
            debug=True
        ),
        target=PulpOpen(),
        output_path=tmp_path,
    )
    utils_pulp_open_run_and_get_checksum_diffs(tmp_path)

# def test_dw_conv2d(tmp_path):
#     # Example test for dw_conv2d
#     inp_width = 64
#     inp_height = 64
#     # set random to 0 for weights and constants generation...
#     np.random.seed(0)
#     mod, params = create_conv_ex(
#         input_shape=(inp_height, inp_width),
#         out_ch=16,
#         inp_ch=16,
#         fil_shape=(3,3),
#         stride=(1,1),
#         padding=(1,1,1,1),
#         requant_pattern=True,
#         nhwc=True,
#         groups=16,
#         right_shift=15,
#     )
#     match.match(
#         model=MatchModel(
#             relay_mod=mod, relay_params=params,
#             model_name="dw_conv2d", executor="graph",
#             default_inputs=get_default_inputs(
#                 mod=mod, params=params,
#                 min_input_val=MIN_INPUT_VAL, max_input_val=MAX_INPUT_VAL
#             ),
#             debug=True
#         ),
#         target=PulpOpen(),
#         output_path=tmp_path,
#     )
#     utils_pulp_open_run_and_get_checksum_diffs(tmp_path)