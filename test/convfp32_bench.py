import os
import numpy as np
import subprocess
import itertools
import pytest

import match
from match.model.model import MatchModel
from targets.pulp_open import PulpOpen
from targets.GAP9 import GAP9

from floats import create_fp_conv_ex, create_fp_conv_transpose_ex

MAX_NODE_SIZE = 2**20  # 1MB, adjust as needed

def run_pulp_open_test(test_path, model_name):
    if not (test_path / f"src/nodes/{model_name}/main_0_params.c").exists():
        Warning(f"This test doesn't contain any MATCH-accelerated node. ")
        return []
    subprocess.run(["make", "clean"], cwd=test_path)
    subprocess.run(["make", "all"], cwd=test_path, capture_output=True)
    process = subprocess.Popen("make run", cwd=test_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    errors = list()
    correct = list()
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        if "relative error between output and checksum by" in line:
            if "0.0000" not in line:
                errors.append(line)
            else:
                correct.append(line)
        print(line.rstrip(), flush=True)

    process.wait()
    subprocess.run(["make", "clean"], cwd=test_path)  # Clean up after test
    print(f"Errors: {errors}, Correct: {correct}")
    print(f"Return code: {process.returncode}")
    passed = process.returncode == 0 and len(errors) == 0
    if passed and not any("MATCH node" in line for line in correct):
        Warning(f"No MATCH node found in output, but test passed. "
                "This might indicate that the test did not run as expected.")
    if not passed:
        errors.append(f"Test failed with return code {process.returncode}. ")
    return errors

def run_gap_test(test_path, model_name):
    if not (test_path / f"src/nodes/{model_name}/main_0_params.c").exists():
        Warning(f"This test doesn't contain any MATCH-accelerated node. ")
        return []
    subprocess.run(["make", "clean"], cwd=test_path)
    subprocess.run(["make", "all", "-j16"], cwd=test_path, capture_output=True)
    process = subprocess.Popen("make run", cwd=test_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    errors = list()
    correct = list()
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        if "relative error between output and checksum by" in line:
            if "0.0000" not in line:
                errors.append(line)
            else:
                correct.append(line)
        print(line.rstrip(), flush=True)

    process.wait()
    subprocess.run(["make", "clean"], cwd=test_path)  # Clean up after test
    print(f"Errors: {errors}, Correct: {correct}")
    print(f"Return code: {process.returncode}")
    passed = process.returncode == 0 and len(errors) == 0
    if passed and not any("MATCH node" in line for line in correct):
        Warning(f"No MATCH node found in output, but test passed. "
                "This might indicate that the test did not run as expected.")
    if not passed:
        errors.append(f"Test failed with return code {process.returncode}. ")
    return errors

def compute_node_size(inp_shapes, out_chs, fil_shape, strides):
    inp_ch, inp_h, inp_w = inp_shapes
    fil_h, fil_w = fil_shape
    out_h = (inp_h + 2 * fil_h - 1) // strides[0] + 1
    out_w = (inp_w + 2 * fil_w - 1) // strides[1] + 1
    is_dw = inp_ch == out_chs and inp_ch > 1
    node_size = 4 * ((out_chs * (inp_ch // (1 if not is_dw else inp_ch)) * fil_h * fil_w ) + (out_chs * out_h * out_w) + (inp_ch * inp_h * inp_w))
    return node_size


def conv_get_test_params():   
    strides = [
        (1, 1),
        (2, 2),
    ]
    fil_shape_and_padding = [
        [(1, 1), (0, 0)],
        [(3, 3), (1, 1)],
    ]
    out_chs = [32, 64, 256]
    inp_shapes = []
    for inp_ch in [32, 64, 256]:
        for inp_h in [64, 256]:
            for inp_w in [64, 256]:
                inp_shapes.append((inp_ch, inp_h, inp_w))

    combination = [strides, fil_shape_and_padding, out_chs, inp_shapes]
    test_params = [(strides, fil_shape_and_padding, out_chs, inp_shapes, compute_node_size(inp_shapes, out_chs, fil_shape_and_padding[0], strides)) for (strides, fil_shape_and_padding, out_chs, inp_shapes) in itertools.product(*combination) if compute_node_size(inp_shapes, out_chs, fil_shape_and_padding[0], strides) < MAX_NODE_SIZE]
    test_ids = [f"ic_{inp_shapes[0]}_ih_{inp_shapes[1]}_iw_{inp_shapes[2]}_fh_{fil_shape_and_padding[0][0]}_fw_{fil_shape_and_padding[0][1]}_oc_{out_chs}_sh_{strides[0]}_sw_{strides[1]}_nodesize_{node_size}" for (strides, fil_shape_and_padding, out_chs, inp_shapes, node_size) in test_params]
    return test_params, test_ids

def conv_transpose_get_test_params():   
    strides = [
        (1, 1),
        # (2, 2),
    ]
    fil_shape_and_padding = [
        [(1, 1), (0, 0)],
        # [(3, 3), (1, 1)],
    ]
    out_chs = [64]
    inp_shapes = []
    for inp_ch in [64]:
        for inp_h in [25]:
            for inp_w in [5]:
                inp_shapes.append((inp_ch, inp_h, inp_w))

    combination = [strides, fil_shape_and_padding, out_chs, inp_shapes]
    test_params = [(strides, fil_shape_and_padding, out_chs, inp_shapes, compute_node_size(inp_shapes, out_chs, fil_shape_and_padding[0], strides)) for (strides, fil_shape_and_padding, out_chs, inp_shapes) in itertools.product(*combination) if compute_node_size(inp_shapes, out_chs, fil_shape_and_padding[0], strides) < MAX_NODE_SIZE]
    test_ids = [f"ic_{inp_shapes[0]}_ih_{inp_shapes[1]}_iw_{inp_shapes[2]}_fh_{fil_shape_and_padding[0][0]}_fw_{fil_shape_and_padding[0][1]}_oc_{out_chs}_sh_{strides[0]}_sw_{strides[1]}_nodesize_{node_size}" for (strides, fil_shape_and_padding, out_chs, inp_shapes, node_size) in test_params]
    return test_params, test_ids

test_params, test_ids = conv_get_test_params()
@pytest.mark.parametrize("test_params", test_params, ids=test_ids)
def test_fp_conv2d(test_params, tmp_path):
    strides, fil_shape_and_padding, out_chs, inp_shapes, node_size = test_params
    # set random to 0 for weights and constants generation...
    np.random.seed(0)
    inp_ch = inp_shapes[0]
    inp_shape = inp_shapes[1:]
    is_dw = inp_ch == out_chs and inp_ch > 1
    mod, params =create_fp_conv_ex(
        inp_shape=inp_shape,
        fil_shape=fil_shape_and_padding[0],
        padding=fil_shape_and_padding[1],
        strides=strides,
        groups=1 if not is_dw else inp_ch,
        out_ch=out_chs,
        inp_ch=inp_ch
    )
    target = PulpOpen()
    model_name = "fp_conv2d_test"

    match.match(
        model=MatchModel(
           relay_mod=mod, relay_params=params,
           model_name=model_name, executor="graph",
           golden_cpu_model=False,
           debug=True,
        ),
        target=target,
        output_path=tmp_path
    )

    ret_test = run_pulp_open_test(test_path=tmp_path, model_name=model_name)

    assert len(ret_test) == 0, f"Test failed for fp_conv2d with params: {test_params}, path: {tmp_path}, errors: {ret_test}"

test_params, test_ids = conv_transpose_get_test_params()
@pytest.mark.parametrize("test_params", test_params, ids=test_ids)
def test_fp_conv2d_transpose(test_params, tmp_path):
    strides, fil_shape_and_padding, out_chs, inp_shapes, node_size = test_params
    # set random to 0 for weights and constants generation...
    np.random.seed(0)
    inp_ch = inp_shapes[0]
    inp_shape = inp_shapes[1:]
    is_dw = inp_ch == out_chs and inp_ch > 1
    mod, params =create_fp_conv_transpose_ex(
        inp_shape=inp_shape,
        fil_shape=fil_shape_and_padding[0],
        padding=fil_shape_and_padding[1],
        strides=strides,
        groups=1 if not is_dw else inp_ch,
        out_ch=out_chs,
        inp_ch=inp_ch
    )
    target = GAP9()
    model_name = "fp_conv2d_transpose_test"

    match.match(
        model=MatchModel(
           relay_mod=mod, relay_params=params,
           model_name=model_name, executor="graph",
           golden_cpu_model=False,
           debug=True,
        ),
        target=target,
        output_path=tmp_path
    )

    ret_test = run_gap_test(test_path=tmp_path, model_name=model_name)

    assert len(ret_test) == 0, f"Test failed for fp_conv2d with params: {test_params}, path: {tmp_path}, errors: {ret_test}"


if __name__ == "__main__":
    inp_shapes = ( 32, 8, 8)
    out_chs = 8
    fil_shape_and_padding = [( 3, 3), ( 1, 1)]
    strides = (1, 1)
    test_params = (strides, fil_shape_and_padding, out_chs, inp_shapes)
    tmp_path = os.path.dirname(__file__)+"/builds/last_build"
    test_fp_conv2d(test_params, tmp_path)
    print("Test completed successfully.")