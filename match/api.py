import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from match.driver.driver import driver
import logging
import argparse

def with_relay(mod,params,target_name):
    """Compile a network defined already in TVM Relay IR for target

    Args:
        mod (tvm.ir.IRModule): network to compile
        params (List[Any]): arguments of the network(weights etc.)
        target_name (str): name of the target
    """
    driver(mod,params,target=target_name)

def with_onnx(onnx_model,target_name):
    """_summary_

    Args:
        onnx_model (ONNXModel): network representes in ONNX format
        target_name (str): name of the target
    """
    mod, params=relay.frontend.from_onnx(onnx_model)
    driver(mod,params,target=target_name)

def relay_conv(target):
    from match.relay_models import create_model_conv_2d
    mod, params = create_model_conv_2d()
    driver(mod,params,target=target)

def relay_add_convs(target):
    from match.relay_models import create_model_add_convs
    mod, params = create_model_add_convs()
    driver(mod,params,target=target)

def main(onnx_filename,target):
    onnx_model=onnx.load(onnx_filename)
    mod, params=relay.frontend.from_onnx(onnx_model)
    driver(mod,params,target=target)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='increase log verbosity')

    parser.add_argument('-d',
                        '--debug',
                        action='store_const',
                        dest='verbose',
                        const=2,
                        help='log debug messages (same as -vv)')

    parser.add_argument('-f','--onnxfilename',dest='onnx_filename',type=str,
                        help='Provide the filename of the ONNX module to compile.')

    parser.add_argument('-t','--target', dest='target', type=str,
                        help='Target platform for the inference of the DNN.')

    parser.add_argument('-c','--convexample',dest='convexample',action="store_true",
                        help="compile a simple conv example, that contains a con2d, a bias add and a requantization step")

    parser.add_argument('-a','--addexample',dest='addexample',action="store_true",
                        help="compile a simple add example between 2 2d convs like the ones in the convexample,"+\
                        "with a final requantization step")

    args = parser.parse_args()

    if args.convexample:
        relay_conv(args.target)
    elif args.addexample:
        relay_add_convs(args.target)
    else:
        main(args.onnx_filename,args.target)