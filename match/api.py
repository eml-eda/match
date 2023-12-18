import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from match.driver.driver import driver
import logging
import argparse

def with_relay(mod,params,device):
    driver(mod,params)

def main(onnx_filename,target):
    #onnx_model=onnx.load(onnx_file)
    #print(onnx_model)
    #mod, params=relay.frontend.from_onnx(onnx_model)
    #breakpoint()
    from relay_conv2d import create_model
    mod, params = create_model()
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

    args = parser.parse_args()
    #if args.verbose == 0:
    #    logging.getLogger().setLevel(level=logging.WARNING)
    #elif args.verbose == 1:
    #    logging.getLogger().setLevel(level=logging.INFO)
    #elif args.verbose == 2:
    #    logging.getLogger().setLevel(level=logging.DEBUG)

    main(args.onnx_filename,args.target)