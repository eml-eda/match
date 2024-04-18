from match.relay.compiled_module import CompiledModule
from match.relay.models import create_model_add_convs, create_model_conv_2d
from match.target.get_target import get_target, reset_target, set_target
from match.driver.driver import driver
import argparse
from match.relay.get_relay import get_relay_from

def match(input_type="onnx",relay_mod=None, relay_params=None, filename=None, params_filename=None, target=None, target_name=None,output_path="./match_output"):
    if relay_mod==None:    
        relay_mod,relay_params=get_relay_from(input_type,filename,params_filename)
    #print(f"Model received {relay_mod}")
    reset_target()
    if target!=None:
        set_target(target=target)
    target=get_target(target_name=target_name)
    driver(relay_mod, relay_params, target=target,output_path=output_path)
    return CompiledModule.result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase log verbosity"
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="verbose",
        const=2,
        help="log debug messages (same as -vv)",
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="input_type",
        type=str,
        help="Type of the input to compile, possible values are [onnx,relay].",
    )

    parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        type=str,
        help="Provide the filename of the module to compile.",
    )

    parser.add_argument(
        "-p",
        "--params",
        dest="params_filename",
        type=str,
        help="Provide the filename of the params needed by the module to compile.",
    )

    parser.add_argument(
        "-t",
        "--target",
        dest="target",
        type=str,
        help="Target platform for the inference of the DNN.",
    )

    parser.add_argument(
        "-c",
        "--convexample",
        dest="convexample",
        action="store_true",
        help="compile a simple 2d convolution example, that contains a con2d, a bias add and a requantization step",
    )

    parser.add_argument(
        "-a",
        "--addexample",
        dest="addexample",
        action="store_true",
        help="compile a simple add example between 2 2d convs like the ones in the convexample,"
        + "with a final requantization step",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        type=str,
        help="Provide the output path"
    )

    args = parser.parse_args()
    input_type=args.input_type
    target_name=args.target
    target=None
    mod=None
    params=None
    filename=args.filename
    params_filename=args.params_filename
    output_path=args.output_path

    if args.convexample:
        mod,params=create_model_conv_2d()
        #from tvm import relay as relay_tvm
        #with open("examples/quant_conv.relay","w") as mod_file:
        #    mod_file.write(relay_tvm.astext(mod))
        #with open("examples/params_quant_conv.txt","wb") as par_file:
        #    par_file.write(relay_tvm.save_param_dict(params=params))
    elif args.addexample:
        mod,params=create_model_add_convs()
        
    match(
        input_type=input_type,
        relay_mod=mod,
        relay_params=params,
        filename=filename,
        params_filename=params_filename,
        target=target,
        target_name=target_name,
        output_path=output_path,
    )
