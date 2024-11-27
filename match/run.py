from pathlib import Path
import shutil
from typing import Dict, List
from match.model.dynamic_dim import DynamicDim
from match.model.model import MatchModel,build_runtime, get_match_inputs_and_outputs
from match.relay.compiled_module import CompiledModule
from match.relay.models import create_model_add_convs, create_model_conv_2d
from match.target.get_target import get_target, reset_target, set_target
import argparse
from match.relay.get_relay import get_relay_from
from match.utils.utils import set_output_path


def match(input_type="onnx", models_to_compile:List[MatchModel]=[],
          filename=None, params_filename=None,
          target=None, target_name=None,
          dynamic_dims:Dict[str,DynamicDim]={},
          match_inputs=None,match_outputs=None,
          output_path="./match_output"):
    if Path(output_path).absolute().is_dir():
        # remove build folder and all contents
        shutil.rmtree(Path(output_path).absolute())
    # make the build folder again
    Path(output_path).absolute().mkdir(parents=True)
    set_output_path(str(Path(output_path).absolute()))
    if len(models_to_compile)==0:    
        models_to_compile,match_inputs,match_outputs,dynamic_dims = get_relay_from(input_type,filename,params_filename)
    #add_save_relay(prefix="start",mod=relay_mod,params=relay_params)
    reset_target()
    if target!=None:
        set_target(target=target)
    target=get_target(target_name=target_name)
    results = {}
    for model_to_compile in models_to_compile:
        model_to_compile.compile_model(target=target,out_path=output_path)
        model_to_compile.move_static_app_to(out_path=output_path)
        results[model_to_compile.name] = CompiledModule.result
    
    runtime="default"
    if len(dynamic_dims)>0:
        runtime="generative"
    
    if match_inputs is None or match_outputs is None:
        match_inputs,match_outputs=get_match_inputs_and_outputs(models_to_compile)
    
    build_runtime(models_to_compile,dynamic_dims,match_inputs,match_outputs,runtime,output_path)
    
    target.gen_libs_and_main(match_inputs=match_inputs,match_outputs=match_outputs,dynamic_dims=dynamic_dims,runtime=runtime,out_path=output_path)
    return results

def get_relay_network(input_type="onnx",filename="examples/temponet_ht.onnx",params_filename=None):
    return get_relay_from(input_type,filename,params_filename)


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
    elif args.addexample:
        mod,params=create_model_add_convs()
        
    match(
        input_type=input_type,
        models_to_compile=[] if mod is None else [MatchModel(
            relay_mod=mod,
            relay_params=params,
        )],
        filename=filename,
        params_filename=params_filename,
        target=target,
        target_name=target_name,
        output_path=output_path,
    )
