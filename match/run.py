from pathlib import Path
import shutil
from typing import Dict
from match.model.model import MatchModel
from match.relay.compiled_module import CompiledModule
from match.relay.models import create_model_add_convs, create_model_conv_2d
from match.target.get_target import get_target, reset_target, set_target
import argparse
from match.relay.get_relay import get_relay_from
from match.target.target import MatchTarget
from match.utils.utils import set_output_path

def match_multi_model(
        models:Dict[str,MatchModel]={}, target: MatchTarget=None,
        output_path: str="./match_output",default_model: str="default",
        ):
    if Path(output_path).absolute().is_dir():
        # remove build folder and all contents
        shutil.rmtree(Path(output_path).absolute())
    # make the build folder again
    Path(output_path).absolute().mkdir(parents=True)
    set_output_path(str(Path(output_path).absolute()))
    # set for the session the correct target
    reset_target()
    set_target(target=target)
    target = get_target()

    results = dict()
    for model_name,model in models.items():
        model.compile(target=target,out_path=output_path)
        results[model_name] = CompiledModule.result
    
    target.gen_libs_and_main(models=models,default_model=default_model,out_path=output_path)
    # log_results(results)
    return results

def match(model: MatchModel=None, target: MatchTarget=None, output_path: str="./match_output"):
    return match_multi_model(models={model.model_name: model}, target=target, output_path=output_path, default_model=model.model_name)

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
        "-e",
        "--executor",
        dest="executor",
        type=str,
        default="aot",
        help="Type of executor to use for the compilation of the model",
    )

    parser.add_argument(
        "--dynamic_model",
        dest="dynamic_model",
        action="store_true",
        help="Flag to signal that the model is a dynamic one",
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
        "-n",
        "--name",
        dest="model_name",
        type=str,
        default="default",
        help="Provide the name of the model to compile.",
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

    parser.add_argument(
        "--relax",
        dest="relax",
        action="store_true",
    )

    args = parser.parse_args()
    input_type = args.input_type
    target_name = args.target
    target = None
    mod = None
    params = None
    filename = args.filename
    params_filename = args.params_filename
    output_path = args.output_path
    model_name = args.model_name
    executor = args.executor
    dynamic_model = args.dynamic_model

    if args.convexample:
        mod, params = create_model_conv_2d()
    elif args.addexample:
        mod, params = create_model_add_convs()
    
    if args.relax:
        # match_relax(
        #     input_type=input_type,
        #     relay_mod=mod,
        #     relay_params=params,
        #     filename=filename,
        #     params_filename=params_filename,
        #     target=target,
        #     target_name=target_name,
        #     output_path=output_path,
        # )
        print("[MATCH] Relax is not supported currently")
    else:
        if dynamic_model:
            print("[MATCH] Warning: dynamic inference is not supported properly, there may be some issues...")
        match(
            model=MatchModel(relay_mod=mod, relay_params=params,
                             filename=filename, params_filename=params_filename,
                             model_type=input_type, model_name=model_name,
                             executor=executor, is_model_dynamic=dynamic_model),
            target=get_target(target_name=target_name), output_path=output_path
        )
