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
import tvm


def match(input_type="onnx", models_to_compile:List[MatchModel]=[],
          relay_mod: tvm.ir.IRModule=None, relay_params=None,
          filename=None, params_filename=None,
          target=None, target_name=None,
          dynamic_dims:Dict[str,DynamicDim]={},
          match_inputs=None,match_outputs=None,
          output_path="./match_output",
          golden_default_cpu_model: bool=False,
          benchmarking: bool=True, executor: str="aot"
          ):
    if Path(output_path).absolute().is_dir():
        # remove build folder and all contents
        shutil.rmtree(Path(output_path).absolute())
    # make the build folder again
    Path(output_path).absolute().mkdir(parents=True)
    set_output_path(str(Path(output_path).absolute()))
    if relay_mod is not None:
        models_to_compile = [MatchModel(relay_mod=relay_mod,relay_params=relay_params)]
    if len(models_to_compile)==0:    
        models_to_compile,match_inputs,match_outputs,dynamic_dims = get_relay_from(input_type,filename,params_filename)
    #add_save_relay(prefix="start",mod=relay_mod,params=relay_params)
    reset_target()
    if target!=None:
        set_target(target=target)
    target=get_target(target_name=target_name)
    results = {}
    models_compiled_added = []
    for model_to_compile in models_to_compile:
        model_to_compile.compile_model(target=target,out_path=output_path,executor=executor)
        model_to_compile.move_static_app_to(target=target,out_path=output_path,executor=executor)
        results[model_to_compile.name] = CompiledModule.result
        static_model_result = CompiledModule.result
        static_model_result.match_inputs,static_model_result.match_outputs = get_match_inputs_and_outputs(model_to_compile)
        if match_inputs is not None:
            static_model_result.match_inputs = match_inputs
        if match_outputs is not None:
            static_model_result.match_outputs = match_outputs
        if golden_default_cpu_model and model_to_compile.name=="default":
            model_cpu_to_compile = MatchModel(relay_mod=model_to_compile.relay_mod,relay_params=model_to_compile.relay_params,
                                              dynamic=False)
            model_cpu_to_compile.name = "golden_cpu_model"
            target_disabled_modules = target.disabled_exec_modules
            new_disabled_modules = []
            for ex_mod in target.exec_modules_dict:
                new_disabled_modules.append(ex_mod)
            target.disabled_exec_modules = new_disabled_modules
            model_cpu_to_compile.compile_model(target=target,out_path=output_path,executor=executor)
            model_cpu_to_compile.move_static_app_to(target=target,out_path=output_path,executor=executor)
            results[model_cpu_to_compile.name] = CompiledModule.result
            results[model_cpu_to_compile.name].match_inputs, results[model_cpu_to_compile.name].match_outputs = results["default"].match_inputs,results["default"].match_outputs
            target.disabled_exec_modules = target_disabled_modules
            models_compiled_added.append(model_cpu_to_compile)
    
    models_to_compile+=models_compiled_added
    
    runtime="default"
    if len(dynamic_dims)>0:
        runtime="generative"
    
    if match_inputs is None or match_outputs is None:
        match_inputs,match_outputs=results["default"].match_inputs,results["default"].match_outputs
    build_runtime(target=target,static_models=models_to_compile,
                  dynamic_dims=dynamic_dims,runtime=runtime,
                  match_inputs=match_inputs,match_outputs=match_outputs,
                  benchmarking=benchmarking,out_path=output_path)
    
    target.gen_libs_and_main(match_inputs=match_inputs,match_outputs=match_outputs,
                             static_models=models_to_compile,dynamic_dims=dynamic_dims,
                             runtime=runtime,out_path=output_path,benchmarking=benchmarking)
    return results

# from match.relay.utils.utils import create_build_dir
# from tvm.driver.tvmc.model import TVMCModel
# from tvm.driver.tvmc.compiler import compile_model
# from tvm.relay.backend import Executor, Runtime
# import os
# import tarfile
# from tvm.relax.transform import PatternPartition

# def match_relax(input_type="onnx",relay_mod=None, relay_params=None, filename=None, params_filename=None, target=None, target_name=None,output_path="./match_output"):
#     if relay_mod==None:    
#         relay_mod,relay_params=get_relay_from(input_type,filename,params_filename)
#     reset_output_path()
#     reset_relay_list()
#     reset_schedules()
#     set_output_path(output_path)
#     add_save_relay(prefix="start",mod=relay_mod,params=relay_params)
#     reset_target()
#     if target!=None:
#         set_target(target=target)
#     target=get_target(target_name=target_name)

#     tvmc_model = TVMCModel(mod, params)
#     ## Placeholders in case profiling code is added
#     create_build_dir(output_path, os.path.dirname(__file__)+"/codegen/template/lib",target=target)

#     target_options=""
#     target_additional_options = {"requant_transform":"0"}
#     for key,val in target_additional_options.items():
#         target_options+=f"-{key}={val} "

#     # Compile new model
#     mlf_path = os.path.join(output_path, "model.tar")
#     # extract mlf file
#     mlf = tarfile.TarFile(mlf_path)
#     mlf.extractall(output_path)
#     # remove the archive
#     os.remove(mlf_path)
#     '''
#     Utility wrapper for TVMC that sets supported
#     :param model: TVMC model that you wish to compile
#     :param target: Can be "match, c" if you want to offload all possible
#         computations to accelerator, and can be "c" for golden model checking.
#     :param fuse_layers: sets relay.FuseOps.max_depth parameter to 1
#         if set to False. This tells relay to not fuse operations.
#         This can be useful when debuggin the TVM-generated c code kernels.
#     '''
#     # Check arguments
#     # Add -device=arm_cpu as default device for TVM C codegen
#     # This will use the arm_cpu relay strategy as opposed to the x86 one.
#     target += f" -device={target.cpu_type}"
#     # This has to be set by default to use the C runtime
#     """
#     These are the existing configurations: tir.ReduceBranchingThroughOvercompute, tir.experimental_dma_bypass_cache,
#     tir.reset_start_id, relay.collage.tvm_max_depth, tir.LoopPartition, tir.usmp.custom_algorithm, tir.instr_siblings,
#     relay.FuseOps.max_depth, tir.debug_keep_trivial_loop, tir.InjectDoubleBuffer, tir.detect_global_barrier, testing.immutable_module,
#     ir.enable_si_builder, tir.use_async_copy, relay.fallback_device_type, te.keep_schedule_record, tir.usmp.algorithm, tir.noalias,
#     tir.disable_storage_rewrite, relay.collage.byoc_fusion_style, tir.Simplify, relay.frontend.fill_span, tir.usmp.use_workspace_io,
#     tir.lwp_disable_func_prof, tir.RemoveNoOp, relay.backend.use_meta_schedule_dispatch, tir.disable_assert, tir.enable_debug,
#     tir.add_lower_pass, tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements, relay.backend.tir_converter,
#     relay.backend.use_auto_scheduler, tir.contrib.ethos-u.copy_compute_reordering_reorder_by_cycles,
#     relay.ToMixedPrecision.keep_orig_output_dtype, tir.instrument_bound_checkers, tir.enable_equiv_terms_in_cse_tir, tir.HoistIfThenElse,
#     tir.lwp_min_height, tir.instrument_lwp, relay.remove_standalone_reshapes.enable, tir.disable_cse_tir, tir.lwp_max_depth,
#     relay.FuseOps.link_params, tir.UnrollLoop, relay.backend.use_meta_schedule, tir.vtcm_capacity, relay.collage.byoc_max_depth,
#     tir.is_entry_func, tir.ptx_ldg32, tir.HoistExpression, tir.usmp.enable, tir.disable_vectorize
#     """
#     pass_context_configs = []
#     # vectorize doesnt' work good with C
#     pass_context_configs.append("tir.disable_vectorize=1")
#     # enable static memory plan
#     pass_context_configs.append(f"tir.usmp.enable={int(target.static_mem_plan)}")
#     # algorithm to use for static memory plan
#     #if static_mem_plan:
#     pass_context_configs.append(f"tir.usmp.algorithm={target.static_mem_plan_algorithm}")
#     #pass_context_configs.append("tir.disable_storage_rewrite=1")
#     #pass_context_configs.append("tir.usmp.use_workspace_io=1")
#     #pass_context_configs.append("tir.InjectDoubleBuffer=1")
#     #pass_context_configs.append("relay.backend.disable_memory_plan=1")
#     if False:
#         pass_context_configs.append('relay.FuseOps.max_depth=1')
#     RelaxExecutor=Executor()
#     # Apply the pattern partitioning pass
#     from tvm.target.target import Executor, Runtime

#     executor = Executor("aot", {"interface-api": "c", "unpacked-api": True})
#     runtime = Runtime("crt")  # Using C runtime (crt)

#     target = "c -keys=cpu -runtime=crt -system-lib=1"  # Target C code generation
#     mod = PatternPartition(pattern_table=compiled_pattern_table)(mod)

#     compile_model(tvmc_model=tvmc_model,
#                   target=target,
#                   opt_level=3,
#                   executor=RelaxExecutor("aot",
#                                     {
#                                         "interface-api": "c",
#                                         "unpacked-api": True,
#                                         #"workspace-byte-alignment": 4,
#                                     },
#                                     ),
#                   runtime=Runtime("crt"),
#                   output_format="mlf",
#                   package_path=mlf_path,
#                   pass_context_configs=pass_context_configs,
#                   #desired_layout="NHWC",
#                   #desired_layout_ops=["nn.conv2d"]
#                   )
#     save_all_relay()
#     save_all_schedules()
#     return CompiledModule.result

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

    parser.add_argument(
        "--relax",
        dest="relax",
        action="store_true",
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
        pass
    else:
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
