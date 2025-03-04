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