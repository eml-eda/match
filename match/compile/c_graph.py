import tarfile
from match.compile.compiler import MatchCompiler
from match.target import DefaultMatchTarget, MatchTarget
from tvm.driver.tvmc.model import TVMCModel
import os
from typing import Dict
import tvm
from match.relay.utils.utils import create_build_dir
import pathlib

def match_tvmc_graph_compile_wrapper(model: TVMCModel, target: str = "match, c",
                            fuse_layers: bool = True,
                            build_path: str = "./build",
                            cpu_type: str = "riscv_cpu",
                            static_mem_plan: bool = True,
                            static_mem_plan_algorithm: str = "hill_climb",
                            mod_name: str = "default",):
    '''
    Utility function that calls tvmc_wrapper and extracts output mlf
    (= TVM model library format) file.
    :param model: TVMC model that you wish to compile
    :param target: Can be "match, c" if you want to offload all possible
        computations to accelerator, and can be "c" for golden model checking.
    :param fuse_layers: sets relay.FuseOps.max_depth parameter to 1
        if set to False. This tells relay to not fuse operations.
        This can be useful when debuggin the TVM-generated c code kernels.
    :param build_path: path to export mlf file output to
    '''
    # Compile new model
    mlf_path = os.path.join(build_path, "model.tar")
    target += f" -device={cpu_type}"
    # This has to be set by default to use the C runtime
    """
    These are the existing configurations: tir.ReduceBranchingThroughOvercompute, tir.experimental_dma_bypass_cache,
    tir.reset_start_id, relay.collage.tvm_max_depth, tir.LoopPartition, tir.usmp.custom_algorithm, tir.instr_siblings,
    relay.FuseOps.max_depth, tir.debug_keep_trivial_loop, tir.InjectDoubleBuffer, tir.detect_global_barrier, testing.immutable_module,
    ir.enable_si_builder, tir.use_async_copy, relay.fallback_device_type, te.keep_schedule_record, tir.usmp.algorithm, tir.noalias,
    tir.disable_storage_rewrite, relay.collage.byoc_fusion_style, tir.Simplify, relay.frontend.fill_span, tir.usmp.use_workspace_io,
    tir.lwp_disable_func_prof, tir.RemoveNoOp, relay.backend.use_meta_schedule_dispatch, tir.disable_assert, tir.enable_debug,
    tir.add_lower_pass, tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements, relay.backend.tir_converter,
    relay.backend.use_auto_scheduler, tir.contrib.ethos-u.copy_compute_reordering_reorder_by_cycles,
    relay.ToMixedPrecision.keep_orig_output_dtype, tir.instrument_bound_checkers, tir.enable_equiv_terms_in_cse_tir, tir.HoistIfThenElse,
    tir.lwp_min_height, tir.instrument_lwp, relay.remove_standalone_reshapes.enable, tir.disable_cse_tir, tir.lwp_max_depth,
    relay.FuseOps.link_params, tir.UnrollLoop, relay.backend.use_meta_schedule, tir.vtcm_capacity, relay.collage.byoc_max_depth,
    tir.is_entry_func, tir.ptx_ldg32, tir.HoistExpression, tir.usmp.enable, tir.disable_vectorize
    """
    pass_context_configs = []
    # vectorize doesnt' work good with C
    pass_context_configs.append("tir.disable_vectorize=1")
    # enable static memory plan
    pass_context_configs.append(f"tir.usmp.enable={int(static_mem_plan)}")
    # algorithm to use for static memory plan
    #if static_mem_plan:
    pass_context_configs.append(f"tir.usmp.algorithm={static_mem_plan_algorithm}")
    #pass_context_configs.append("tir.disable_storage_rewrite=1")
    #pass_context_configs.append("tir.usmp.use_workspace_io=1")
    #pass_context_configs.append("tir.InjectDoubleBuffer=1")
    #pass_context_configs.append("relay.backend.disable_memory_plan=1")
    if not fuse_layers:
        pass_context_configs.append('relay.FuseOps.max_depth=1')

    tvmc_pkg = tvm.driver.tvmc.compiler.compile_model(tvmc_model=model,
        target=target,
        opt_level=3,
        executor=tvm.relay.backend.Executor("graph",
                        {
                            # "interface-api": "c",
                            # "unpacked-api": True,
                            #"workspace-byte-alignment": 4,
                        },
                        ),
        runtime=tvm.relay.backend.Runtime("crt"),
        output_format="tar",
        package_path=mlf_path,
        pass_context_configs=pass_context_configs,
        mod_name=mod_name,
    )
    # extract mlf file
    mlf = tarfile.TarFile(mlf_path)
    mlf.extractall(build_path)
    # remove the archive
    os.remove(mlf_path)

class MatchCompilerCGraph(MatchCompiler):

    def __init__(self, mod, params, build_dir = "./match_output", no_of_inputs = 1, target = ..., mod_name = "default"):
        super().__init__(mod=mod, params=params, build_dir=build_dir, lib_name="c", no_of_inputs=no_of_inputs, target=target, mod_name=mod_name)
        self.model = tvm.driver.tvmc.model.TVMCModel(mod, params)

    def tvm_compile(self,target_additional_options: Dict[str,str] = {"requant_transform":"0"},
                    fusion: bool = True,):
        target_options=""
        for key,val in target_additional_options.items():
            target_options+=f"-{key}={val} "
        match_tvmc_graph_compile_wrapper(self.model, 
                                      target=f'match {target_options} ,c',
                                      fuse_layers=fusion,
                                      build_path=self.build_dir,
                                      cpu_type=self.target.cpu_type,
                                      static_mem_plan=self.target.static_mem_plan,
                                      static_mem_plan_algorithm=self.target.static_mem_plan_algorithm,
                                      mod_name=self.mod_name,)