from match.target.target import DefaultMatchTarget, MatchTarget
from tvm.driver.tvmc.model import TVMCModel
import os
from typing import Dict
import tvm
from match.relay.utils.utils import tvmc_compile_and_unpack,create_build_dir
import pathlib


class MatchDriver:
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "./match_output",
                 no_of_inputs: int = 1,
                 target: MatchTarget=DefaultMatchTarget()):
        self.model = TVMCModel(mod, params)
        self.build_dir = build_dir
        self.no_of_inputs = no_of_inputs
        self.target = target
        ## Placeholders in case profiling code is added
        create_build_dir(self.build_dir, os.path.dirname(__file__)+"/../codegen/template/lib",target=self.target)
        
    def tvm_compile(self, 
                    target_additional_options: Dict[str,str] = {"requant_transform":"0"},
                    fusion: bool = True,
                    ):
        """Compiles network to C code with TVM

        This is a wrapper method around TVMC that generates C code for the 
        network and C code that calls the network.
        All output is stored in self.build_dir

        :param target: target parameter passed to TVMC
        :param fusion: Enable/Disable operator fusion pass for TVM generated
            kernels
        :param init_value: input value set in calling wrapper
        :param indefinite: put infinite loop around TVM network. Useful for
            power measurements.
        :param boot_cluster: put cluster cores boot code in C wrapper before
            calling TVM generated code.
        """
        target_options=""
        for key,val in target_additional_options.items():
            target_options+=f"-{key}={val} "
        tvmc_compile_and_unpack(self.model, 
                                      target=f'match {target_options} ,c',
                                      fuse_layers=fusion,
                                      build_path=self.build_dir)


def driver(mod: tvm.ir.IRModule, 
           params: Dict[str, tvm.nd.array],
           target: MatchTarget=DefaultMatchTarget(),
           output_path="./match_output"):
    """Compile a model for MATCH

    Args:
        mod (tvm.ir.IRModule): network to compile in TVM Relay IR format
        params (Dict[str, tvm.nd.array]): params of the network
        target (MatchTarget, optional): target instance for the compilation. Defaults to DefaultMatchTarget.
    """
    match_driver = MatchDriver(mod, params,
                          target=target,
                          build_dir=output_path)
    match_driver.tvm_compile(fusion=True)