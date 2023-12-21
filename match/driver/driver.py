from tvm.driver.tvmc.model import TVMCModel
import os
from typing import Dict,List
from abc import ABC, abstractmethod
import tvm
import numpy as np
from match.target import get_target
from match.relay_utils import utils
import subprocess
import pathlib
import shutil
import argparse

import tvm.relay as relay


class MatchDriver:
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "./outputs/last_build",
                 no_of_inputs: int = 1,
                 target: str = "gap9",):
        self.model = TVMCModel(mod, params)
        self.build_dir = build_dir
        self.no_of_inputs = no_of_inputs
        self.target = target
        self.target_inst=get_target(self.target)
        ## Placeholders in case profiling code is added
        utils.create_build_dir(self.build_dir, os.path.dirname(__file__)+"/../codegen/template/lib",target=self.target_inst)
        
    def tvm_compile(self, 
                    target_additional_options: str = "",
                    fusion: bool = True,
                    init_value: int = 1,
                    indefinite: bool = False,
                    boot_cluster: bool = True,
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
        devices_target=f'-target="{self.target}"' if self.target!="" else ""
        utils.tvmc_compile_and_unpack(self.model, 
                                      target=f'match -requant_transform=0 {devices_target} {target_additional_options} ,c',
                                      fuse_layers=fusion,
                                      build_path=self.build_dir)


def driver(mod: tvm.ir.IRModule, 
           params: Dict[str, tvm.nd.array],
           target: str="gap9"):
    """Compile a model for MATCH

    Args:
        mod (tvm.ir.IRModule): network to compile in TVM Relay IR format
        params (Dict[str, tvm.nd.array]): params of the network
        target (str, optional): name of the target for the compilation. Defaults to "gap9".
    """
    match_driver = MatchDriver(mod, params,
                          target=target)
    match_driver.tvm_compile(fusion=True)