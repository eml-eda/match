from match.target import DefaultMatchTarget, MatchTarget
import os
from typing import Dict
import tvm
from match.relay.utils.utils import create_build_dir
import pathlib

class MatchCompiler:
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "./match_output",
                 lib_name: str = "c",
                 no_of_inputs: int = 1,
                 target: MatchTarget=DefaultMatchTarget(),
                 mod_name: str="default"):
        self.relay_model = mod
        self.relay_params = params 
        self.build_dir = build_dir
        self.no_of_inputs = no_of_inputs
        self.target = target
        self.mod_name = mod_name
        self.lib_name = lib_name
        self.lib_path = os.path.dirname(__file__)+"/../libs/"+self.lib_name
        ## Placeholders in case profiling code is added
        create_build_dir(self.build_dir, self.lib_path,target=self.target)

    def tvm_compile(self, 
                    target_additional_options: Dict[str,str] = {"requant_transform":"0"},
                    fusion: bool = True,
                    ):
        pass