

from match.compile.compiler import MatchCompiler

class MatchCompilerCGraph(MatchCompiler):

    def __init__(self, mod, params, build_dir = "./match_output", no_of_inputs = 1, target = ..., mod_name = "default"):
        super().__init__(mod=mod, params=params, build_dir=build_dir, lib_name="c", no_of_inputs=no_of_inputs, target=target, mod_name=mod_name)
        
    def tvm_compile(self, target_additional_options = ..., fusion = True):
        return super().tvm_compile(target_additional_options, fusion)