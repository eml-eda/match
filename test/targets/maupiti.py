from match.target.target import MatchTarget
from modules.maupiti.maupiti import MaupitiKernels


DEFAULT_MATCH_LIB = True


class MaupitiTarget(MatchTarget):
    def __init__(self):
        super(MaupitiTarget, self).__init__(
            [
                MaupitiKernels(),
            ],
            name="maupiti",
        )
        # set here makefile path and others if needed
        if not DEFAULT_MATCH_LIB:
            self.makefile_path = ""
            self.main_template_path = ""
        self.static_mem_plan = False
        self.cpu_type = "riscv"
        self.input_macros = ""
        self.include_list = ["maupiti"]