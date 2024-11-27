from match.target.target import MatchTarget
from match.target.llmcpu.llmcpuex import LLMCpuEx

class LLMCpu(MatchTarget):
    def __init__(self,**kwargs):
        super(LLMCpu,self).__init__([
            LLMCpuEx(**kwargs),
        ],name="LLMCpu",**kwargs)
        # weird stats gathered, using USMP with riscv_cpu gets better results than USMP with arm_cpu
        # but the best overall results are with USMP disabled and arm_cpu
        self.static_mem_plan=False
        self.cpu_type="arm_cpu"