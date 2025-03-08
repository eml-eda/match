from match.target.target import MatchTarget
from match.target.diana.digital import DigitalAccelerator

class Diana(MatchTarget):
    def __init__(self,**kwargs):
        super(Diana,self).__init__([
            DigitalAccelerator(**kwargs),
        ],name="diana",**kwargs)
        # weird stats gathered, using USMP with riscv_cpu gets better results than USMP with arm_cpu
        # but the best overall results are with USMP disabled and arm_cpu
        self.static_mem_plan=False
        self.cpu_type="arm_cpu"
