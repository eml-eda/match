from match.target.gap9.ne16 import Gap9NE16
from match.target.target import MatchTarget
from match.target.gap9.cluster import Gap9Cluster

class Gap9(MatchTarget):
    def __init__(self,**kwargs):
        super(Gap9,self).__init__([
            Gap9Cluster(**kwargs),
            Gap9NE16(**kwargs)
        ],**kwargs,name="gap9")
        # like with DIANA this configuration should be faster
        self.static_mem_plan=False
        self.cpu_type="arm_cpu"
