from match.target.gap9.ne16 import Gap9NE16
from match.target.target import MatchTarget
from match.target.gap9.cluster import Gap9Cluster

class Gap9(MatchTarget):
    def __init__(self):
        super(Gap9,self).__init__([
            Gap9Cluster(),
            Gap9NE16()
        ],name="gap9")
        self.disabled_exec_modules=["NE16"]