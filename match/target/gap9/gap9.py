from match.target.target import MatchTarget
from match.target.gap9.cluster import Gap9Cluster

class Gap9(MatchTarget):
    def __init__(self):
        super(Gap9,self).__init__([
            Gap9Cluster()
        ],name="gap9")