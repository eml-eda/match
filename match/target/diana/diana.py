from match.target.target import MatchTarget
from match.target.diana.digital import DigitalAccelerator

class Diana(MatchTarget):
    def __init__(self,**kwargs):
        super(Diana,self).__init__([
            DigitalAccelerator(**kwargs),
        ],name="diana",**kwargs)
