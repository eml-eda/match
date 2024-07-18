from match.target.target import MatchTarget
from match.target.diana.digital import DigitalAccelerator

class Diana(MatchTarget):
    def __init__(self):
        super(Diana,self).__init__([
            DigitalAccelerator(),
        ],name="diana")
