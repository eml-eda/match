from match.target.gap9 import Gap9
from match.target.target import DefaultMatchTarget, MatchTarget

TARGETS={
    "gap9":Gap9
}
target=None

def set_target(target_class: MatchTarget):
    global target
    target=target_class

def get_target(target_name:str=""):
    if target!=None:
        return target()
    elif target_name not in TARGETS:
        set_target(DefaultMatchTarget)
        return DefaultMatchTarget()
    else:
        assert issubclass(TARGETS[target_name],MatchTarget)
        set_target(TARGETS[target_name])
        return TARGETS[target_name]()