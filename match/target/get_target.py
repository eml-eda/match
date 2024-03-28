from match.target.gap9 import Gap9
from match.target.target import DefaultMatchTarget, MatchTarget

TARGETS={
    "gap9":Gap9
}
target=None

def reset_target():
    global target
    target=None

def set_target(target_class: MatchTarget=DefaultMatchTarget):
    global target
    if target_class==None:
        target_class=DefaultMatchTarget
    else:
        assert issubclass(target_class,MatchTarget)
        target=target_class

def get_target(target_name:str=None):
    if (target_name==None or target_name=="") and target!=None:
        return target()
    elif (target_name==None or target_name=="") or (target_name not in TARGETS):
        set_target()
        return target()
    else:
        assert issubclass(TARGETS[target_name],MatchTarget)
        set_target(TARGETS[target_name])
        return TARGETS[target_name]()