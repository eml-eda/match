from match.target.gap9 import Gap9
from match.target.target import DefaultMatchTarget, MatchTarget

TARGETS={
    "gap9":Gap9(),
}
target_inst=None

def reset_target():
    global target_inst
    target_inst=None

def set_target(target: MatchTarget=None):
    global target_inst
    if target==None:
        target_inst=DefaultMatchTarget()
    else:
        assert issubclass(target.__class__,MatchTarget)
        target_inst=target

def get_target(target_name:str=None):
    global target_inst
    if (target_name==None or target_name=="") and target_inst!=None:
        return target_inst
    elif (target_name==None or target_name=="") or (target_name not in TARGETS):
        set_target()
        return target_inst
    else:
        assert issubclass(TARGETS[target_name].__class__,MatchTarget)
        set_target(TARGETS[target_name])
        return TARGETS[target_name]