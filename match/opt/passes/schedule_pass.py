from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from match.schedule.schedule import MatchSchedule



class MatchScheduleOptPassContext:
    """Context object that holds state and configuration for compilation passes."""
    
    def __init__(
        self,configs={}
    ):
        self.configs = configs
    
    def __enter__(self):
        return self

    def __exit__(self, ptype, value, trace):
        return

class MatchScheduleOptPass:
    """Base class for all compilation passes."""
    
    def __init__(self):
        pass

    def __call__(self, schedule) -> MatchSchedule:
        return schedule

class MatchScheduleOptSequentialPasses(MatchScheduleOptPass):
    """A pass that works on a sequence of pass objects. Multiple passes can be
    executed sequentially using this class.

    Note that users can also provide a series of passes that they don't want to
    apply when running a sequential pass. Pass dependency will be resolved in
    the backend as well.

    Parameters
    ----------
    passes : Optional[List[Pass]]
        A sequence of passes candidate for optimization.

    opt_level : Optional[int]
        The optimization level of this sequential pass.
        The opt_level of a default sequential pass is set to 0.
        Note that some of the passes within the Sequantial may still not be executed
        if their opt_level is higher than the provided opt_level.

    name : Optional[str]
        The name of the sequential pass.

    required : Optional[List[str]]
        The list of passes that the sequential pass is dependent on.
    """

    def __init__(self, passes: List[MatchScheduleOptPass]=[]):
        self.passes = passes if passes else []
        if not isinstance(passes, (list, tuple)):
            raise TypeError("passes must be a list of Pass objects.")
        
    def __call__(self, schedule):
        new_schedule = schedule
        for pass_call in self.passes:
            new_schedule = pass_call(new_schedule) 
        return new_schedule
        
        
    