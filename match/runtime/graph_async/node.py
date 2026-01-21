from typing import Optional, List

from match.node.node import MatchNode
from match.schedule.schedule import MatchSchedule

from .tensor import MatchMemoryTensor


class MatchGraphRuntimeNodeCall:
    """
    This class represent a call to a node (function) in the MATCH graph runtime.
    """
    def __init__(
        self,
        inputs: Optional[List[MatchMemoryTensor]] = None,
        outputs: Optional[List[MatchMemoryTensor]] = None,
        fn_name: str = "default_lib_1",
        name: str = "default_lib_1",
        node_info: dict = {},
        node_id: int = 0,
        node_name: Optional[str] = None,
        schedule: Optional[MatchSchedule] = None,
        match_node: Optional[MatchNode] = None,
        inp_tensor_ids: Optional[List[int]] = None,
        out_tensor_ids: Optional[List[int]] = None,
        num_parents: int = 0,
        children: List[int] = None
    ):
        self.inputs = inputs
        self.outputs = outputs
        
        self.fn_name = fn_name
        self.name = name
        
        self.node_info = node_info
        self.fallback = "match" not in self.fn_name
        
        self.node_id = node_id
        self.node_name = node_name
        
        self.schedule = schedule
        
        self.match_node = match_node
        self.num_parents = num_parents
        
        self.inp_tensor_ids = inp_tensor_ids or []
        self.out_tensor_ids = out_tensor_ids or []
        
        self.children = children or []
        self.device_id = 0 if schedule is None else (schedule.exec_module.id + 1)
        