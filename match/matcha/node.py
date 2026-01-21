from dataclasses import dataclass, field
from typing import Optional

from match.schedule.schedule import MatchSchedule
from match.node.node import MatchNode

@dataclass
class Node:
    id: int = 0
    
    inp_tids: list = field(default_factory=list)
    out_tids: list = field(default_factory=list)
    children_nids: list = field(default_factory=list)
    device_id: int = 0
    duration: int = 1
    chunks : int = 1
    is_concat : bool = False
    
    @property
    def chunk_duration(self) -> int:
        return int(self.duration / self.chunks) if self.chunks > 0 else self.duration
    

@dataclass
class SuperNode(Node):
    pattern_id : int = 0
    match_id : int = 0
    sub_nids : list[int] = None
    
    
@dataclass
class RuntimeNode(Node):
    inputs: list = None
    outputs: list = None
    
    fn_name : str = "node_fn"
    name : str = "node"
    
    node_info : dict = None
    node_id : int = 0
    node_name : str = "node"
    
    mapping : Optional[MatchSchedule] = None
    match_node : Optional[MatchNode] = None
    
    num_parents : int = 0
    
    tensor_soc_segments_ids : dict = None
    
    @property
    def fallback(self) -> bool:
        return "match" not in self.fn_name