
from typing import Dict, List
from match.dim.dim import MatchTiledDim
from match.schedule.block import MatchBlock
from match.schedule.instr import MatchInstr
from match.tensor.tensor import MatchTensor, MatchTensorTile
import pprint

class MatchSchedule:
    def __init__(self,blocks: List[MatchBlock]=[],
                 tensors: Dict[str,MatchTensor]={},
                 tensor_tiles: Dict[str,List[MatchTensorTile]]={},
                 init_instrs: List[MatchInstr]=[],
                 instrs: List[MatchInstr]=[],) -> None:
        self.blocks = blocks
        self.tensors = tensors
        self.init_instrs = init_instrs
        self.instrs = instrs
        self.tensor_tiles = tensor_tiles
        if len(self.tensor_tiles)==0:
            self.set_default_tensor_tiles()

    def set_default_tensor_tiles(self):
        for tensor in self.tensors.values():
            self.tensor_tiles[tensor.name] = [MatchTensorTile(tensor=tensor,
                                            tiled_dims=[MatchTiledDim(dim=dim,size=dim.size) for dim in tensor.dims])]
    
    def __str__(self):
        return pprint(self)