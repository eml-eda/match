
from typing import Dict, List
from match.dim.dim import MatchTiledDim
from match.schedule.block import MatchBlock
from match.tensor.tensor import MatchTensor, MatchTensorTile

class MatchSchedule:
    def __init__(self,blocks: List[MatchBlock]=[],
                 tensor_tiles: Dict[str,List[MatchTensorTile]]={}) -> None:
        self.blocks = blocks
        self.tensor_tiles = tensor_tiles

    def set_default_tensor_tiles(self, tensors: List[MatchTensor]):
        for tensor in tensors:
            self.tensor_tiles[tensor.name] = [MatchTensorTile(tensor=tensor,
                                            tiled_dims=[MatchTiledDim(dim=dim,size=dim.size) for dim in tensor.dims])]