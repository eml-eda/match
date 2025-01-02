from typing import List
from match.dim.dim import MatchDim, MatchTiledDim
import numpy as np
from numpy import typing as npt

class MatchTensor:
    def __init__(self, name: str="tensor_A", dims: List[MatchDim]=[], dtype: npt.DTypeLike=np.dtype("int8"),
                 tensor_type:str="var", data: npt.NDArray=np.array([])) -> None:
        self.name = name
        self.name_up = name.upper()
        self.dims = dims
        self.num_dims = len(dims)
        self.dtype = dtype
        self.bits = dtype.itemsize * 8
        self.tensor_type = tensor_type
        self.data = data

    def __eq__(self, other):
        return self.name == other.name and self.dtype == other.dtype and self.dims == other.dims
    
class MatchTensorTile:
    def __init__(self,tensor: MatchTensor=MatchTensor(),tiled_dims: List[MatchTiledDim]=[]) -> None:
        self.tiled_dims = tiled_dims
        self.tensor = tensor