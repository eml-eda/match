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
    
    @property
    def c_offset_expr(self):
        dims_expr = []
        for idx,dim in enumerate(self.dims):
            global_idx = f"{dim.name}->global_idx"
            sizes_ = [str(inner_dim.size) for inner_dim in self.dims[idx:] if inner_dim.size > 1]
            if dim.size > 1:
                if sizes_:
                    dims_expr.append(f"{global_idx} * {' * '.join(sizes_)}")
                else:
                    dims_expr.append(f"{global_idx}")
        if len(dims_expr) == 0:
            return "0"
        return " + ".join(dims_expr)
    
class MatchTensorTile:
    def __init__(self,tensor: MatchTensor=MatchTensor(),tiled_dims: List[MatchTiledDim]=[]) -> None:
        self.tiled_dims = tiled_dims
        self.tensor = tensor