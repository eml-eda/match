from typing import List
from match.dim.dim import MatchDim, MatchTiledDim
import numpy as np
from numpy import typing as npt

class MatchTensor:
    def __init__(self, name: str="tensor_A", dims: List[MatchDim]=[], dtype: npt.DTypeLike=np.dtype("int8"),
                 tensor_type: str="var", data: npt.NDArray=np.array([]), layout: str="") -> None:
        self.name = name
        self.name_up = name.upper()
        self.dims = dims
        self.num_dims = len(dims)
        self.dtype = dtype
        self.bits = dtype.itemsize * 8
        self.tensor_type = tensor_type
        self.data = data
        self.original_data = data
        self.layout = layout
        self.is_fused = False
        self.unsupported_layout = False
        self.stored_in_ext_mem = False

    def __eq__(self, other):
        return self.name == other.name and self.dtype == other.dtype and self.dims == other.dims
    
    @property
    def c_offset_expr(self):
        dims_expr = []
        for idx,dim in enumerate(self.dims):
            global_idx = f"{dim.name}->global_idx"
            sizes_ = [str(inner_dim.size) for inner_dim in self.dims[idx+1:] if inner_dim.size > 1]
            if dim.size > 1:
                if sizes_:
                    dims_expr.append(f"{global_idx} * {' * '.join(sizes_)}")
                else:
                    dims_expr.append(f"{global_idx}")
        if len(dims_expr) == 0:
            return "0"
        return " + ".join(dims_expr)
    
    def c_offset_expr_sw_mem(self,mem):
        dims_expr = []
        for idx,dim in enumerate(self.dims):
            global_idx = f"({dim.name}->global_idx - {self.name}_tiles_[{mem}][{idx}].start_idx)"
            sizes_ = [f"{self.name}_tiles_[{mem}*{self.num_dims}+{inner_idx+idx}].size" for inner_idx,inner_dim in enumerate(self.dims[idx+1:]) if inner_dim.size > 1]
            if dim.size > 1:
                if sizes_:
                    dims_expr.append(f"{global_idx} * {' * '.join(sizes_)}")
                else:
                    dims_expr.append(f"{global_idx}")
        if len(dims_expr) == 0:
            return "0"
        return " + ".join(dims_expr)
    
    def c_offset_expr_size_sw_mem(self,mem):
        sizes_ = [f"{self.name}_tiles_[{mem}*{self.num_dims}+{inner_idx}].size" for inner_idx,inner_dim in enumerate(self.dims) if inner_dim.size > 1]
        if len(sizes_) == 0:
            return "0"
        return " * ".join(sizes_)

    @property
    def prod_shape(self):
        return str(np.prod([dim.size for dim in self.dims]))
    
class MatchTensorTile:
    def __init__(self,tensor: MatchTensor=MatchTensor(),tiled_dims: List[MatchTiledDim]=[]) -> None:
        self.tiled_dims = tiled_dims
        self.tensor = tensor