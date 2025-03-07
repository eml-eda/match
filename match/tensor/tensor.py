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
    
    def c_offset_expr(self, node_name):
        dims_expr = []
        for idx,dim in enumerate(self.dims):
            global_idx = f"{node_name}_{dim.name}->global_idx"
            sizes_ = [str(inner_dim.size) for inner_dim in self.dims[idx+1:] if inner_dim.size > 1]
            if dim.size > 1:
                if sizes_:
                    dims_expr.append(f"{global_idx} * {' * '.join(sizes_)}")
                else:
                    dims_expr.append(f"{global_idx}")
        if len(dims_expr) == 0:
            return "0"
        return " + ".join(dims_expr)
    
    def c_offset_expr_sw_mem(self, mem, node_name):
        dims_expr = []
        for idx,dim in enumerate(self.dims):
            global_idx = f"({node_name}_{dim.name}->global_idx - {node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{idx}].start_idx)"
            sizes_ = [f"{node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{inner_idx+idx}].size" for inner_idx,inner_dim in enumerate(self.dims[idx+1:]) if inner_dim.size > 1]
            if dim.size > 1:
                if sizes_:
                    if self.bits!=8:
                        sizes_.append(f"{self.bits/8}")
                    dims_expr.append(f"{global_idx} * {' * '.join(sizes_)}")
                else:
                    dims_expr.append(f"{global_idx}")
        if len(dims_expr) == 0:
            return "0"
        return "(int)("+" + ".join(dims_expr)+")"
    
    def c_offset_expr_size_sw_mem(self, mem, node_name):
        sizes_ = [f"{node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{inner_idx}].size" for inner_idx,inner_dim in enumerate(self.dims) if inner_dim.size > 1]
        if len(sizes_) == 0:
            return "0"
        if self.bits!=8:
            sizes_.append(f"{self.bits/8}")
        return " * ".join(sizes_)

    @property
    def prod_shape(self):
        return str(np.prod([dim.size for dim in self.dims]))
    
class MatchTensorTile:
    def __init__(self,tensor: MatchTensor=MatchTensor(),tiled_dims: List[MatchTiledDim]=[]) -> None:
        self.tiled_dims = tiled_dims
        self.tensor = tensor