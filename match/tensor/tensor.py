from typing import List
from match.dim.dim import MatchDim, MatchTiledDim
import numpy as np
from numpy import typing as npt

SUPPORTED_TENSOR_LAYOUTS = (
    "NHWC", "NCHW", "HWIO", "OHWI", "OIHW",
    "HWC", "CHW", "WHC", "WCH",
    "NC", "CN", "HW", "WH", "WC", "CW", 
    "H", "W", "C", "N"
)

SUPPORTED_DIVIDED_TENSOR_LAYOUTS = (
    "NCHWc16"
)
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
        self.stored_in_ext_mem = False

    @property
    def unsupported_layout(self):
        return self.layout not in SUPPORTED_TENSOR_LAYOUTS and self.layout not in SUPPORTED_DIVIDED_TENSOR_LAYOUTS

    def __eq__(self, other):
        return other is not None and (self.name == other.name and self.dtype == other.dtype and self.dims == other.dims)
    
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
    
    def get_subtile(self,):
        dims_with_subtiles = dict()
        dims_subtiles = dict()
        for idx,(dim, layout_key) in enumerate(zip(self.dims, self.layout)):
            if layout_key.isupper() and layout_key.lower() in self.layout:
                size_subtile = ""
                idx_subtile = self.layout.index(layout_key.lower())+1
                for layout_val in self.layout[idx_subtile:]:
                    if layout_val.isdigit():
                        size_subtile += layout_val
                    else:
                        break
                size_subtile = int(size_subtile)
                dims_with_subtiles[(dim.name, idx)] = size_subtile
                dims_subtiles[(dim.name, idx_subtile-1)] = size_subtile
        return dims_with_subtiles, dims_subtiles

    def c_offset_expr_sw_mem(self, mem, schedule, block_idx, loop_idx, node_name):
        dims_expr = []
        if self.layout in SUPPORTED_DIVIDED_TENSOR_LAYOUTS:
            lps_dims = [lp.dim for lp in schedule.blocks[block_idx].loops[:loop_idx]]
            dims_with_subtiles, dims_subtiles = self.get_subtile()
            for idx, dim in enumerate(self.dims):
                dim_size = dim.size
                if (dim.name, idx) in dims_with_subtiles:
                    dim_size = dim.size//dims_with_subtiles[(dim.name, idx)]
                elif (dim.name, idx) in dims_subtiles:
                    dim_size = dims_subtiles[(dim.name, idx)]
                if dim_size>1 and ((dim in lps_dims) or (dim.dim_dependency is not None and any([dim_ in dim.dim_dependency.dependencies for dim_ in lps_dims]))):
                    global_idx_str = f"{node_name}_{dim.name}->global_idx"
                    start_idx_str = f"{node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{idx}].start_idx"
                    global_idx = f"({global_idx_str} - {start_idx_str})"
                    if dim.dim_dependency is not None:
                        global_idx = f"({global_idx_str}>0?{global_idx_str}-({start_idx_str}>0?{start_idx_str}:0):0)"
                    if (dim.name, idx) in dims_with_subtiles:
                        global_idx = f"({global_idx} / {dims_with_subtiles[(dim.name, idx)]})"
                    elif (dim.name, idx) in dims_subtiles:
                        global_idx = f"({global_idx} % {dims_subtiles[(dim.name, idx)]})"
                    sizes_ = list()
                    for inner_idx, inner_dim in enumerate(self.dims[idx+1:]):
                        if inner_dim.size > 1:
                            size_ = f"{node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{inner_idx+idx+1}].size"
                            if (inner_dim.name, inner_idx+idx+1) in dims_with_subtiles:
                                size_ = f"({size_} / {dims_with_subtiles[(inner_dim.name, inner_idx+idx+1)]})"
                            elif (inner_dim.name, inner_idx+1) in dims_subtiles:
                                size_ = f"({size_} % {dims_subtiles[(inner_dim.name, inner_idx+1)]})"
                            sizes_.append(size_)
                        
                    if len(sizes_)>0:
                        dims_expr.append(f"{global_idx} * {' * '.join(sizes_)}")
                    else:
                        dims_expr.append(f"{global_idx}")
        else:
            lps_dims = [lp.dim for lp in schedule.blocks[block_idx].loops[:loop_idx]]
            for idx,dim in enumerate(self.dims):
                if dim.size>1 and ((dim in lps_dims) or (dim.dim_dependency is not None and any([dim_ in dim.dim_dependency.dependencies for dim_ in lps_dims]))):
                    global_idx_str = f"{node_name}_{dim.name}->global_idx"
                    start_idx_str = f"{node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{idx}].start_idx"
                    global_idx = f"({global_idx_str} - {start_idx_str})"
                    if dim.dim_dependency is not None:
                        global_idx = f"({global_idx_str}>0?{global_idx_str}-({start_idx_str}>0?{start_idx_str}:0):0)"
                    sizes_ = [f"{node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{inner_idx+idx+1}].size" for inner_idx,inner_dim in enumerate(self.dims[idx+1:]) if inner_dim.size > 1]
                    if len(sizes_)>0:
                        dims_expr.append(f"{global_idx} * {' * '.join(sizes_)}")
                    else:
                        dims_expr.append(f"{global_idx}")
        if len(dims_expr) == 0:
            return "0"
        bytes_expr = ""
        if self.bits!=8:
            bytes_expr = f" * {int(self.bits/8) if self.bits%8==0 else self.bits/8}"
        return "(int)("+" + ".join(dims_expr)+")"+bytes_expr
    
    def c_offset_expr_size_sw_mem(self, mem, node_name):
        if self.layout in SUPPORTED_DIVIDED_TENSOR_LAYOUTS:
            dims_with_subtiles, dims_subtiles = self.get_subtile()
            sizes_ = list()
            for idx, dim in enumerate(self.dims):
                if dim.size > 1:
                    size_ = f"{node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{idx}].size"
                    if (dim.name, idx) in dims_with_subtiles:
                        size_ = f"({size_} / {dims_with_subtiles[(dim.name, idx)]})"
                    elif (dim.name, idx) in dims_subtiles:
                        size_ = f"({size_} % {dims_subtiles[(dim.name, idx)]})"
                    sizes_.append(size_)
        else:
            sizes_ = [f"{node_name}_{self.name}_tiles_[{mem}*{self.num_dims}+{inner_idx}].size" for inner_idx,inner_dim in enumerate(self.dims) if inner_dim.size > 1]
        if len(sizes_) == 0:
            return "0"
        if self.bits!=8:
            sizes_.append(f"{self.bits/8}")
        return " * ".join(sizes_)

    @property
    def prod_shape(self):
        return str(np.prod([dim.size for dim in self.dims]))
    
    @property
    def prod_shape_int(self):
        return int(np.prod([dim.size for dim in self.dims]))
    
class MatchTensorTile:
    def __init__(self,tensor: MatchTensor=MatchTensor(),tiled_dims: List[MatchTiledDim]=[]) -> None:
        self.tiled_dims = tiled_dims
        self.tensor = tensor