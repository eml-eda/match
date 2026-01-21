import math
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from match.utils.utils import c_friendly_npvalue, numpy_dtype_to_c_type


class TensorType(Enum):
    CONST = "const"
    INPUT = "input"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"
    INOUT = "inout"


@dataclass
class Tensor:
    id: int
    type: TensorType
    shape: Tuple[int]
    size: int = None
    name: str = None
    tiling_dim : int = None
    
    def __post_init__(self):
        if self.size is None:
            self.size = self.size if self.size is not None else int(math.prod(self.shape))
            
    @property
    def chunks(self) -> int:
        return self.shape[self.tiling_dim] if self.tiling_dim is not None else 1
    

@dataclass
class RuntimeTensor(Tensor):
    name : str = "tensor"
    
    is_intermediate: bool = False
    is_constant: bool = False
    is_output: bool = False
    is_input: bool = False
    
    constant_val: np.ndarray = np.array([1])
    original_constant_val: np.ndarray = np.array([1])
    dtype: np.dtype = np.dtype("float16")
    
    node_id : int = 0
    node_info: dict = None
    
    soc_mem_offsets : list[int] = None
    ext_mem_offsets : list[int] = None
    
    static_in_soc_mem : bool = False
    static_in_ext_mem : bool = False
    
    used_by_tvm : bool = False
    
    @property
    def num_bytes(self) -> int:
        return self.size * self.dtype.itemsize
    
    @property
    def c_type(self) -> str:
        return numpy_dtype_to_c_type(self.dtype)
    
    @property
    def c_value(self) -> str:
        return "{}" if not self.is_constant else c_friendly_npvalue(self.constant_val)
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.type == TensorType.CONST:
            self.is_constant = True
        elif self.type == TensorType.INPUT:
            self.is_input = True
        elif self.type == TensorType.OUTPUT:
            self.is_output = True
        elif self.type == TensorType.INTERMEDIATE:
            self.is_intermediate = True
        elif self.type == TensorType.INOUT:
            self.is_input = True
            self.is_output = True
    