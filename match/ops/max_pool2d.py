import numpy as np
from numpy import typing as npt
from typing import List, Tuple

from match.ops.op import MatchOp


class MatchOpMaxPool2D(MatchOp):
    def __init__(self, out_arr: List = [], var_arr: List = [], const_arr: List = [],
                 padding: Tuple[int] = (0, 0, 0, 0), strides: Tuple[int] = (1, 1),
                 dilation: Tuple[int] = (1, 1), pool_size: Tuple[int] = (1, 1),
                 data_layout: str = "NCHW", out_layout: str = "NCHW",
                 out_dtype: npt.DTypeLike = np.dtype("float16"), **kwargs) -> None:
        super().__init__(out_arr, var_arr, const_arr, op="MaxPool2D", **kwargs)
        self.padding = padding
        self.strides = strides
        self.dilation = dilation
        self.pool_size = pool_size
        self.data_layout = data_layout
        self.out_layout = out_layout
        self.out_dtype = out_dtype
        self.op_code = 21