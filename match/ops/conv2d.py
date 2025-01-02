from match.ops.op import MatchOp
from typing import List,Tuple

class MatchOpConv2D(MatchOp):
    def __init__(self, out_arr: List=[], var_arr: List=[], const_arr: List=[],
                 padding: Tuple[int]=(0,0,0,0), strides: Tuple[int]=(1,1), dilation: Tuple[int]=(1,1),
                 groups: int=1, kernel_size: Tuple[int]=(1,1),
                 depthwise: bool=False, **kwargs) -> None:
        super().__init__(out_arr, var_arr, const_arr, op="Conv2D", **kwargs)
        self.padding = padding
        self.strides = strides
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.op_code = 3