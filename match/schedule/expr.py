

from match.dim.dim import MatchDim
from match.tensor.tensor import MatchTensor
from match.utils.utils import numpy_dtype_to_c_type
import numpy as np
from numpy import typing as npt

class MatchExpr:
    def __init__(self, name: str="expr") -> None:
        self.name = name
    
    @property
    def c_expr(self):
        return self.name

class MatchEmptyExpr(MatchExpr):
    def __init__(self):
        super().__init__("")

class MatchEqExpr(MatchExpr):
    def __init__(self):
        super().__init__("==")

class MatchNeqExpr(MatchExpr):
    def __init__(self):
        super().__init__("!=")

class MatchGtExpr(MatchExpr):
    def __init__(self):
        super().__init__(">")

class MatchGteExpr(MatchExpr):
    def __init__(self):
        super().__init__(">=")

class MatchLtExpr(MatchExpr):
    def __init__(self):
        super().__init__("<")

class MatchLteExpr(MatchExpr):
    def __init__(self):
        super().__init__("<=")

class MatchAndExpr(MatchExpr):
    def __init__(self):
        super().__init__("&&")

class MatchOrExpr(MatchExpr):
    def __init__(self):
        super().__init__("||")

class MatchNotExpr(MatchExpr):
    def __init__(self):
        super().__init__("!")

class MatchAssignExpr(MatchExpr):
    def __init__(self):
        super().__init__("=")

class MatchAddExpr(MatchExpr):
    def __init__(self):
        super().__init__("+")

class MatchPlusEqExpr(MatchExpr):
    def __init__(self):
        super().__init__("+=")

class MatchMinusEqExpr(MatchExpr):
    def __init__(self):
        super().__init__("-=")

class MatchSubExpr(MatchExpr):
    def __init__(self):
        super().__init__("-")
    
class MatchMulExpr(MatchExpr):
    def __init__(self):
        super().__init__("*")

class MatchDivExpr(MatchExpr):
    def __init__(self):
        super().__init__("/")

class MatchTernaryExpr(MatchExpr):
    def __init__(self, if_expr: MatchExpr=MatchEmptyExpr(), then_expr: MatchExpr=MatchEmptyExpr(),
                 else_expr: MatchExpr=MatchEmptyExpr()) -> None:
        self.if_expr = if_expr
        self.then_expr = then_expr
        self.else_expr = else_expr
        super().__init__("ternary")

    @property
    def c_expr(self):
        return f"({self.if_expr.c_expr} ? {self.then_expr.c_expr} : {self.else_expr.c_expr}"

class MatchIfExpr(MatchExpr):
    def __init__(self, cond_expr: MatchExpr=MatchEmptyExpr(), then_expr: MatchExpr=MatchEmptyExpr(),
                 else_expr: MatchExpr=MatchEmptyExpr()) -> None:
        self.cond_expr = cond_expr
        self.then_expr = then_expr
        self.else_expr = else_expr
        super().__init__("if")

    @property
    def c_expr(self):
        return f"if ({self.cond_expr.c_expr}) {self.then_expr.c_expr}"

class MatchBreakExpr(MatchExpr):
    def __init__(self, name: str="break") -> None:
        super().__init__(name)
    
class MatchContinueExpr(MatchExpr):
    def __init__(self, name: str="continue") -> None:
        super().__init__(name)

class MatchPrimitiveExpr(MatchExpr):
    def __init__(self, name = "prim", dtype: npt.DTypeLike=np.dtype("int8"), const: bool=False,
                 val= None, init_expr: MatchExpr=None) -> None:
        super().__init__(name)
        self.dtype = dtype
        self.init_expr = init_expr
        self.initialized = False
        self.const = const
        self.val = val

    @property
    def c_expr(self):
        if self.const:
            return str(self.val)
        if not self.initialized:
            self.initialized = True
            return f"{numpy_dtype_to_c_type(self.dtype)} {self.name} = {self.init_expr.c_expr}"
        return self.name

class MatchTensorExpr(MatchExpr):
    def __init__(self, tensor: MatchTensor=MatchTensor()) -> None:
        super().__init__(tensor.name)
        self.tensor = tensor

    @property
    def c_expr(self):
        return f"(({numpy_dtype_to_c_type(self.tensor.dtype)}*){self.tensor.name})[{self.tensor.c_offset_expr}]"

class MatchDimIdxExpr(MatchExpr):
    def __init__(self, name: str="dim_idx", dim: MatchDim=MatchDim()) -> None:
        super().__init__(name)
        self.dim = dim

    @property
    def c_expr(self):
        return f"{self.dim.name}->global_idx"
    
