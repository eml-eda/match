

from typing import Set
from match.schedule.expr import MatchEmptyExpr, MatchExpr


class MatchInstr(MatchExpr):
    def __init__(self, lhs_expr: MatchExpr=MatchEmptyExpr(), eq_expr: MatchExpr=MatchEmptyExpr(),
                 rhs_expr: MatchExpr=MatchEmptyExpr(),
                #  must_be_after: Set[str]={},
                 **kwargs) -> None:
        self.lhs_expr = lhs_expr
        self.eq_expr = eq_expr
        self.rhs_expr = rhs_expr
        # self.must_be_after = must_be_after
        super().__init__("instr")
    
    @property
    def c_expr(self):
        return f"{self.lhs_expr.c_expr} {self.eq_expr.c_expr} {self.rhs_expr.c_expr}"