

from match.schedule.expr import MatchEmptyExpr, MatchExpr


class MatchInstr(MatchExpr):
    def __init__(self, lhs_expr: MatchExpr=MatchEmptyExpr(), eq_expr: MatchExpr=MatchEmptyExpr(),
                 rhs_expr: MatchExpr=MatchEmptyExpr(), **kwargs) -> None:
        self.lhs_expr = lhs_expr
        self.eq_expr = eq_expr
        self.rhs_expr = rhs_expr
        super().__init__("instr")
    
    @property
    def c_expr(self):
        return f"{self.lhs_expr.c_expr} {self.eq_expr.c_expr} {self.rhs_expr.c_expr}"