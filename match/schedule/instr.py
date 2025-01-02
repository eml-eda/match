

class MatchInstr:
    def __init__(self, lhs_expr, eq_expr, rhs_expr) -> None:
        self.lhs_expr = lhs_expr
        self.eq_expr = eq_expr
        self.rhs_expr = rhs_expr
    
    @property
    def c_expr(self):
        return f"{self.lhs_expr.c_expr} ${self.eq_expr.c_expr} {self.eq_expr.c_expr} {self.rhs_expr.c_expr}"