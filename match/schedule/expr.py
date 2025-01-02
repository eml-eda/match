

class MatchExpr:
    def __init__(self, expr):
        self.expr = expr
    
    @property
    def c_expr(self):
        return ""