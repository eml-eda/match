
from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAssignExpr, MatchMulExpr, MatchPlusEqExpr, MatchPrimitiveExpr, MatchTensorExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpDense(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., inp_features:int=1, out_features: int=1, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="Dense", **kwargs)
        self.inp_features = inp_features
        self.out_features = out_features
        self.op_code = 1

    def basic_schedules(self):
        output, activations, weights = self.outs[0], self.vars[0], self.consts[0]
        batch_size_dim = output.dims[0]
        out_chs_dim = output.dims[1]
        inp_chs_dim = activations.dims[1]
        loops = []
        dense_sum = MatchPrimitiveExpr(name="conv2d_sum",dtype=self.out_dtype, const=False,
                                    init_expr=MatchPrimitiveExpr(name="zero",dtype=self.out_dtype,const=True,val=0))
        loops.append(MatchLoop(dim=out_chs_dim, size=out_chs_dim.size, name="out_ch",init_instrs=[
            MatchInstr(lhs_expr=dense_sum)
        ], instrs=[
            MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),rhs_expr=dense_sum)
        ]))
        loops.append(MatchLoop(dim=inp_chs_dim, size=inp_chs_dim.size, name="inp_ch",instrs=[
            MatchInstr(lhs_expr=MatchTensorExpr(name="output",tensor=output),eq_expr=MatchPlusEqExpr(),
                       rhs_expr=MatchInstr(lhs_expr=MatchTensorExpr(name="activations",tensor=activations),eq_expr=MatchMulExpr(),
                                           rhs_expr=MatchTensorExpr(name="weights",tensor=weights)))
        ]))
        basic_dense_schedule = MatchSchedule(
            blocks=[
                MatchBlock(
                    loops=loops
                )
            ]
        )
        return basic_dense_schedule