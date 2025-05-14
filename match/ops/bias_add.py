from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAddExpr, MatchAssignExpr, MatchPrimitiveExpr, MatchTensorExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpBiasAdd(MatchOp):
    def __init__(self, out_arr = ..., var_arr = ..., const_arr = ..., axis: int=-1, bias: int=0, **kwargs):
        super().__init__(out_arr, var_arr, const_arr, op="BiasAdd", **kwargs)
        self.axis = axis
        self.bias = bias
        self.op_code = 2

    def basic_schedules(self):
        output, activations, biases = self.outs[0], self.vars[0], self.consts[0]
        loops = []
        for dim in output.dims:
            loops.append(MatchLoop(dim=dim, size=dim.size, name=dim.name, init_instrs=[], instrs=[]))
        bias_val = MatchPrimitiveExpr(name="bias_val",dtype=output.dtype, const=False, init_expr=MatchTensorExpr(tensor=biases))
        axis = -1
        if self.axis >= 0:
            axis = self.axis
        loops[axis].init_instrs.append(MatchInstr(lhs_expr=bias_val))
        loops[-1].instrs.append(MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),
                                           rhs_expr=MatchInstr(lhs_expr=MatchTensorExpr(tensor=activations),eq_expr=MatchAddExpr(),
                                                                rhs_expr=bias_val)))
        basic_bias_add_schedule = MatchSchedule(
            blocks=[
                MatchBlock(
                    loops=loops,
                    init_instrs=[],
                    instrs=[],
                )
            ],
            init_instrs=[],
            instrs=[],
        )
        return [basic_bias_add_schedule]