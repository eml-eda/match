import numpy as np
from numpy import typing as npt

from match.ops.op import MatchOp
from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAssignExpr, MatchMulExpr, MatchPlusEqExpr, MatchPrimitiveExpr, MatchTensorExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule


class MatchOpBatchMatMul(MatchOp):
    """
    Batch Matrix Multiplication Operation
    Performs batch matrix multiplication on input tensors.
    Given two input tensors of shapes (B, M, N) and (B, N, K), the output tensor will have shape (B, M, K).
    Where:
    - B: Batch size
    - M: Number of rows in the first matrix
    - N: Number of columns in the first matrix / rows in the second matrix
    - K: Number of columns in the second matrix
    """

    def __init__(
        self,
        out_arr=...,
        var_arr=...,
        const_arr=...,
        dim_b: int = 1,
        dim_m: int = 1,
        dim_n: int = 1,
        dim_k: int = 1,
        out_dtype: npt.DTypeLike = np.dtype("int8"),
        **kwargs,
    ):
        super().__init__(out_arr, var_arr, const_arr, op="BatchMatMul", **kwargs)
        self.dim_b = dim_b
        self.dim_m = dim_m
        self.dim_n = dim_n
        self.dim_k = dim_k
        self.out_dtype = out_dtype
        self.op_code = 14

    def basic_schedules(self):
        output = self.outs[0]
        x1 = self.vars[0]
        x2 = self.vars[1] if len(self.vars) > 1 else self.consts[0]

        dim_b = output.dims[0]  # batch_size
        dim_m = output.dims[1]
        dim_n = x1.dims[1]
        dim_k = x2.dims[1]
        loops = []
        
        zero_prim = MatchPrimitiveExpr(name="zero", dtype=self.out_dtype, const=True, val=0)
        dense_sum = MatchPrimitiveExpr(
            name="dense_sum",
            dtype=self.out_dtype,
            const=False,
            init_expr=zero_prim,
        )
        
        loops.append(
            MatchLoop(
                dim=dim_b, 
                size=dim_b.size, 
                name="batch", 
                instrs=[], 
                init_instrs=[]
            )
        )
        
        loops.append(
            MatchLoop(
                dim=dim_m,
                size=dim_m.size,
                name="out_row",
                instrs=[],
                init_instrs=[],
            )
        )
        
        loops.append(
            MatchLoop(
                dim=dim_k,
                size=dim_k.size,
                name="out_ch",
                init_instrs=[MatchInstr(lhs_expr=dense_sum, eq_expr=MatchAssignExpr(), rhs_expr=zero_prim)],
                instrs=[MatchInstr(lhs_expr=MatchTensorExpr(tensor=output), eq_expr=MatchAssignExpr(), rhs_expr=dense_sum)],
            )
        )
        
        mul_instr = MatchInstr(
            lhs_expr=MatchTensorExpr(tensor=x1),
            eq_expr=MatchMulExpr(),
            rhs_expr=MatchTensorExpr(tensor=x2),
        )
        loops.append(
            MatchLoop(
                dim=dim_n,
                size=dim_n.size,
                name="inp_ch",
                instrs=[
                    MatchInstr(
                        lhs_expr=dense_sum,
                        eq_expr=MatchPlusEqExpr(),
                        rhs_expr=mul_instr,
                    )
                ],
                init_instrs=[],
            )
        )
        
        basic_batch_matmul_schedule = MatchSchedule(
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
        
        return [basic_batch_matmul_schedule]