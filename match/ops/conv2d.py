import numpy as np
from numpy import typing as npt
from match.ops.op import MatchOp
from typing import List,Tuple

from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAssignExpr, MatchContinueExpr, MatchDimIdxExpr, MatchGteExpr, MatchIfExpr, MatchLtExpr, MatchMulExpr, MatchOrExpr, MatchPlusEqExpr, MatchPrimitiveExpr, MatchSubExpr, MatchTensorExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpConv2D(MatchOp):
    def __init__(self, out_arr: List=[], var_arr: List=[], const_arr: List=[],
                 padding: Tuple[int]=(0,0,0,0), strides: Tuple[int]=(1,1), dilation: Tuple[int]=(1,1),
                 groups: int=1, kernel_size: Tuple[int]=(1,1),
                 data_layout: str="NCHW", kernel_layout: str="OIHW", out_dtype: npt.DTypeLike=np.dtype("int8"),
                 depthwise: bool=False, **kwargs) -> None:
        super().__init__(out_arr, var_arr, const_arr, op="Conv2D", **kwargs)
        self.padding = padding
        self.strides = strides
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.data_layout = data_layout
        self.kernel_layout = kernel_layout
        self.out_dtype = out_dtype
        self.op_code = 3
    
    def basic_schedules(self) -> List[MatchSchedule]:
        output, activations, weights = self.outs[0], self.vars[0], self.consts[0]
        batch_size_dim = output.dims[0]
        out_chs_dim = output.dims[1] if self.data_layout == "NCHW" else output.dims[3]
        out_h_dim = output.dims[2] if self.data_layout == "NCHW" else output.dims[1]
        out_w_dim = output.dims[3] if self.data_layout == "NCHW" else output.dims[2]
        inp_chs_dim = activations.dims[1] if self.data_layout == "NCHW" else activations.dims[3]
        inp_height_dim = activations.dims[2] if self.data_layout == "NCHW" else activations.dims[1]
        inp_width_dim = activations.dims[3] if self.data_layout == "NCHW" else activations.dims[2]
        ker_h_dim = weights.dims[2] if self.kernel_layout == "OIHW" else weights.dims[1]
        ker_w_dim = weights.dims[3] if self.kernel_layout == "OIHW" else weights.dims[2]
        conv2d_sum = MatchPrimitiveExpr(name="conv2d_sum",dtype=self.out_dtype, const=False,
                                    init_expr=MatchPrimitiveExpr(name="zero",dtype=self.out_dtype,const=True,val=0))
        loops = []
        if batch_size_dim.size > 1:
            loops.append(MatchLoop(dim=batch_size_dim,size=batch_size_dim.size,name="batch",instrs=[]))
        if out_chs_dim.size > 1:
            loops.append(MatchLoop(dim=out_chs_dim,size=out_chs_dim.size,name="out_ch",instrs=[]))
        if out_h_dim.size > 1:
            loops.append(MatchLoop(dim=out_h_dim,size=out_h_dim.size,name="out_h",instrs=[]))
        loops.append(MatchLoop(dim=out_w_dim,size=out_w_dim.size,name="out_w",init_instrs=[
                        MatchInstr(lhs_expr=conv2d_sum)
                    ],instrs=[
                        MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),rhs_expr=conv2d_sum)
                    ]))
        if ker_h_dim.size > 1:
            loops.append(MatchLoop(dim=ker_h_dim,size=ker_h_dim.size,name="ker_h",instrs=[]))
        loops.append(MatchLoop(dim=ker_w_dim,size=ker_w_dim.size,name="ker_w",
            init_instrs=[
                # if padding continue
                MatchInstr(lhs_expr=MatchIfExpr(
                    MatchInstr(
                        # if less than 0 it means that its padding
                        lhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_height_val",dim=inp_height_dim),eq_expr=MatchLtExpr(),rhs_expr=MatchPrimitiveExpr(name="zero",dtype="int32",const=True,val=0)),
                        eq_expr=MatchOrExpr(),
                        # if greater or equal than the input height it means that its padding
                        rhs_expr=MatchInstr(
                            lhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_height_val",dim=inp_height_dim),eq_expr=MatchGteExpr(),rhs_expr=MatchPrimitiveExpr(name="inp_height_val",dtype="int32",const=True,val=inp_height_dim.size)),
                            eq_expr=MatchOrExpr(),
                            # if less than 0 it means that its padding
                            rhs_expr=MatchInstr(
                                lhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_width_val",dim=inp_width_dim),eq_expr=MatchLtExpr(),rhs_expr=MatchPrimitiveExpr(name="zero",dtype="int32",const=True,val=0)),
                                eq_expr=MatchOrExpr(),
                                # if greater or equal than the input width it means that its padding
                                rhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_width_val",dim=inp_width_dim),eq_expr=MatchGteExpr(),rhs_expr=MatchPrimitiveExpr(name="inp_width_val",dtype="int32",const=True,val=inp_width_dim.size)),
                            )
                        )   
                    ),
                    then_expr=MatchContinueExpr(),
                ))
        ],instrs=[]))
        loops.append(MatchLoop(dim=inp_chs_dim,size=inp_chs_dim.size,name="inp_ch", instrs=[
                        # do actual convolution
                        MatchInstr(lhs_expr=conv2d_sum,eq_expr=MatchPlusEqExpr(),rhs_expr=MatchInstr(
                            lhs_expr=MatchTensorExpr(tensor=activations),eq_expr=MatchMulExpr(),rhs_expr=MatchTensorExpr(tensor=weights))
                        )
                    ],init_instrs=[]))
        basic_conv2d = MatchSchedule(
            blocks = [
                MatchBlock(
                    loops=loops,
                    init_instrs=[],
                    instrs=[],
                )
            ],
            init_instrs=[],
            instrs=[],
        )
        return [basic_conv2d]
