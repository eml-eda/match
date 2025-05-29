import numpy as np
from numpy import typing as npt
from match.ops.op import MatchOp
from typing import List,Tuple

from match.schedule.block import MatchBlock
from match.schedule.expr import MatchAssignExpr, MatchContinueExpr, MatchDimIdxExpr, MatchGteExpr, MatchIfExpr, MatchLtExpr, MatchMulExpr, MatchOrExpr, MatchPlusEqExpr, MatchPrimitiveExpr, MatchSubExpr, MatchTensorExpr
from match.schedule.instr import MatchInstr
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule

class MatchOpConv3D(MatchOp):
    def __init__(self, out_arr: List=[], var_arr: List=[], const_arr: List=[],
                 padding: Tuple[int]=(0,0,0,0,0,0), strides: Tuple[int]=(1,1,1), dilation: Tuple[int]=(1,1,1),
                 groups: int=1, kernel_size: Tuple[int]=(1,1),
                 data_layout: str="NCHWZ", kernel_layout: str="OIHWZ", out_dtype: npt.DTypeLike=np.dtype("int8"),
                 depthwise: bool=False, **kwargs) -> None:
        super().__init__(out_arr, var_arr, const_arr, op="Conv3D", **kwargs)
        self.padding = padding
        self.strides = strides
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.data_layout = data_layout
        self.kernel_layout = kernel_layout
        self.out_dtype = out_dtype
        self.op_code = 11
    
    def basic_schedules(self) -> List[MatchSchedule]:
        output, activations, weights = self.outs[0], self.vars[0], self.consts[0]
        batch_size_dim = output.dims[0]
        out_chs_dim = output.dims[1] if self.data_layout == "NCDHW" else output.dims[4]
        out_d_dim = output.dims[2] if self.data_layout == "NCDHW" else output.dims[1]
        out_h_dim = output.dims[3] if self.data_layout == "NCDHW" else output.dims[2]
        out_w_dim = output.dims[4] if self.data_layout == "NCDHW" else output.dims[3]
        inp_chs_dim = activations.dims[1] if self.data_layout == "NCDHW" else activations.dims[4]
        inp_depth_dim = activations.dims[2] if self.data_layout == "NCDHW" else activations.dims[1]
        inp_height_dim = activations.dims[3] if self.data_layout == "NCDHW" else activations.dims[2]
        inp_width_dim = activations.dims[4] if self.data_layout == "NCDHW" else activations.dims[3]
        ker_d_dim = weights.dims[2] if self.kernel_layout == "OIDHW" else weights.dims[1]
        ker_h_dim = weights.dims[3] if self.kernel_layout == "OIDHW" else weights.dims[2]
        ker_w_dim = weights.dims[4] if self.kernel_layout == "OIDHW" else weights.dims[3]
        conv3d_sum = MatchPrimitiveExpr(
            name="conv3d_sum",dtype=self.out_dtype, const=False,
            init_expr=MatchPrimitiveExpr(name="zero",dtype=self.out_dtype,const=True,val=0)
        )
        # instrs
        init_conv3d_sum_instr = MatchInstr(lhs_expr=conv3d_sum,must_be_after={"out_ch"})
        save_output_instr = MatchInstr(lhs_expr=MatchTensorExpr(tensor=output),eq_expr=MatchAssignExpr(),rhs_expr=conv3d_sum)
        mac_instr = MatchInstr(lhs_expr=conv3d_sum,eq_expr=MatchPlusEqExpr(),rhs_expr=MatchInstr(
            lhs_expr=MatchTensorExpr(tensor=activations),eq_expr=MatchMulExpr(),rhs_expr=MatchTensorExpr(tensor=weights))
        )
        padding_instr = MatchInstr(lhs_expr=MatchIfExpr(
            MatchInstr(
                # if less than 0 it means that its padding
                lhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_depth_val",dim=inp_depth_dim),eq_expr=MatchLtExpr(),rhs_expr=MatchPrimitiveExpr(name="zero",dtype="int32",const=True,val=0)),
                eq_expr=MatchOrExpr(),
                rhs_expr=MatchInstr(
                    lhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_depth_val",dim=inp_depth_dim),eq_expr=MatchGteExpr(),rhs_expr=MatchPrimitiveExpr(name="inp_depth_val",dtype="int32",const=True,val=inp_depth_dim.size)),
                    eq_expr=MatchOrExpr(),
                    # if less than 0 it means that its padding
                    rhs_expr=MatchInstr(
                        lhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_height_val",dim=inp_height_dim),eq_expr=MatchLtExpr(),rhs_expr=MatchPrimitiveExpr(name="zero",dtype="int32",const=True,val=0)),
                        eq_expr=MatchOrExpr(),
                        # if greater or equal than the input height it means that its padding
                        rhs_expr=MatchInstr(lhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_height_val",dim=inp_height_dim),eq_expr=MatchGteExpr(),rhs_expr=MatchPrimitiveExpr(name="inp_height_val",dtype="int32",const=True,val=inp_height_dim.size)),
                            eq_expr=MatchOrExpr(),
                            # if less than 0 it means that its padding
                            rhs_expr=MatchInstr(
                                lhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_width_val",dim=inp_width_dim),eq_expr=MatchLtExpr(),rhs_expr=MatchPrimitiveExpr(name="zero",dtype="int32",const=True,val=0)),
                                eq_expr=MatchOrExpr(),
                                # if greater or equal than the input width it means that its padding
                                rhs_expr=MatchInstr(lhs_expr=MatchDimIdxExpr(name="inp_width_val",dim=inp_width_dim),eq_expr=MatchGteExpr(),rhs_expr=MatchPrimitiveExpr(name="inp_width_val",dtype="int32",const=True,val=inp_width_dim.size)),
                            )
                        )
                    )
                )
            ),
            then_expr=MatchContinueExpr(),
        ))
        # loops
        loops = []
        loops.append(MatchLoop(dim=batch_size_dim,size=batch_size_dim.size,name="batch",instrs=[]))
        loops.append(MatchLoop(dim=out_chs_dim,size=out_chs_dim.size,name="out_ch",instrs=[]))
        loops.append(MatchLoop(dim=out_d_dim,size=out_d_dim.size,name="out_d",instrs=[]))
        loops.append(MatchLoop(dim=out_h_dim,size=out_h_dim.size,name="out_h",instrs=[]))
        loops.append(MatchLoop(dim=out_w_dim,size=out_w_dim.size,name="out_w",init_instrs=[
                        init_conv3d_sum_instr],instrs=[save_output_instr]))
        loops.append(MatchLoop(dim=ker_d_dim,size=ker_d_dim.size,name="ker_d",instrs=[]))
        loops.append(MatchLoop(dim=ker_h_dim,size=ker_h_dim.size,name="ker_h",instrs=[]))
        loops.append(MatchLoop(dim=ker_w_dim,size=ker_w_dim.size,name="ker_w",
            init_instrs=[
                # if padding continue
            padding_instr] if sum(self.padding[::2])!=0 else [],instrs=[]))
        loops.append(MatchLoop(dim=inp_chs_dim,size=inp_chs_dim.size,name="inp_ch", instrs=[
                        # do actual convolution
                        mac_instr],init_instrs=[]))
        basic_conv3d = MatchSchedule(
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
        return [basic_conv3d]
