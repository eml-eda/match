
from typing import Dict, List
from match.dim.dim import MatchTiledDim
from match.schedule.block import MatchBlock
from match.schedule.buffer import MatchMemBuffer
from match.schedule.expr import MatchDimIdxExpr, MatchExpr, MatchTensorExpr
from match.schedule.instr import MatchInstr
from match.tensor.tensor import MatchTensor, MatchTensorTile
import pprint

EXPRS_WITH_NODE_NAME = (MatchTensorExpr, MatchDimIdxExpr)

class MatchSchedule:
    def __init__(self,blocks: List[MatchBlock]=[],
                 tensors: Dict[str,MatchTensor]={},
                 tensor_tiles: Dict[str,List[MatchTensorTile]]={},
                 init_instrs: List[MatchInstr]=[],
                 instrs: List[MatchInstr]=[],
                 buffers: List[MatchMemBuffer]=[]) -> None:
        self.blocks = blocks
        self.tensors = tensors
        self.init_instrs = init_instrs
        self.instrs = instrs
        self.tensor_tiles = tensor_tiles
        self.buffers = buffers
        self.num_units = 1
        self.exec_module = None

    def set_default_tensor_tiles(self):
        for tensor in self.tensors.values():
            self.tensor_tiles[tensor.name] = [MatchTensorTile(tensor=tensor,
                                            tiled_dims=[MatchTiledDim(dim=dim,size=dim.size) for dim in tensor.dims])]

    def update_exprs_with_node_name(self, node_name: str="main_0"):
        def deep_check_expr(expr, node_name_):
            if isinstance(expr, EXPRS_WITH_NODE_NAME):
                expr.node_name = node_name_
            for expr_val in expr.__dict__.values():
                if isinstance(expr_val, EXPRS_WITH_NODE_NAME):
                    expr_val.node_name = node_name_
                if isinstance(expr_val, MatchExpr):
                    deep_check_expr(expr_val, node_name_)

        for instr in self.instrs:
            deep_check_expr(instr, node_name)
        for init_instr in self.init_instrs:
            deep_check_expr(init_instr, node_name)
        
        for block in self.blocks:
            for instr in block.instrs:
                deep_check_expr(instr, node_name)
            for init_instr in block.init_instrs:
                deep_check_expr(init_instr, node_name)
            for lp in block.loops:
                for instr in lp.instrs:
                    deep_check_expr(instr, node_name)
                for init_instr in lp.init_instrs:
                    deep_check_expr(init_instr, node_name) 

    def __str__(self):
        return pprint(self)