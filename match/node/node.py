


from typing import Dict, List
from match.ops.op import MatchOp
from tvm.relay import TypeCall
from match.dim.dim import MatchDim
from match.tensor.tensor import MatchTensor


class MatchNode:

    def __init__(self, ops: Dict = {}, calls: Dict = {}, dims: Dict = {}) -> None:
        self.ops: Dict[str, MatchOp] = dict()
        self.ops_occurrences: Dict[str, List[str]] = dict()
        self.calls: Dict[str, TypeCall] = dict()
        self.var_tensors: Dict[str, MatchTensor] = dict()
        self.const_tensors: Dict[str, MatchTensor] = dict()
        self.output_tensors: Dict[str, MatchTensor] = dict()
        self.intermediate_tensors: Dict[str, MatchTensor] = dict()
        self.dims: Dict[str, MatchDim] = dict()

    @property
    def var_names(self):
        return [n for n in self.var_tensors.keys()]
    
    @property
    def const_names(self):
        return [n for n in self.const_tensors.keys()]
    
    @property
    def output_names(self):
        return [n for n in self.output_tensors.keys()]
    
    @property
    def intermediate_names(self):
        return [n for n in self.intermediate_tensors.keys()]

    @property
    def tensors(self):
        return {**self.var_tensors,**self.const_tensors,**self.output_tensors,**self.intermediate_tensors}
    
    @property
    def tensors_arr(self):
        return [t for t in self.tensors.values()]

    @property
    def dim_arr(self):
        return [v for v in self.dims.values()]

    @property
    def independent_dims(self):
        return [dim for dim in self.dims.values() if dim.is_independent]
    
    @property
    def dependent_dims(self):
        return [dim for dim in self.dims.values() if not dim.is_independent]
    
    @property
    def default_dim(self):
        return MatchDim("default", 1,)

    def __eq__(self,other):
        return self.tensors == other.tensors and self.dims == other.dims and self.ops_occurrences == other.ops_occurrences