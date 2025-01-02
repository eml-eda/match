


from typing import Dict

from match.dim.dim import MatchDim


class MatchNode:

    def __init__(self, ops: Dict = {}, calls: Dict = {}, dims: Dict = {}) -> None:
        self.ops = dict()
        self.ops_occurrences = dict()
        self.calls = dict()
        self.var_tensors = dict()
        self.const_tensors = dict()
        self.output_tensors = dict()
        self.intermediate_tensors = dict()
        self.dims = dict()

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