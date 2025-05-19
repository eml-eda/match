from typing import Dict


class DimDependency:
    def __init__(self, idx_dependencies: Dict={}, size_dependencies: Dict={}) -> None:
        self.idx_dependencies = idx_dependencies
        self.size_dependencies = size_dependencies
    
    @property
    def dependencies(self):
        return {**self.idx_dependencies, **self.size_dependencies}
    
    def __eq__(self, other):
        return other is not None and self.idx_dependencies == other.idx_dependencies and self.size_dependencies == other.size_dependencies

class MatchDim:
    def __init__(self, name: str="width", size: int=1, is_dynamic: bool=False, dim_dependency: DimDependency=None) -> None:
        self.name = name
        self.original_name = name
        self.is_dynamic = is_dynamic
        self.size = size
        self.dim_dependency = dim_dependency
    
    @property
    def start_idx(self):
        if self.dim_dependency is None:
            return 0
        else:
            start_idx = 0
            for ind_dim,mult in self.dim_dependency.idx_dependencies.items():
                start_idx += (mult*(ind_dim if not hasattr(ind_dim,"name") else 0))
            return int(start_idx)

    @property
    def max_size(self):
        if self.dim_dependency is None:
            return self.size
        else:
            max_size = 0
            for ind_dim,mult in self.dim_dependency.size_dependencies.items():
                max_size += (mult*(ind_dim if not hasattr(ind_dim,"name") else ind_dim.size))
            return int(max_size)
        
    @property
    def is_independent(self):
        return self.dim_dependency is None
    
    def __hash__(self):
        return hash((self.original_name,self.size))
    
    def __eq__(self, other):
        return self.name == other.name and self.size == other.size and self.is_dynamic == other.is_dynamic and self.dim_dependency == other.dim_dependency
    
class MatchTiledDim:
    def __init__(self,dim: MatchDim=MatchDim(), size: int=1, max_size: int=1) -> None:
        self.dim = dim
        self.name = dim.name
        self.size = size
        self.max_size = max_size