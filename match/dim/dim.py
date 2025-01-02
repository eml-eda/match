class DimDependency:
    def __init__(self,dependencies) -> None:
        self.dependencies = dependencies
    
    def __eq__(self, other):
        return self.dependencies == other.dependencies

class MatchDim:
    def __init__(self, name: str="width", size: int=1, is_dynamic: bool=False, dim_dependency: DimDependency=None) -> None:
        self.name = name
        self.original_name = name
        self.is_dynamic = is_dynamic
        self.size = size
        self.dim_dependency = dim_dependency
    
    @property
    def is_independent(self):
        return self.dim_dependency is None
    
    def __hash__(self):
        return hash((self.original_name,self.size))
    
    def __eq__(self, other):
        return self.name == other.name and self.size == other.size and self.is_dynamic == other.is_dynamic and self.dim_dependency == other.dim_dependency
    
class MatchTiledDim:
    def __init__(self,dim: MatchDim=MatchDim(),size: int=1) -> None:
        self.dim = dim
        self.name = dim.name
        self.size = size