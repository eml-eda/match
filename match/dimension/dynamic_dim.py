

from typing import List


class DynamicDim:
    def __init__(self,name:str="",dim_min:int=1,dim_max:int=1,preferred_cutoffs:List[int]=[]) -> None:
        self.min = dim_min
        self.max = dim_max
        self.cutoffs = set([self.min,self.max])
        if len(preferred_cutoffs)>0:
            self.cutoffs = set(list(self.preferred_cutoffs)+preferred_cutoffs)
        self.name = name


def get_combinations(dynamic_dims:List[DynamicDim]=[]):
    combination = [[(dim.name,cutoffs) for cutoff in dim.cutoffs] for dim in dynamic_dims]
    return list(itertools.product(*combination))