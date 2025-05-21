from math import prod
from typing import Tuple
import numpy as np
import numpy.typing as npt

from match.utils.utils import c_friendly_npvalue, numpy_dtype_to_c_type

class MatchMemoryTensor:
    def __init__(
            self, name: str="p1", is_intermediate: bool=False,
            is_constant: bool=False, is_output: bool=False,
            is_input: bool=False,
            constant_val: npt.ArrayLike=np.array([1]),
            original_constant_val: npt.ArrayLike=np.array(1),
            shape: Tuple[int]=(1,),
            dtype: npt.DTypeLike=np.dtype("uint8"),
            node_id: int=0,
            node_info={}
        ):
        self.name=name
        self.is_intermediate=is_intermediate
        self.is_constant=is_constant
        self.is_output=is_output
        self.is_input=is_input
        # Note: This has been removed since a tensor can be a combination of the single types, e.g. an input and an output at the same time
        # if sum([self.is_intermediate,self.is_constant,self.is_output,self.is_input])!=1:
            # raise Exception(f"Match Memory Tensor can only be one option between(intermediate,constant,output,input)")
        self.constant_val=constant_val
        self.original_constant_val=original_constant_val
        self.shape=shape
        self.dtype=dtype
        self.node_id=node_id
        self.last_usage=node_id
        self.mem_offset = -1
        self.stored_in_external_memory = False
        self.move_temp_to_ext_mem = list()
        self.load_from_ext_mem_at = list()
        self.c_type = numpy_dtype_to_c_type(self.dtype)
        self.c_value = "{}" if not self.is_constant else c_friendly_npvalue(self.constant_val)
        self.prod_shape = prod(self.shape) 
        self.node_info = node_info
        self.start_usage = -1 if (int(self.is_intermediate)+int(self.is_output))==0 else self.node_id
        self.used_at = list()
        self.mem_offset_at = dict()
        self.used_by_tvm = False

    @property
    def lifetime(self):
        return (self.start_usage,self.last_usage)
    
    @property
    def lifetime_span(self):
        return self.last_usage-self.start_usage

    @property
    def elems(self):
        return prod(self.shape)

    @property
    def num_bytes(self):
        return self.prod_shape * self.dtype.itemsize
    
    def update_last_usage(self,new_ending_idx):
        if self.start_usage==-1:
            self.start_usage=new_ending_idx
        self.used_at.append(new_ending_idx)
        self.last_usage=new_ending_idx