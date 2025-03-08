
from typing import List
import numpy as np
from match.utils.utils import get_random_np_array
from tvm.ir import IRModule


def get_default_inputs(mod: IRModule=None, input_files: List[str]=[], min_input_val=None, max_input_val=None):
    default_inputs = []
    if input_files is not None and len(input_files)==len(mod["main"].params):
        default_inputs = [ np.loadtxt(input_files[param_idx], delimiter=',',
                            dtype=np.dtype(param.type_annotation.dtype),
                            usecols=[0]).reshape([int(i) for i in param.type_annotation.shape])
                            for param_idx, param in enumerate(mod["main"].params)]
    else:
        default_inputs = [get_random_np_array(dtype=param.type_annotation.dtype, shape=param.type_annotation.shape, min_val=min_input_val, max_val=max_input_val)
                           for param in mod["main"].params]
    return default_inputs