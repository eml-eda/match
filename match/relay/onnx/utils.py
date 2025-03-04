from typing import Dict
import onnx
import copy
from math import prod
from match.utils.utils import c_friendly_npvalue, get_random_np_array, numpy_dtype_to_c_type
from match.model import DynamicDim

def get_inputs_outputs(onnx_model: onnx.ModelProto,dynamic_dims:Dict[str,DynamicDim]={}):
    match_inputs = {inp.name:{
        "name":inp.name,
        "c_type":numpy_dtype_to_c_type(onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)),
        "dims":[dim.dim_value if dim.dim_param=="" else dim.dim_param for dim in inp.type.tensor_type.shape.dim],
        "dynamic":all([dim.dim_param=="" for dim in inp.type.tensor_type.shape.dim]),
        "prod_shape":int(prod([int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim])) if all([dim.dim_param=="" for dim in inp.type.tensor_type.shape.dim]) else 1000,
        "shape":[dim.dim_value for dim in inp.type.tensor_type.shape.dim],
        "c_arr_size":int(prod([int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim])) if all([dim.dim_param=="" for dim in inp.type.tensor_type.shape.dim]) else 1000,
        "c_arr_values":c_friendly_npvalue(get_random_np_array(dtype=onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type),shape=tuple([int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim]))) if all([dim.dim_param=="" for dim in inp.type.tensor_type.shape.dim]) else "{"+str([1 for _ in range(1000)])[1:-1]+"}",
        } for inp in onnx_model.graph.input}
    match_outputs = {f"output{idx if len(onnx_model.graph.output)>1 else ''}":{
        "name":f"output{idx if len(onnx_model.graph.output)>1 else ''}",
        "c_type":numpy_dtype_to_c_type(onnx.helper.tensor_dtype_to_np_dtype(out.type.tensor_type.elem_type)),
        "dims":[dim.dim_value if dim.dim_param=="" else dim.dim_param for dim in out.type.tensor_type.shape.dim],
        "dynamic":all([dim.dim_param=="" for dim in out.type.tensor_type.shape.dim]),
        "prod_shape":int(prod([int(dim.dim_value) for dim in out.type.tensor_type.shape.dim])) if all([dim.dim_param=="" for dim in out.type.tensor_type.shape.dim]) else 1000,
        "shape":[dim.dim_value for dim in out.type.tensor_type.shape.dim],
        "c_arr_size":int(prod([int(dim.dim_value) for dim in out.type.tensor_type.shape.dim])) if all([dim.dim_param=="" for dim in out.type.tensor_type.shape.dim]) else 1000,
        } for idx,out in enumerate(onnx_model.graph.output)}
    return match_inputs,match_outputs

def get_onnx_static_model(onnx_model: onnx.ModelProto=None,static_params:Dict={}):
    static_onnx_model = copy.deepcopy(onnx_model)
    #breakpoint()
    # static definement of inputs
    for i_idx in range(len(onnx_model.graph.input)):
        for d_idx in range(len(onnx_model.graph.input[i_idx].type.tensor_type.shape.dim)):
            dim_name = onnx_model.graph.input[i_idx].type.tensor_type.shape.dim[d_idx].dim_param
            dynamic_dims = dim_name.split(" ")[::2]
            if any([dyn_dim in static_params for dyn_dim in dynamic_dims]):
                dim_value = 0
                for dyn_dim in dynamic_dims:
                    if dyn_dim.isdigit():
                        dim_value+=int(dyn_dim)
                    elif dyn_dim in static_params:
                        dim_value+=static_params[dyn_dim]
                static_onnx_model.graph.input[i_idx].type.tensor_type.shape.dim[d_idx].dim_value = dim_value
    # static definement of outputs
    for o_idx in range(len(onnx_model.graph.output)):
        for d_idx in range(len(onnx_model.graph.output[o_idx].type.tensor_type.shape.dim)):
            dim_name = onnx_model.graph.output[o_idx].type.tensor_type.shape.dim[d_idx].dim_param
            dynamic_dims = dim_name.split(" ")[::2]
            if any([dyn_dim in static_params for dyn_dim in dynamic_dims]):
                dim_value = 0
                for dyn_dim in dynamic_dims:
                    if dyn_dim.isdigit():
                        dim_value+=int(dyn_dim)
                    elif dyn_dim in static_params:
                        dim_value+=static_params[dyn_dim]
                static_onnx_model.graph.output[o_idx].type.tensor_type.shape.dim[d_idx].dim_value = dim_value

    return static_onnx_model