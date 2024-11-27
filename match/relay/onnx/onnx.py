from typing import Dict, List
import onnx
from match.model import DynamicDim, get_cutoff_combinations
from match.utils.utils import add_save_relay, get_output_path
import tvm
import tvm.relay as relay
from match.relay.onnx.onnx_utils import get_onnx_static_model, sanitize_onnx_plinio,sanitize_onnx_only_remove
from match.model import MatchModel

def numpy_dtype_to_c_type(dtype):
    """Translate NumPy dtype to corresponding C type."""
    dtype_str = str(dtype)
    
    # Mapping NumPy dtype to C type
    mapping = {
        'float32': 'float',
        'float64': 'double',
        'int32': 'int',
        'int64': 'long int',  # or 'long long int'
        'int8': 'char',
        'uint8': 'unsigned char',
        'uint16': 'unsigned short',
        'uint32': 'unsigned int',
        'uint64': 'unsigned long int',  # or 'unsigned long long int'
        'bool': '_Bool'  # or 'bool' in C99
    }
    
    # Check if dtype exists in mapping
    if dtype_str in mapping:
        return mapping[dtype_str]
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def get_dynamic_inputs_outputs(onnx_model: onnx.ModelProto):
    match_inputs = {inp.name:{"name":inp.name,"c_type":numpy_dtype_to_c_type(onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)),"dims":[dim.dim_value if dim.dim_param=="" else dim.dim_param for dim in inp.type.tensor_type.shape.dim]} for inp in onnx_model.graph.input}
    match_outputs = {f"output{idx}":{"name":f"output{idx}","c_type":numpy_dtype_to_c_type(onnx.helper.tensor_dtype_to_np_dtype(out.type.tensor_type.elem_type)),"dims":[dim.dim_value if dim.dim_param=="" else dim.dim_param for dim in out.type.tensor_type.shape.dim]} for idx,out in enumerate(onnx_model.graph.output)}
    return match_inputs,match_outputs

def onnx_to_relay(onnx_filename,dynamic_dims:Dict[str,DynamicDim]={}):
    onnx_model = onnx.load(onnx_filename)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        sanitize_onnx_plinio(onnx_model=onnx_model)
        #sanitize_onnx_only_remove(onnx_model=onnx_model)
    match_inputs, match_outputs = get_dynamic_inputs_outputs(onnx_model)
    models_to_compile = []
    if len(dynamic_dims)>0:
        relay_mod,relay_params = relay.frontend.from_onnx(onnx_model,freeze_params=False)
        add_save_relay(prefix="dynamic_graph",mod=relay_mod)
        cutoffs=get_cutoff_combinations(list(dynamic_dims.values()))
        for idx,cutoff_list in enumerate(cutoffs):
            dyn_is_max = all([cut_val==dynamic_dims[cut_name].max for cut_name,cut_val in cutoff_list])
            static_mod = get_onnx_static_model(onnx_model=onnx_model,static_params={cut_name:cut_val for cut_name,cut_val in cutoff_list})
            onnx.save(static_mod,get_output_path()+"/onnxmodel_"+str(idx)+".onnx")
            relay_mod,relay_params = relay.frontend.from_onnx(static_mod,freeze_params=False)
            models_to_compile.append(
                MatchModel(
                    relay_mod=relay_mod,
                    relay_params=relay_params,
                    dynamic=True,
                    dyn_is_max=dyn_is_max,
                    dynamic_sizes = {cut_name:cut_val for cut_name,cut_val in cutoff_list},
                )
            )
    else:
        relay_mod,relay_params = relay.frontend.from_onnx(onnx_model,freeze_params=False)
        models_to_compile.append(
            MatchModel(
                relay_mod=relay_mod,
                relay_params=relay_params,
                dynamic=False,
            )
        )

    return models_to_compile,match_inputs,match_outputs,dynamic_dims