from typing import Dict, List
import onnx
from match.model import DynamicDim, get_cutoff_combinations
import tvm.relay as relay
from match.relay.onnx.onnx_utils import get_onnx_static_model, sanitize_onnx_plinio,sanitize_onnx_only_remove
from match.model import MatchModel

def onnx_to_relay(onnx_filename,dynamic_dims:Dict[str,DynamicDim]={"sequence_length":DynamicDim(name="sequence_length",dim_min=1,dim_max=16),
                                                               "batch_size":DynamicDim(name="batch_size",dim_min=1,dim_max=16),
                                                               "past_sequence_length":DynamicDim(name="past_sequence_length",dim_min=1,dim_max=16)}):
    onnx_model = onnx.load(onnx_filename)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        sanitize_onnx_plinio(onnx_model=onnx_model)
        #sanitize_onnx_only_remove(onnx_model=onnx_model)
    
    models_to_compile = []
    if len(dynamic_dims)>0:
        cutoffs=get_cutoff_combinations(list(dynamic_dims.values()))
        for cutoff_list in cutoffs:
            dyn_is_max = all([cut_val==dynamic_dims[cut_name].max for cut_name,cut_val in cutoff_list])
            static_mod = get_onnx_static_model(onnx_model=onnx_model,static_params={cut_name:cut_val for cut_name,cut_val in cutoff_list})
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

    return models_to_compile