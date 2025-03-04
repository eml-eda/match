from typing import Dict
import onnx
from match.model import DynamicDim, get_cutoff_combinations
from match.utils.utils import get_output_path
import tvm.relay as relay
from match.relay.onnx.utils import get_onnx_static_model

def onnx_to_relay(onnx_filename):
    onnx_model = onnx.load(onnx_filename)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print("[ONNX FRONTEND] The model is not correct")
        raise e
    return relay.frontend.from_onnx(onnx_model)

def dyn_onnx_to_relay(onnx_filename,dynamic_dims:Dict[str,DynamicDim]={}, model_name: str="default", dyn_algorithm: str="cuts"):
    onnx_model = onnx.load(onnx_filename)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print("[ONNX FRONTEND] The model is not correct")
        raise e
    
    models = dict()
    if dyn_algorithm=="cuts":
        maxed_relay, maxed_params = None, None
        relay_mod,relay_params = relay.frontend.from_onnx(onnx_model,freeze_params=False)
        cutoffs = get_cutoff_combinations(list(dynamic_dims.values()))
        for idx,cutoff_list in enumerate(cutoffs):
            dyn_is_max = all([cut_val==dynamic_dims[cut_name].max for cut_name,cut_val in cutoff_list])
            static_mod = get_onnx_static_model(onnx_model=onnx_model,static_params={cut_name:cut_val for cut_name,cut_val in cutoff_list})
            onnx.save(static_mod,get_output_path()+"/onnxmodel_"+str(idx)+".onnx")
            relay_mod,relay_params = relay.frontend.from_onnx(static_mod,freeze_params=False)
            if dyn_is_max:
                maxed_relay, maxed_params = relay_mod, relay_params
            else:
                models[model_name+f"_cut_{idx}"] = (relay_mod,relay_params, {cut_name:cut_val for cut_name,cut_val in cutoff_list},)
        return maxed_relay, maxed_params, models
    else:
        raise Exception("[ONNX FRONTED] Currently only the cuts algorithm is supported for dynamic models")