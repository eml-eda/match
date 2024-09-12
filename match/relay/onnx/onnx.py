from typing import Dict, List
import onnx
from match.dimension import DynamicDim
import tvm.relay as relay
from match.relay.onnx.onnx_utils import sanitize_onnx_plinio,sanitize_onnx_only_remove,obtain_onnx_static_models

def onnx_to_relay(onnx_filename,dynamic_dims_cutoffs:Dict[str,DynamicDim]={}):
    onnx_model = onnx.load(onnx_filename)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        sanitize_onnx_plinio(onnx_model=onnx_model)
        #sanitize_onnx_only_remove(onnx_model=onnx_model)
    cutoffs get_combinations
    return relay.frontend.from_onnx(onnx_model)