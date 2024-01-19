import onnx
import tvm.relay as relay
from match.relay.onnx.onnx_utils import sanitize_onnx

def onnx_to_relay(onnx_filename):
    onnx_model = onnx.load(onnx_filename)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        sanitize_onnx(onnx_model)
    return relay.frontend.from_onnx(onnx_model)