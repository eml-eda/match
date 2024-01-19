from match.relay.onnx.onnx import onnx_to_relay
from tvm import relay

def relay_from_file(relay_filename,params_filename):
    mod_text=""
    params_bytes=bytes("","utf8")
    with open(relay_filename,"r") as mod_file:
        mod_text=mod_file.read()
    with open(params_filename,"rb") as params_file:
        params_bytes=params_file.read()
    mod=relay.fromtext(mod_text)
    params=relay.load_param_dict(params_bytes)
    return mod,params

def get_relay_from(input_type,filename,params_filename):
    if input_type=="onnx":
        return onnx_to_relay(filename)
    elif input_type=="relay":
        return relay_from_file(filename,params_filename)
    else:
        raise Exception("Input type not supported!")