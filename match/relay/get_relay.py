from match.relay.onnx.onnx import dyn_onnx_to_relay, onnx_to_relay
from tvm import relay
import tvm

def relay_from_file(relay_filename,params_filename, odl=False):
    mod_text=""
    params_bytes=bytes("","utf8")
    with open(relay_filename,"r") as mod_file:
        mod_text=mod_file.read()
    if params_filename is not None:
        with open(params_filename,"rb") as params_file:
            params_bytes=params_file.read()
    if not odl:
        mod=relay.fromtext(mod_text)
    else:
        # this comes from compatibilty with the version of the ONNX converter tool
        # FIXME: must be improved
        SEMVER = '#[version = "0.0.5"]\n'
        mod = tvm.parser.parse(
            SEMVER + mod_text,
            "from_string",
            None,
            None,
        )
    params={} if len(params_bytes)==0 else relay.load_param_dict(params_bytes)
    return mod, params

def get_relay_from(input_type, filename, params_filename):
    if input_type=="onnx":
        return onnx_to_relay(filename)
    elif input_type=="relay":
        return relay_from_file(filename,params_filename)
    elif input_type=="relay_odl":
        return relay_from_file(filename,params_filename, odl=True)
    else:
        raise Exception(f"[MATCH FRONTEND] The input file type of the model is not among the supported ones, used {input_type}")
    
def get_dyn_relay_from(input_type, filename, params_filename, model_name, dynamic_dims, dynamic_algorithm):
    if input_type=="onnx":
        return dyn_onnx_to_relay(onnx_filename=filename, model_name=model_name, dynamic_dims=dynamic_dims, dyn_algorithm=dynamic_algorithm)
    else:
        raise Exception(f"[MATCH FRONTEND] Only ONNX files are currently supported for dynamic models, used {input_type}")