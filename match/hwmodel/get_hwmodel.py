from match.hwmodel.gap9cluster import Gap9ClusterHwModel
from match.hwmodel.hwmodel import HwModel

DEVICES_MODELS={
    "gap9cluster":Gap9ClusterHwModel
}

def get_model(device_name:str=""):
    if device_name not in DEVICES_MODELS:
        return HwModel
    else:
        assert issubclass(DEVICES_MODELS[device_name],HwModel)
        return DEVICES_MODELS[device_name]()