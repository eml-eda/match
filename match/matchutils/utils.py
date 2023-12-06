import match.hwmodel as hwmodel

def get_hw_model(device_name:str=""):
    return hwmodel.get_model(device_name=device_name)

def mock_func(*args):
    return None