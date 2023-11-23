import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from driver.driver import driver




if __name__=="__main__":
    #model_url="https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/b385b1b242dc89a35dd808235b885ed8a19aedc1/resnet18_1.0.onnx"
    #model_path = download_testdata(model_url, "resnet18_1.onnx", module="onnx")
    #print(model_path)
    # now you have super_resolution.onnx on disk
    #onnx_model = onnx.load(model_path)
    #onnx_file="https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    #onnx_model=onnx.load(onnx_file)
    #print(onnx_model)
    #mod, params=relay.frontend.from_onnx(onnx_model)
    from relay_conv2d import create_model
    mod, params = create_model()
    print(mod)
    driver(mod,params)