from typing import Tuple
from match.relay.utils.utils import create_random_array
import tvm
from tvm import relay


def create_conv_ex(inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                       padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                       groups:int=1,out_ch:int=1,inp_ch:int=3,
                       requant_pattern:bool=False,
                       right_shift:int=1,nhwc:bool=True,**kwargs):
    tens_inp_shape = (1,inp_ch)+inp_shape if not nhwc else (1,)+inp_shape+(inp_ch,)
    x = relay.var("input_0", relay.TensorType(tens_inp_shape, "uint8"))
    # Get or generate weight_values
    tens_weights_shape = (out_ch,inp_ch)+fil_shape if not nhwc else fil_shape+(inp_ch,out_ch)
    weights = create_random_array(tens_weights_shape,"int8")
    # Get or generate bias values
    scale = create_random_array((out_ch,), "int32", min_val=-2500, max_val=2500)
    bias = create_random_array((out_ch,), "int32", min_val=-2500, max_val=2500)
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "conv_weights"
    scale_name = "conv_scale"
    bias_name = "conv_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, scale_name: scale, bias_name: bias}

    # define operations
    x = relay.op.nn.conv2d(
        x, w,
        strides=strides,
        padding=padding,
        groups=groups,
        kernel_size=fil_shape,
        data_layout="NHWC" if nhwc else "NCHW",
        kernel_layout="HWIO" if nhwc else "OIHW",
        out_dtype="int32",
    )
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    s = relay.var(scale_name, relay.TensorType(scale.shape, scale.dtype))
    # x = relay.op.cast(x, "int32")
    # x = relay.frontend.common.set_span(x, "FAKE_CAST_TVM_OUT_DTYPE")
    x = relay.op.multiply(x, s)
    x = relay.op.add(x, b)
    if requant_pattern:
        x = relay.op.right_shift(x, relay.const(right_shift))
        x = relay.op.clip(x, a_min=0, a_max=255)
        x = relay.op.cast(x, "uint8")
    else:
        x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_dense_ex(inp_features:int=256,out_features:int=128,
                        activation:bool=True,
                        requant_pattern:bool=False,
                        right_shift:int=1,**kwargs):
    """Generate a small network in TVM Relay IR that performs a requantized convolution
    """
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType((1,inp_features), "uint8"))
    # Get or generate weight_values
    weights = create_random_array((out_features,inp_features),"int8")
    # Get or generate bias values
    bias = create_random_array((out_features,), "int32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "dense_weights"
    bias_name = "dense_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, bias_name: bias}

    # define operations
    x = relay.op.nn.dense(x, w, out_dtype=bias.dtype)
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=-1)
    if activation:
        if requant_pattern:
            x = relay.op.right_shift(x, relay.const(right_shift))
            x = relay.op.clip(x, a_min=0, a_max=255)
            x = relay.op.cast(x, "uint8")
        else:
            x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_simple_sum_ex(inp_shape:Tuple=(32,32),**kwargs):
    x = relay.var("input_0", relay.TensorType((1,3)+inp_shape, "uint8"))
    y = relay.var("input_1", relay.TensorType((1,3)+inp_shape, "uint8"))
    # Get or generate scale values
    scale = create_random_array((1, 3, 1, 1), "int32", min_val=1, max_val=1)
    scale_name = "simple_sum_scale"

    # define relay input vars
    sc_ = relay.var(scale_name, relay.TensorType(scale.shape, scale.dtype))

    # define weights and bias values in params
    params = {scale_name: scale, }

    # define operations
    x = relay.cast(x, "int32")
    y = relay.cast(y, "int32")
    x = relay.op.add(x,y)
    x = relay.op.multiply(x, sc_)
    x = relay.op.right_shift(x, relay.const(0))
    x = relay.op.clip(x, a_min=0, a_max=255)
    x = relay.op.cast(x, "uint8")
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_sumnet_ex(inp_shape:Tuple=(32,32),**kwargs):
    x = relay.var("input_0", relay.TensorType((1,3)+inp_shape, "uint8"))
    y = relay.var("input_1", relay.TensorType((1,3)+inp_shape, "uint8"))
    z = relay.var("input_2", relay.TensorType((1,3)+inp_shape, "uint8"))
    w = relay.var("input_3", relay.TensorType((1,3)+inp_shape, "uint8"))
    # Get or generate scale values
    scale_1 = create_random_array((1, 3, 1, 1), "int32", min_val=1, max_val=1)
    scale_1_name = "sumnet_scale_1"

    scale_2 = create_random_array((1, 3, 1, 1), "int32", min_val=1, max_val=1)
    scale_2_name = "sumnet_scale_2"

    scale_3 = create_random_array((1, 3, 1, 1), "int32", min_val=1, max_val=1)
    scale_3_name = "sumnet_scale_3"
    # define relay input vars
    sc_1 = relay.var(scale_1_name, relay.TensorType(scale_1.shape, scale_1.dtype))
    sc_2 = relay.var(scale_2_name, relay.TensorType(scale_2.shape, scale_2.dtype))
    sc_3 = relay.var(scale_3_name, relay.TensorType(scale_3.shape, scale_3.dtype))
    # define weights and bias values in params
    params = {scale_1_name: scale_1, scale_2_name: scale_2, scale_3_name: scale_3}

    # define operations
    x = relay.cast(x, "int32")
    y = relay.cast(y, "int32")
    x = relay.op.add(x,y)
    x = relay.op.multiply(x, sc_1)
    x = relay.op.right_shift(x, relay.const(0))
    x = relay.op.clip(x, a_min=0, a_max=255)
    x = relay.op.cast(x, "uint8")

    z = relay.cast(z, "int32")
    w = relay.cast(w, "int32")
    z = relay.op.add(z,w)
    z = relay.op.multiply(z, sc_2)
    z = relay.op.right_shift(z, relay.const(0))
    z = relay.op.clip(z, a_min=0, a_max=255)
    z = relay.op.cast(z, "uint8")

    x = relay.cast(x, "int32")
    z = relay.cast(z, "int32")
    x = relay.op.add(x,z)
    x = relay.op.multiply(x, sc_3)
    x = relay.op.right_shift(x, relay.const(0))
    x = relay.op.clip(x, a_min=0, a_max=255)
    x = relay.op.cast(x, "uint8")
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params