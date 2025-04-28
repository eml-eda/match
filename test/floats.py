from typing import Tuple
from match.relay.utils.utils import numpy_to_array
from match.utils.utils import get_random_np_array
import tvm
from tvm import relay


def create_fp_conv_ex(inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                       padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                       groups:int=1,out_ch:int=1,inp_ch:int=3,**kwargs):
    x = relay.var("input_0", relay.TensorType((1,inp_ch)+inp_shape))
    # Get or generate weight_values
    weights = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_ch,inp_ch)+fil_shape),dtype="float32")
    # Get or generate bias values
    bias = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_ch,)),dtype="float32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "conv_weights"
    bias_name = "conv_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, bias_name: bias}

    # define operations
    x = relay.op.nn.conv2d(x, w,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           kernel_size=fil_shape,
                           )
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=1)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_globalavgpool_ex(inp_shape:Tuple=(1,32,32),**kwargs):
    x = relay.var("input_0", relay.TensorType((1,)+inp_shape))

    # define weights and bias values in params
    params = {}

    # define operations
    x = relay.op.nn.global_avg_pool2d(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_add_ex(inp_shape:Tuple=(1,32,32),**kwargs):
    np.random.seed(0)
    x = relay.var("input_0", relay.TensorType((1,)+inp_shape))
    y = relay.var("input_1", relay.TensorType((1,)+inp_shape))
    # define params
    params = {}
    # define operations
    x = relay.op.add(x,y)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_dense_ex(inp_features:int=256,out_features:int=128,**kwargs):
    x = relay.var("input_0", relay.TensorType((1,inp_features)))
    # Get or generate weight_values
    weights = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_features,inp_features)),dtype="float32")
    # Get or generate bias values
    bias = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_features,)),dtype="float32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "dense_weights"
    bias_name = "dense_bias"

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights.shape, weights.dtype))

    # define weights and bias values in params
    params = {weights_name: weights, bias_name: bias}

    # define operations
    x = relay.op.nn.dense(x, w)
    b = relay.var(bias_name, relay.TensorType(bias.shape, bias.dtype))
    x = relay.op.nn.bias_add(x, b, axis=1)
    x = relay.op.nn.relu(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params