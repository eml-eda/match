import match.relay.utils.utils as utils
import tvm
import tvm.relay as relay
import numpy as np
from typing import Dict, List, Tuple, Optional
from numpy import typing as npt

def create_model_conv_2d(weight_bits: int = 8,
                 act: bool = True,
                 input_shape: Tuple[int, ...] = (1, 1, 2, 2),
                 weights_shape: Tuple[int, ...] = (3, 1, 3, 3),
                 weights_values: Optional[npt.NDArray] = None,
                 bias_values: Optional[npt.NDArray] = None,
                 padding: Tuple[int, int] = (1, 1),
                 strides: Tuple[int, int] = (1, 1),
                 shift_bits: int = 3,
                 depthwise: bool = False,
                 input_pad: List=None,
                 batchnorm: bool=True,
                 ):
    """Generate a small network in TVM Relay IR that performs a requantized convolution
    """
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType(input_shape, 'uint8'))
    # Get or generate weight_values
    if weights_values is None:
        weights = utils.create_random_array(weights_shape, 
                                            f'int{weight_bits}')
        weights = utils.numpy_to_array(np.ones(weights_shape,f"int{weight_bits}"),f"int{weight_bits}")
    else:
        weights = utils.numpy_to_array(weights_values,weights_values.dtype.name)
    # Get or generate bias values
    if bias_values is None:
        bias = utils.create_random_array(weights_shape[0], 'int32')
        bias = utils.numpy_to_array(np.ones(weights_shape[0],"int32"),"int32")
    else:
        bias = utils.numpy_to_array(bias_values,bias_values.dtype.name)
    # Generate the conv2d call
    if input_pad is not None:
        x= relay.nn.pad(x,input_pad,pad_mode="constant",pad_value=0)
    x, params1 = utils.relay_conv2d_uint8_requant(x, 'conv1', weights, bias, 
                                         padding=padding, 
                                         strides=strides,
                                         act=act, 
                                         shift_bits=shift_bits,
                                         groups=weights_shape[0] if depthwise else 1,
                                         batchnorm=batchnorm)
    #if input_pad is not None:
    #    x= relay.reshape(x,(16))
    params = params1
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_model_add_convs(weight_bits: int = 8,
                 act: bool = True,
                 input_shape: Tuple[int, ...] = (1, 32, 32, 32),
                 weights_shape: Tuple[int, ...] = (32, 32, 3, 3),
                 weights_values: Optional[tvm.nd.array] = None,
                 bias_values: Optional[tvm.nd.array] = None,
                 padding: Tuple[int, int] = (1, 1),
                 strides: Tuple[int, int] = (1, 1),
                 shift_bits: int = 0
                 ):
    """Generate a small network in TVM Relay IR that does 2 requantized convolutions and add their results
    """
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType(input_shape, 'uint8'))
    y = relay.var("input_1", relay.TensorType(input_shape, "uint8"))
    # Get or generate weight_values
    if weights_values is None:
        weights = utils.create_random_array(weights_shape, 
                                            f'int{weight_bits}')
        weights = utils.numpy_to_array(np.ones(weights_shape,f"int{weight_bits}"),f"int{weight_bits}")
    else:
        weights = weights_values
    # Get or generate bias values
    if bias_values is None:
        bias = utils.create_random_array(weights_shape[0], 'int32')
        bias = utils.numpy_to_array(np.ones(weights_shape[0],"int32"),"int32")
    else:
        bias = bias_values
    # Generate the conv2d call
    x, params1 = utils.relay_conv2d_uint8_requant(x, 'conv1', weights, bias, 
                                         padding=padding, 
                                         strides=strides,
                                         act=act, 
                                         shift_bits=shift_bits,
                                         batchnorm=True)
    y, params2 = utils.relay_conv2d_uint8_requant(y, 'conv2', weights, bias, 
                                         padding=padding, 
                                         strides=strides,
                                         act=act, 
                                         shift_bits=shift_bits,
                                         batchnorm=True)
    z = utils.relay_add_uint8_requant(x,y,"add")
    params1.update(params2)
    params = params1
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(z)
    return mod, params