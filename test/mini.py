import math
from typing import Tuple
from match.relay.utils.utils import create_random_array, numpy_to_array
from match.utils.utils import get_random_np_array
import tvm
from tvm import relay

def create_dense_conv_dense_ex(inp_features:int=256,out_features:int=128,
                            inp_shape:Tuple=(32,32),fil_shape:Tuple=(1,1),
                            padding:Tuple=(0,0,0,0),strides:Tuple=(1,1),
                            groups:int=1,requant_pattern:bool=False,
                            right_shift:int=1,**kwargs):
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType((1,inp_features), "uint8"))
    # Get or generate weight_values
    weights_1 = create_random_array((out_features*math.prod(inp_shape),inp_features),"int8")
    weights_2 = create_random_array((inp_features,out_features)+fil_shape,"int8")
    weights_3 = create_random_array((out_features,inp_features*math.prod([int(inp_shape[idx]/strides[idx]) for idx in range(len(inp_shape))])), "int8")
    # Get or generate bias values
    bias_1 = create_random_array((out_features*math.prod(inp_shape),), "int32")
    bias_2 = create_random_array((inp_features,), "int32")
    bias_3 = create_random_array((out_features,), "int32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_1_name = "dense_1_weights"
    bias_1_name = "dense_1_bias"
    weights_2_name = "conv_weights"
    bias_2_name = "conv_bias"
    weights_3_name = "dense_2_weights"
    bias_3_name = "dense_2_bias"

    # define relay input vars
    w_1 = relay.var(weights_1_name, relay.TensorType(weights_1.shape, weights_1.dtype))
    w_2 = relay.var(weights_2_name, relay.TensorType(weights_2.shape, weights_2.dtype))
    w_3 = relay.var(weights_3_name, relay.TensorType(weights_3.shape, weights_3.dtype))
    b_1 = relay.var(bias_1_name, relay.TensorType(bias_1.shape, bias_1.dtype))
    b_2 = relay.var(bias_2_name, relay.TensorType(bias_2.shape, bias_2.dtype))
    b_3 = relay.var(bias_3_name, relay.TensorType(bias_3.shape, bias_3.dtype))

    # define weights and bias values in params
    params = {weights_1_name: weights_1, bias_1_name: bias_1,
              weights_2_name: weights_2, bias_2_name: bias_2,
              weights_3_name: weights_3, bias_3_name: bias_3}

    # define operations
    x = relay.op.nn.dense(x, w_1, out_dtype=bias_1.dtype)
    x = relay.op.nn.bias_add(x, b_1, axis=-1)
    if requant_pattern:
        x = relay.op.right_shift(x, relay.const(right_shift))
        x = relay.op.clip(x, a_min=0, a_max=255)
        x = relay.op.cast(x, "uint8")
    else:
        x = relay.op.nn.relu(x)
    x = relay.op.cast(x, "uint8")
    x = relay.op.reshape(x, (1, out_features)+inp_shape)
    x = relay.op.nn.conv2d(x, w_2,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           kernel_size=fil_shape,
                           out_dtype="int32",
                           )
    x = relay.op.nn.bias_add(x, b_2, axis=1)
    if requant_pattern:
        x = relay.op.right_shift(x, relay.const(right_shift))
        x = relay.op.clip(x, a_min=0, a_max=255)
        x = relay.op.cast(x, "uint8")
    else:
        x = relay.op.nn.relu(x)
    x = relay.op.reshape(x, (1, inp_features*math.prod([int(inp_shape[idx]/strides[idx]) for idx in range(len(inp_shape))])))
    x = relay.op.nn.dense(x, w_3, out_dtype=bias_3.dtype)
    x = relay.op.nn.bias_add(x, b_3, axis=-1)
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

def create_fp_conv_vit(div_out_chs_by:int=1, out_ch:int=384, **kwargs):
    inp_shape = (224,224)
    inp_ch = 3
    fil_shape = (16,16)
    strides = (16,16)
    out_ch = out_ch//div_out_chs_by
    padding = (0,0,0,0)
    groups = 1
    x = relay.var("input_0", relay.TensorType((1,inp_ch)+inp_shape))
    # Get or generate weight_values
    weights_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_ch,inp_ch)+fil_shape),dtype="float32")
    # Get or generate bias values
    bias_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_ch,)),dtype="float32")
    concat_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,out_ch)),dtype="float32")
    add_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,197,out_ch)),dtype="float32")
    # Generate the conv2d call
    # define weights and bias variables
    weights_name = "conv_weights"
    bias_name = "conv_bias"
    concat_name = "conv_concat"
    add_name = "conv_add"
    # define weights and bias values in params
    params = {weights_name: weights_c, bias_name: bias_c,
              concat_name: concat_c, add_name: add_c}

    # define operations
    weights_v = relay.var(weights_name, relay.TensorType(weights_c.shape, weights_c.dtype))
    x = relay.op.nn.conv2d(x, weights_v,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           kernel_size=fil_shape,
                           )
    bias_v = relay.var(bias_name, relay.TensorType(bias_c.shape, bias_c.dtype))
    x = relay.op.nn.bias_add(x, bias_v, axis=1)
    x = relay.op.reshape(x, newshape=(1,out_ch,196))
    x = relay.op.transpose(x,axes=(0,2,1))
    concat_v = relay.var(concat_name, relay.TensorType(concat_c.shape,concat_c.dtype))
    x = relay.op.concatenate(relay.Tuple([x,concat_v]), axis=1)
    add_v = relay.var(add_name, relay.TensorType(add_c.shape,add_c.dtype))
    x = relay.op.add(x,add_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_layer_norm_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, **kwargs):
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((1,out_ch)+inp_shape))
    # Get or generate weight_values
    beta_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=inp_shape),dtype="float32")
    gamma_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=inp_shape),dtype="float32")
    # Generate the conv2d call
    # define weights and bias variables
    beta_name = "layer_norm_beta"
    gamma_name = "layer_norm_gamma"
    # define weights and bias values in params
    params = {beta_name: beta_c, gamma_name: gamma_c}

    # define operations
    beta_v = relay.var(beta_name, relay.TensorType(beta_c.shape, beta_c.dtype))
    gamma_v = relay.var(gamma_name, relay.TensorType(gamma_c.shape, gamma_c.dtype))
    x = relay.op.nn.layer_norm(x,gamma=gamma_v,beta=beta_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_fp_proj_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, projections:int=3, **kwargs):
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((out_ch,)+inp_shape))
    # Get or generate weight_values
    weights_name = "proj_weights"
    add_name = "proj_add"
    weights_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(inp_size*projections,inp_size)),dtype="float32")
    add_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,inp_size*projections)),dtype="float32")
    params = {weights_name: weights_c, add_name: add_c}
    weights_v = relay.var(weights_name, relay.TensorType(weights_c.shape, weights_c.dtype))
    x = relay.op.nn.dense(x,weights_v)
    add_v = relay.var(add_name, relay.TensorType(add_c.shape, add_c.dtype))
    x = relay.op.add(x,add_v)
    x = relay.op.reshape(x,newshape=(1,out_ch,projections,inp_size//64,64))
    x = relay.op.transpose(x,axes=(2,0,3,1,4))
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_attention_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, **kwargs):
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((3,1,inp_size//64,out_ch,64)))
    # Get or generate weight_values
    weights_name = "attention_weights"
    add_name = "attention_add"
    sqrt_1_name = "attention_sqrt_1"
    sqrt_2_name = "attention_sqrt_2"
    weights_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(inp_size,inp_size)),dtype="float32")
    add_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,inp_size)),dtype="float32")
    sqrt_1_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,)),dtype="float32")
    sqrt_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,)),dtype="float32")
    params = {sqrt_1_name: sqrt_1_c, sqrt_2_name: sqrt_2_c, add_name: add_c, weights_name: weights_c}
    x = relay.op.split(x,indices_or_sections=(1,2))
    q = relay.op.reshape(x[0], newshape=(1,1,inp_size//64,out_ch,64))
    k = relay.op.reshape(x[1], newshape=(1,1,inp_size//64,out_ch,64))
    v = relay.op.reshape(x[2], newshape=(1,1,inp_size//64,out_ch,64))
    q = relay.op.squeeze(q,axis=0)
    sqrt_1_v = relay.var(sqrt_1_name, relay.TensorType(sqrt_1_c.shape, sqrt_1_c.dtype))
    q = relay.op.multiply(q,sqrt_1_v)
    k = relay.op.squeeze(k,axis=0)
    k = relay.op.transpose(k,axes=(0,1,3,2))
    sqrt_2_v = relay.var(sqrt_2_name, relay.TensorType(sqrt_2_c.shape, sqrt_2_c.dtype))
    k = relay.op.multiply(k,sqrt_2_v)
    k = relay.op.reshape(k,newshape=(-1,64,out_ch))
    k = relay.op.transpose(k, axes=(0,2,1))
    q = relay.op.reshape(q,newshape=(-1,out_ch,64))
    qk = relay.op.nn.batch_matmul(q,k)
    qk = relay.op.nn.softmax(qk,axis=-1)
    v = relay.op.squeeze(v,axis=0)
    v = relay.op.reshape(v,newshape=(-1,out_ch,64))
    v = relay.op.transpose(v, axes=(0,2,1))
    qk = relay.op.reshape(qk,newshape=(-1,out_ch,out_ch))
    x = relay.op.nn.batch_matmul(qk,v)
    x = relay.op.reshape(x,newshape=(1,inp_size//64,out_ch,64))
    x = relay.op.transpose(x,axes=(0,2,1,3))
    x = relay.op.reshape(x,newshape=(out_ch,inp_size))
    weights_v = relay.var(weights_name, relay.TensorType(weights_c.shape, weights_c.dtype))
    x = relay.op.nn.dense(x,weights_v)
    add_v = relay.var(add_name, relay.TensorType(add_c.shape, add_c.dtype))
    x = relay.op.add(x,add_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_addlayernorm_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, **kwargs):
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((1,out_ch)+inp_shape))
    y = relay.var("input_1", relay.TensorType((1,out_ch)+inp_shape))
    # Get or generate weight_values
    beta_name = "layer_norm_beta"
    gamma_name = "layer_norm_gamma"
    beta_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=inp_shape),dtype="float32")
    gamma_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=inp_shape),dtype="float32")
    params = {beta_name: beta_c, gamma_name: gamma_c}
    x = relay.op.add(x,y)
    beta_v = relay.var(beta_name, relay.TensorType(beta_c.shape, beta_c.dtype))
    gamma_v = relay.var(gamma_name, relay.TensorType(gamma_c.shape, gamma_c.dtype))
    x = relay.op.nn.layer_norm(x,gamma=gamma_v,beta=beta_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_add_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, **kwargs):
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((1,out_ch)+inp_shape))
    y = relay.var("input_1", relay.TensorType((1,out_ch)+inp_shape))
    # Get or generate weight_values
    params = {}
    x = relay.op.add(x,y)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_classifier_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, out_neurons:int=10, **kwargs):
    inp_shape = (inp_size,)
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((1,out_ch)+inp_shape))
    # Get or generate weight_values
    weights_name = "classifier_weights"
    add_name = "classifier_add"
    weights_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(out_neurons,inp_size)),dtype="float32")
    add_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,out_neurons)),dtype="float32")
    params = {weights_name: weights_c, add_name: add_c}
    x = relay.op.take(x,relay.const(0),axis=1)
    weights_v = relay.var(weights_name, relay.TensorType(weights_c.shape, weights_c.dtype))
    x = relay.op.nn.dense(x,weights_v)
    add_v = relay.var(add_name, relay.TensorType(add_c.shape, add_c.dtype))
    x = relay.op.add(x,add_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def gelu_helper(data):
    const1 = relay.const(math.sqrt(2.0))
    const2 = relay.const(1.0)
    const3 = relay.const(0.5)
    divisor = relay.op.divide(data, const1)
    val_erf = relay.op.erf(divisor)
    added_erf = relay.op.add(val_erf, const2)
    mul1 = relay.op.multiply(data, added_erf)
    out = relay.op.multiply(mul1, const3)
    return out

def create_densegelu_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, projections:int=3, **kwargs):
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((out_ch,inp_size)))
    # Get or generate weight_values
    add_1_name = "mlp_add_1"
    add_2_name = "mlp_add_2"
    weights_1_name = "mlp_weights_1"
    weights_2_name = "mlp_weights_2"
    add_1_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=((1,1,(inp_size+128)*projections))),dtype="float32")
    add_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,inp_size)),dtype="float32")
    weights_1_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=((inp_size+128)*projections,inp_size)),dtype="float32")
    weights_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(inp_size,(inp_size+128)*projections)),dtype="float32")
    params = {add_1_name: add_1_c, add_2_name: add_2_c, weights_1_name: weights_1_c, weights_2_name: weights_2_c}
    weights_1_v = relay.var(weights_1_name, relay.TensorType(weights_1_c.shape, weights_1_c.dtype))
    x = relay.op.nn.dense(x,weights_1_v)
    add_1_v = relay.var(add_1_name, relay.TensorType(add_1_c.shape, add_1_c.dtype))
    x = relay.op.add(x,add_1_v)
    x = gelu_helper(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_densemlp_fp_vit(div_out_chs_by:int=1, out_ch:int=197, inp_size: int = 384, projections:int=3, **kwargs):
    out_ch = out_ch//div_out_chs_by
    x = relay.var("input_0", relay.TensorType((out_ch,(inp_size+128)*projections)))
    # Get or generate weight_values
    add_2_name = "mlp_add_2"
    weights_2_name = "mlp_weights_2"
    add_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(1,1,inp_size)),dtype="float32")
    weights_2_c = numpy_to_array(np_arr=get_random_np_array(dtype="float32",shape=(inp_size,(inp_size+128)*projections)),dtype="float32")
    params = {add_2_name: add_2_c, weights_2_name: weights_2_c}
    weights_2_v = relay.var(weights_2_name, relay.TensorType(weights_2_c.shape, weights_2_c.dtype))
    x = relay.op.nn.dense(x,weights_2_v)
    add_2_v = relay.var(add_2_name, relay.TensorType(add_2_c.shape, add_2_c.dtype))
    x = relay.op.add(x,add_2_v)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params

def create_easy_dense_int32_ex(inp_features: int=48, out_features: int=16, **kwargs):
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType((1,inp_features), "int32"))
    # define weights and bias variables
    num_nodes = (inp_features//out_features) -1
    weights = []
    biases = []
    weights_names = []
    biases_names = []
    weights_vars = []
    biases_vars = []
    params = {}
    for idx in range(num_nodes):
        dense_weights = create_random_array((inp_features/(idx+2),inp_features/(idx+1)),"int32")
        dense_bias = create_random_array((inp_features/(idx+2),),"int32")
        dense_weights_name = f"dense_{idx}_weights"
        dense_bias_name = f"dense_{idx}_bias"
        weights.append(dense_weights)
        biases.append(dense_bias)
        weights_names.append(dense_weights_name)
        biases_names.append(dense_bias_name)
        weights_vars.append(relay.var(dense_weights_name, relay.TensorType(dense_weights.shape, dense_weights.dtype)))
        biases_vars.append(relay.var(dense_bias_name, relay.TensorType(dense_bias.shape, dense_bias.dtype)))
        params[dense_weights_name] = dense_weights
        params[dense_bias_name] = dense_bias

    # define operations
    for idx in range(num_nodes):
        x = relay.op.nn.dense(x, weights_vars[idx])
        x = relay.op.nn.bias_add(x, biases_vars[idx], axis=-1)
        x = relay.op.nn.relu(x)
    
    # x = relay.op.nn.softmax(x)
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params