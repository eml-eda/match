from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant
from tvm.relay import Constant
from match.partition.utils import add_checks_get_first_op

USE_ONLY_FC = False  # Set to True to use only fully connected layers in training patterns
IS_FW_TRAIN = True  # Set to True to use training patterns
TRAIN_FW_USE_CLUSTER = not USE_ONLY_FC  # Set to True to use only fully connected layers in training patterns
TRAIN_BW_USE_CLUSTER = not USE_ONLY_FC  # Set to True to use only fully connected layers in training patterns


def conv3d_pt_requant():
    #Create pattern for a 3D Conv block, with bias and ReLU.
    conv3d = is_op("nn.conv3d")(
        wildcard(), wildcard()
    )
    conv3d = is_op("cast")(conv3d) | conv3d
    bias_add = is_op("nn.bias_add")(conv3d, wildcard()) | is_op("add")(conv3d, wildcard())
    scale = is_op("multiply")(conv3d, wildcard()) | is_op("multiply")(wildcard(), conv3d)
    bias = is_op("add")(scale, wildcard()) | is_op("add")(wildcard(), scale)
    right_shift = is_op("right_shift")(bias_add | bias, is_constant())
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip)
    return cast

def conv_pt_requant():
    #Create pattern for a 2D Conv block, with bias and ReLU.
    conv2d = is_op("nn.conv2d")(
        wildcard(), wildcard()
    )
    conv2d = is_op("cast")(conv2d) | conv2d
    bias_add = is_op("nn.bias_add")(conv2d, wildcard()) | is_op("add")(conv2d, wildcard())
    scale = is_op("multiply")(conv2d, wildcard()) | is_op("multiply")(wildcard(), conv2d)
    bias = is_op("add")(scale, wildcard()) | is_op("add")(wildcard(), scale)
    right_shift = is_op("right_shift")(bias_add | bias, is_constant())
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip)
    return cast


def dense_pt_requant():
    """Create pattern for conv2D with optional fused relu."""
    dense = is_op("nn.dense")(
        wildcard(), wildcard()
    )
    dense = is_op("cast")(dense) | dense
    bias_add = is_op("nn.bias_add")(dense, wildcard()) | is_op("add")(dense, wildcard())
    scale = is_op("multiply")(dense, wildcard()) | is_op("multiply")(wildcard(), dense)
    bias = is_op("add")(scale, wildcard()) | is_op("add")(wildcard(), scale)
    right_shift = is_op("right_shift")(bias_add | bias, is_constant())
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip)
    return cast

def dense_pt_out():
    dense = is_op("nn.dense")(
        wildcard(), wildcard()
    )
    add = is_op("add")(dense, is_constant()) | is_op("add")(is_op("cast")(dense),is_constant())
    return add

def add_pt_requant():
    cast_a = is_op("cast")(wildcard())
    cast_b = is_op("cast")(wildcard())
    add = is_op("add")(cast_a , cast_b )
    add = add | is_op("cast")(is_op("add")(wildcard() , wildcard()))
    # pattern cast cast add clip cast cast multiply right shift cast
    mul = is_op("multiply")(is_constant(), add) | is_op("multiply")(add, is_constant())
    rshift = is_op("right_shift")(mul, is_constant())
    # cast for both paths
    pt = is_op("cast")(is_op("clip")(rshift))
    return pt

# training layers 
def conv2d_fw():
    #Create pattern for a 2D Conv block, with bias and ReLU.
    conv2d = is_op("nn.conv2d")(
        wildcard(), is_constant()
    )
    conv2d = is_op("cast")(conv2d) | conv2d
#            bias_add = is_op("nn.bias_add")(conv2d, wildcard()) | is_op("add")(conv2d, wildcard()) | conv2d
    return conv2d

def conv2d_bw():
    #Create pattern for a 2D Conv block, with bias and ReLU.
    conv2d = is_op("nn.conv2d")(
        wildcard(), wildcard()
    )
    conv2d = is_op("cast")(conv2d) | conv2d
    bias_add = is_op("nn.bias_add")(conv2d, wildcard()) | is_op("add")(conv2d, wildcard()) | conv2d
    return bias_add

def conv2d_transpose_ptrain_pt():
    #Create pattern for a 2D Conv transpose block, with bias and ReLU.
    conv2d_transpose = is_op("nn.conv2d_transpose")(
        wildcard(), wildcard()
    )
    return conv2d_transpose

def bw_instance_norm_tail_pt():
    # Create pattern for a backward instance normalization tail layer
    add = is_op("add")(wildcard(), is_constant())
    sqrt = is_op("sqrt")(add)
    divide = is_op("divide")(is_constant(), sqrt)
    repeat = is_op("repeat")(divide)
    reshape = is_op("reshape")(repeat)
    multiply_1 = is_op("multiply")(reshape, reshape)
    multiply_2 = is_op("multiply")(is_constant(), wildcard())
    multiply_3 = is_op("multiply")(multiply_1, reshape)
    multiply_4 = is_op("multiply")(multiply_2, multiply_3)
    multiply_5 = is_op("multiply")(multiply_4, is_constant())
    multiply_6 = is_op("multiply")(wildcard(), multiply_5)
    multiply_7 = is_op("multiply")(wildcard(), reshape)
    multiply_8 = is_op("multiply")(multiply_6, is_constant())
    add_2 = is_op("add")(multiply_7, multiply_8)
    return add_2

def fw_instance_norm_tail_pt():
    # tvmgen_test_bp_tvm_fused_subtract_add_rsqrt_multiply_multiply_add_reshape
    # Create pattern for a forward instance normalization tail layer
    subtract = is_op("subtract")(wildcard(), wildcard())
    add = is_op("add")(wildcard(), is_constant())
    rsqrt = is_op("rsqrt")(add)
    multiply_1 = is_op("multiply")(subtract, rsqrt)
    multiply_2 = is_op("multiply")(multiply_1, is_constant()) | multiply_1
    add_2 = is_op("add")(multiply_2, is_constant()) | multiply_2
    reshape = is_op("reshape")(add_2)
    return reshape

def only_out_uint8(node):
    cast_node = node if node.op.name == "cast" else None
    if cast_node is not None:
        return cast_node.attrs.dtype=="uint8"
    if hasattr(node, 'checked_type') and hasattr(node.checked_type, 'dtype'):
        return node.checked_type.dtype=="uint8"
    return False

def only_out_int32(node):
    cast_node = node if node.op.name == "cast" else None
    if cast_node is not None:
        return cast_node.attrs.dtype=="int32"
    if hasattr(node, 'checked_type') and hasattr(node.checked_type, 'dtype'):
        return node.checked_type.dtype=="int32"
    return False

def only_std_convs(node):
    conv = add_checks_get_first_op(node, "nn.conv2d")
    if not only_out_uint8(node):
        return False
    # theres a pointwise specific pattern
    if tuple([int(i) for i in conv.attrs.kernel_size]) == (1,1):
        return False
    if conv.attrs.groups!=1:
        return False
    if conv.attrs.data_layout!="NHWC":
        return False
    return True

def only_pw_convs(node):
    conv = add_checks_get_first_op(node, "nn.conv2d")
    if not only_out_uint8(node):
        return False
    if tuple([int(i) for i in conv.attrs.kernel_size]) != (1,1):
        return False
    if conv.attrs.groups!=1:
        return False
    if conv.attrs.data_layout!="NHWC":
        return False
    return True

def only_dw_convs(node):
    conv = add_checks_get_first_op(node, "nn.conv2d")
    out_chs = conv.args[1].checked_type.shape[3]
    if not only_out_uint8(node):
        return False
    if conv.attrs.groups!=out_chs:
        return False
    if conv.attrs.data_layout!="NHWC":
        return False
    return True

# checks for training
def std_convs_fp32(node):
    if not TRAIN_FW_USE_CLUSTER:
        return False
    conv = add_checks_get_first_op(node, "nn.conv2d")
    if conv.checked_type.dtype != 'float32':
        return False
    if conv.attrs.groups != 1:
        return False
    # check for dilated convs
    if any([int(val)!=1 for val in conv.attrs.dilation]):
        return False
    return True

def dw_convs_fp32_pulp(node):
    if not TRAIN_FW_USE_CLUSTER:
        return False
    conv = add_checks_get_first_op(node, "nn.conv2d")
    out_chs = conv.args[1].checked_type.shape[0]
    in_chs = conv.args[1].checked_type.shape[1]
    if conv.checked_type.dtype != 'float32':
        return False
    if conv.attrs.groups == 1:
        return False
    if conv.attrs.groups != out_chs or in_chs != 1 :
        return False
    if conv.attrs.data_layout!="NCHW":
        return False
    # check for dilated convs
    if any([int(val)!=1 for val in conv.attrs.dilation]):
        return False
    #add checks for stride (temp)
    if conv.attrs.strides[0] != 1 or conv.attrs.strides[1] != 1:
        return False
    return True

def conv2d_transpose_ptrain_check(node):
    if not TRAIN_BW_USE_CLUSTER:
        return False
    conv2d_transpose = add_checks_get_first_op(node, "nn.conv2d_transpose")
    if conv2d_transpose.checked_type.dtype != 'float32':
        return False
    # check for dilated convs
    if any([int(val)!=1 for val in conv2d_transpose.attrs.dilation]):
        return False
    return True

def conv2d_bw_check(node):
    if not TRAIN_BW_USE_CLUSTER:
        return False
    conv2d = add_checks_get_first_op(node, "nn.conv2d")
    out_chs = conv2d.args[1].checked_type.shape[0]
    if isinstance(conv2d.args[1], Constant):
        return False
    if conv2d.attrs.data_layout != "NCHW":
        return False
    if conv2d.checked_type.dtype != 'float32':
        return False
    if conv2d.attrs.groups != out_chs and conv2d.attrs.groups != 1:
        return False
    return True
