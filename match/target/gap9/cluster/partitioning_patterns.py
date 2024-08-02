# Imports
import tvm
import logging
from tvm.relay.dataflow_pattern import wildcard, is_op, is_var, is_constant
from match.partition.partitioning_pattern import PartitioningPattern

logger = logging.getLogger("Gap9Cluster")


def batchnorm_pattern(prev_op):
    """Add batchnorm pattern (multiply->add)"""
    mult = is_op("multiply")(prev_op, is_constant())
    add = is_op("add")(mult,is_constant())
    return add

def _requant_pattern(prev_op):
    """Add requant pattern (right_shift -> clip -> cast) to prev_op"""
    right_shift = is_op("right_shift")(prev_op, is_constant())
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip)
    return cast

def conv2d_bnorm_requant_pattern():
    conv2d = is_op("nn.conv2d")(
            wildcard(), wildcard()
    )
    bnorm = batchnorm_pattern(is_op("cast")(conv2d)) | batchnorm_pattern(conv2d)
    return _requant_pattern(bnorm)

def dense_bnorm_requant_pattern():
    dense = is_op("nn.dense")(
            wildcard(), wildcard()
    )
    cast = is_op("cast")(dense)
    bnorm = batchnorm_pattern(cast)
    return _requant_pattern(bnorm)

def dense_out_pattern():
    dense = is_op("nn.dense")(
            wildcard(), wildcard()
    )
    add = is_op("add")(dense, is_constant()) | is_op("add")(is_op("cast")(dense),is_constant())
    return add

def _biasadd_requant_pattern(linear_op):
    """Add pattern bias_add-requant to linear_op"""

    bias_add = is_op("nn.bias_add")(linear_op, wildcard()) | is_op("add")(linear_op, wildcard())
    return _requant_pattern(bias_add)


def conv2d_pattern():
    """Create pattern for conv2D with optional fused relu."""
    conv2d = is_op("nn.conv2d")(
            wildcard(), wildcard()
    )
    return _biasadd_requant_pattern(conv2d | is_op("cast")(conv2d))

def only_conv_2d_and_bias_pattern():
    """Create pattern for conv2D"""
    conv2d = is_op("nn.conv2d")(
        wildcard(),wildcard()
    )
    bias_add = is_op("nn.bias_add")(conv2d, wildcard())
    return bias_add

def fully_connected_pattern():
    """Create pattern for nn.dense with optional fused relu."""

    fc = is_op("nn.dense")(
        wildcard(), wildcard()
    )
    return _biasadd_requant_pattern(fc)


def element_wise_add_pattern():
    """Create pattern for element-wise-add with optional fused relu."""

    cast_a = is_op("cast")(wildcard())
    cast_b = is_op("cast")(wildcard())
    add = is_op("add")(cast_a, cast_b)
    # pattern cast cast add clip casst cast multiply right shift cast
    clip = is_op("clip")(add)
    cast_c = is_op("cast")(clip)
    cast_d = is_op("cast")(cast_c)
    mul = is_op("multiply")(is_constant(),cast_d)
    rshift = is_op("right_shift")(mul, is_constant())
    # pattern cast cast add right shif clip cast
    rshift_clip = is_op("clip")(is_op("right_shift")(add,is_constant()))
    # cast for both paths
    pt = is_op("cast")(rshift | rshift_clip)
    return pt

def _check_requant(pattern):
    """Check if requant pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the op before this sequence if supported
    """
    cast = pattern
    right_shift = cast.args[0].args[0]

    # Check range of shift factor
    shift_factor = right_shift.args[1].data.numpy()
    if shift_factor < 0 or shift_factor > 31:
        logger.warning("shift factor of accelerator operation must be in range [0, 31], but got {shift_factor}. Acceleration for this op is not supported.")
        return None

    right_shift_input = right_shift.args[0]

    return right_shift_input


def _check_biasadd_requant(pattern):
    """Check if bias_add-requant pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the linear op before this sequence if supported
    """

    right_shift_input = _check_requant(pattern)
    if right_shift_input is None:
        return None

    # For now, we don't support linears without bias
    if str(right_shift_input.op.name) not in ["nn.bias_add", "add"]:
        logger.warning("Found conv/dense op without nn.bias_add. Acceleration for this op is not supported.")
        return None

    bias_add = right_shift_input

    # Check bias dtype
    bias_dtype = bias_add.args[1].checked_type.dtype
    if bias_dtype != 'int32':
        logger.warning(f"Expected nn.bias_add parameters to be of type int32, but got {bias_dtype}. Acceleration for this op is not supported.")
        return None

    return bias_add.args[0]

def check_conv2d(pattern):
    """Check if the Conv2D is supported by the soma dory accelerator"""
    conv2d = _check_biasadd_requant(pattern)
    if conv2d is None:
        return False

    num_output_channels = conv2d.args[1].data.shape[0]
    # Don't offload grouped analog convolutions
    if conv2d.args[1].checked_type.dtype == "int2":
        if conv2d.attrs['groups'] != 1:
            return False

    def is_conv2d_attr_value_supported(attrs, name, supported_values):
        attr = attrs[name]

        if isinstance(attr, tvm.ir.container.Array):
            attr = list(attr)

        if attr not in supported_values:
            logger.warning(f"Expected nn.conv2d {name} to be one of {supported_values}, but got {attr}. " +\
                            "Acceleration for this op is not supported.")
            return False

        return True

    def is_filter_and_padding_supported(attrs):
        kernel_size = list()
        if "kernel_size" in dict(attrs) and attrs["kernel_size"]!=None:
            kernel_size = list(attrs["kernel_size"])
        else:
            kernel_size = list([int(v) for v in conv2d.args[1].checked_type.shape][2:])
        kernel_h = kernel_size[0]
        kernel_w = kernel_size[1]
        supported_kernels = [1, 3, 5, 7]
        if (kernel_h not in supported_kernels) or (kernel_w not in supported_kernels):
            logger.warning(f"Expected nn.conv2d kernel width and height to be one of {supported_kernels}, " +\
                           f"but got {kernel_size}. " +\
                            "Acceleration for this op is not supported.")
            return False

        # In topi, padding is [padt, padl, padb, padr]
        padding = list(attrs["padding"])
        # Only support equal left-right and top-bottom padding
        if (padding[0] != padding[2]) or (padding[1] != padding[3]):
            logger.warning(f"Expected equal top and bottom padding, and equal left and right padding," +\
                           f"but got {[padding[0], padding[2]]} and {[padding[1], padding[3]]}, respectively. " +\
                            "Acceleration for this op is not supported.")
            return False

        # Only support output with same output dimension on accelerator
        if (kernel_w - 2*padding[1] != 1) and (kernel_h - 2*padding[0] != 1):
            expected_pad = [(kernel_w - 1) // 2, (kernel_h - 1) // 2]
            logger.warning(f"Accelerator only supports 'SAME' padding. " +\
                           f"Expected nn.conv2d with kernel size {kernel_size} to have padding {expected_pad}, " +\
                           f"but got {padding[:2]}.")
            return False

        return True


    # check conv2d attributes
    if (not is_filter_and_padding_supported(conv2d.attrs)
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'strides', [[1, 1], [2, 2]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'dilation', [[1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'groups', [1, num_output_channels])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_layout', ['OIHW'])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'data_layout', ['NCHW'])):

        return False

    #conv2d_input = conv2d.args[0]
    conv2d_weight = conv2d.args[1]

    weights_dtype = conv2d_weight.data.dtype
    if not weights_dtype.startswith('int'):
        logger.warning(f"Expected Conv2D weights to be of integer type, got {weights_dtype}. \
                        Acceleration for this conv2d is not supported")
        return False

    return True


def check_fully_connected(pattern):
    """Check if the fully connected layer is supported by the soma dory accelerator"""

    fc = _check_biasadd_requant(pattern)
    if fc is None:
        return False

    return True


def partitioning_patterns():
    return [
        PartitioningPattern(name="conv2d_bnorm_requant",pattern=conv2d_bnorm_requant_pattern,ordered_operation="nn.conv2d"),
        PartitioningPattern(name="conv2d_bias_add_requant",pattern=conv2d_pattern,ordered_operation="nn.conv2d"),
        PartitioningPattern(name="conv2d_bias_add",pattern=only_conv_2d_and_bias_pattern,ordered_operation="nn.conv2d"),
        PartitioningPattern(name="dense_bnorm_requant",pattern=dense_bnorm_requant_pattern,ordered_operation="nn.dense"),
        PartitioningPattern(name="dense_bias_add_requant",pattern=fully_connected_pattern,additional_checks=check_fully_connected,ordered_operation="nn.dense"),
        PartitioningPattern(name="add_requant",pattern=element_wise_add_pattern,ordered_operation="add"),
        PartitioningPattern(name="dense_out",pattern=dense_out_pattern,ordered_operation="dense")
    ]