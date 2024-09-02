# Imports
import tvm
import logging
from tvm.relay.dataflow_pattern import wildcard, is_op, is_var, is_constant
from match.partition.partitioning_pattern import PartitioningPattern

logger = logging.getLogger("Gap9NE16")


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

def conv2d_biasadd_requant_pattern():
    conv2d = is_op("nn.conv2d")(
            wildcard(), wildcard()
    )
    bnorm = is_op("nn.bias_add")(is_op("cast")(conv2d), is_constant()) | is_op("nn.bias_add")(conv2d, is_constant())
    return _requant_pattern(bnorm)


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

def get_conv2d(pattern):
    while pattern.op.name!="nn.conv2d":
        pattern=pattern.args[0]
    return pattern

def check_conv2d(pattern):
    """Check if the Conv2D is supported by the soma dory accelerator"""
    conv2d = get_conv2d(pattern)
    if conv2d is None:
        return False

    num_output_channels = conv2d.args[1].data.shape[0]
    num_input_channels = conv2d.args[1].data.shape[1]

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

        # In topi, padding is [padt, padl, padb, padr]
        padding = list(attrs["padding"])
        
        return (all([ksize==3 for ksize in kernel_size])\
                and all([pad==1 for pad in padding])) or (all([ksize==1 for ksize in kernel_size]) \
                                                          and all([pad==0 for pad in padding]) and attrs["groups"]==1)


    # check conv2d attributes
    if (not is_filter_and_padding_supported(conv2d.attrs)
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'strides', [[1, 1], [2, 2]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'dilation', [[1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'groups', [1, num_output_channels])
        #or not (((num_input_channels%16)==0 and conv2d.attrs["groups"]==1)
        #or ((num_output_channels%16)==0 and conv2d.attrs["groups"]>1))
        ):

        return False

    return True

def partitioning_patterns():
    return [
        PartitioningPattern(name="conv2d_bnorm_requant",pattern=conv2d_bnorm_requant_pattern,
                            additional_checks=check_conv2d,
                            ordered_operation="nn.conv2d"),
        PartitioningPattern(name="conv2d_bias_add_requant",pattern=conv2d_biasadd_requant_pattern,
                            additional_checks=check_conv2d,
                            ordered_operation="nn.conv2d"),
    ]
