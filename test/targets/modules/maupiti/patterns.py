# Imports
import logging
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant
from match.partition.partitioning_pattern import PartitioningPattern

logger = logging.getLogger("Maupiti")

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

def maxpool2d_pattern():
    """Create pattern for conv2D with optional fused relu."""
    pattern = is_op("nn.max_pool2d")(wildcard())
    
    return pattern

def maupiti_patterns():
    return [
        PartitioningPattern(name="conv2d_bnorm_requant",pattern=conv2d_bnorm_requant_pattern,ordered_operation="nn.conv2d"),
        PartitioningPattern(name="dense_bnorm_requant",pattern=dense_bnorm_requant_pattern,ordered_operation="nn.dense"),
        PartitioningPattern(name="dense_out",pattern=dense_out_pattern,ordered_operation="nn.dense"),
        PartitioningPattern(name="maxpool2d",pattern=maxpool2d_pattern,ordered_operation="nn.max_pool2d"),
    ]