from tvm.relay.dataflow_pattern import wildcard, is_op
from match.partition.partitioning_pattern import PartitioningPattern
from match.partition.utils import add_checks_get_first_op

def fused_conv_pt():
    # Create pattern for a 2D Conv block, with bias and ReLU.
    conv2d = is_op("nn.conv2d")(
        wildcard(), wildcard()
    )
    bias_add = is_op("nn.bias_add")(conv2d, wildcard())
    relu = is_op("nn.relu")(bias_add)
    return relu

def additional_checks_conv2d(node):
    # get conv2d from the node
    conv2d = add_checks_get_first_op(node,"nn.conv2d")
    k_size = conv2d.attrs["kernel_size"]
    # if filter is not 3x3 don't match
    if k_size[0]!=3 or k_size[1]!=3:
        return False
    return True

PartitioningPattern(name="fused_conv",pattern=fused_conv_pt,
                    additional_checks=additional_checks_conv2d)