import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    rewrite,
    wildcard,
    is_op,
    is_constant,
)
from tvm.topi.utils import get_const_tuple


class GAPopCountConv(DFPatternCallback):
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.data = wildcard()
        self.weight = wildcard()
        self.conv = is_op("nn.conv2d")(self.data, self.weight)
        self.pattern = self.conv
        self.counter = 0

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        weight = node_map[self.weight][0]
        attrs = post.attrs
        stride_h, stride_w = attrs.strides
        kernel_size = [int(x) for x in attrs.kernel_size]
        padding = [int(x) for x in attrs.padding]
        groups = attrs.groups
        print(f"Conv Layer {self.counter} with kernel size: {kernel_size} and groups: {groups}")
        self.counter += 1
        # data size
        data_shape = relay.op.shape_of(data)  # shape = (N, C, H, W)
        #data_shape = data.checked_type.shape
        data_shape = get_const_tuple(data.checked_type.shape)
        data_nb, data_c, data_h, data_w = data_shape
        print("Input shape:", data_shape)



        # compute output shape
        out_h = (data_h + padding[0] + padding[2] - kernel_size[0]) // stride_h + 1
        out_w = (data_w + padding[1] + padding[3] - kernel_size[1]) // stride_w + 1
        out_shape = [data_nb, attrs.channels, out_h, out_w]
        print("Output shape:", out_shape)

        # compute number of operations 
        mac = data_nb * out_h * out_w * attrs.channels * data_c * kernel_size[0] * kernel_size[1] 
        if groups > 1:
            mac //= groups
        print("ConvMACs:", mac)  # Debugging output, can be removed later
        print("Number of parameters:", weight.data.numpy().size, f"({weight.data.numpy().size*4} Bytes)")
        print("\n")
        # create a new constant for the number of operations

        return post


@tvm.ir.transform.module_pass(opt_level=0)
class GAPopCount:
    """Find and rewrite MATCH ONNX requant to requant for internal use:
    """

    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(GAPopCountConv(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)