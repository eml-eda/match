# TVM imports
import numpy as np
from match import ops
from match.dim.dim import MatchDim, DimDependency
from match.parser.tvm_parser import MatchTVMParser
from match.tensor.tensor import MatchTensor

class MatchRelayParser(MatchTVMParser):
    def __init__(self, node, args_list = ..., exec_module = None, pattern_name = "", partitioned = False, pattern_inst=None, match_node = None):
        super().__init__(node, args_list, exec_module, pattern_name, partitioned, pattern_inst, match_node)
        self.visit_router = {
            "nn.conv1d": self.visit_conv_1d,
            "nn.conv2d": self.visit_conv_2d,
            "nn.conv2d_transpose": self.visit_conv_2d_transpose,
            "nn.conv3d": self.visit_conv_3d,
            "nn.instance_norm": self.visit_instance_norm,
            "cast": self.visit_cast,
            "right_shift": self.visit_right_shift,
            "clip": self.visit_clip,
            "nn.bias_add": self.visit_bias_add,
            "nn.dense": self.visit_dense,
            "add": self.visit_add,
            "multiply": self.visit_multiply,
            "nn.relu": self.visit_relu,
            "sqrt": self.visit_sqrt,
            "divide": self.visit_divide,
            "repeat": self.visit_repeat,
            "reshape": self.visit_reshape,
            "sum": self.visit_sum,
            "subtract": self.visit_subtract,
            "rsqrt": self.visit_rsqrt,
            "nn.batch_matmul": self.visit_batch_matmul,
        }

    def visit_rsqrt(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call, call.args[0], 0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)
        out_tensor = MatchTensor(name=name, dims=inp_tensor.dims, dtype=inp_tensor.dtype, tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name] = out_tensor
        op = ops.MatchOpRsqrt(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
        )
        self.update_match_node(op=op, call=call, name=name)

    def visit_subtract(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        if inp_tensor.layout=="":
            if len(inp_tensor.dims)==4:
                inp_tensor.layout = "NCHW"
            elif len(inp_tensor.dims)==3:
                inp_tensor.layout = "NCH"
            elif len(inp_tensor.dims)==2:
                inp_tensor.layout = "NC"
            elif len(inp_tensor.dims)==1:
                inp_tensor.layout = "N"
        odtype = call.checked_type.dtype
        out_dims, axeses = self.check_broadcasting_and_get_out_dims(inp_tensor=inp_tensor, w_tensor=w_tensor)
        out_tensor = MatchTensor(name=name,dims=out_dims,dtype=np.dtype(odtype),tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op=ops.MatchOpSubtract(
            out_arr=[out_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            axis=-1 if len(axeses)>1 or len(axeses)==0 else axeses[0],
            subtractor=int(w_tensor.data) if w_tensor.tensor_type=="const" and len(w_tensor.dims)==0 else 0,
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_relu(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)
        out_tensor = MatchTensor(name=name,dims=inp_tensor.dims,dtype=inp_tensor.dtype,tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op = ops.MatchOpReLU(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_cast(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)
        out_tensor = MatchTensor(name=name,dims=inp_tensor.dims, dtype=np.dtype(attrs.dtype),
                                 tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op = ops.MatchOpCast(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
            cast_dtype=np.dtype(attrs.dtype)
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_right_shift(self, call, attrs, name):
        # nothing to do actually, right shift has no attrs and the arg is already saved before
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        out_tensor = MatchTensor(name=name,dims=inp_tensor.dims,dtype=inp_tensor.dtype,tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op = ops.MatchOpRightShift(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
            right_shift=int(call.args[1].data.numpy()),
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_clip(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)
        out_tensor = MatchTensor(name=name,dims=inp_tensor.dims,dtype=inp_tensor.dtype,tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op = ops.MatchOpClip(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
            clip_max=int(attrs.a_max),
            clip_min=int(attrs.a_min)
        )
        self.update_match_node(op=op,call=call,name=name)


    def visit_bias_add(self, call, attrs, name):
        axis = int(attrs.axis) if hasattr(attrs,"axis") else 0
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        for w_dim in w_tensor.dims:
            if w_dim.size!=1:
                self.update_all_dim_names_occurrences_with(old_dim_name=w_dim.name,new_dim_name=inp_tensor.dims[-1].name if axis>=len(inp_tensor.dims) or axis<0 else inp_tensor.dims[axis].name)
        if inp_tensor.layout=="":
            if len(inp_tensor.dims)==4:
                inp_tensor.layout = "NCHW"
            elif len(inp_tensor.dims)==3:
                inp_tensor.layout = "NCH"
            elif len(inp_tensor.dims)==2:
                inp_tensor.layout = "NC"
            elif len(inp_tensor.dims)==1:
                inp_tensor.layout = "N"
        w_tensor.layout = inp_tensor.layout
        out_tensor = MatchTensor(name=name,dims=inp_tensor.dims,dtype=inp_tensor.dtype,tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op=ops.MatchOpBiasAdd(
            out_arr=[out_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            axis=axis,
            bias=int(w_tensor.data) if w_tensor.tensor_type=="const" and len(w_tensor.dims)==0 else 0,
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_multiply(self,call,atts,name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        if inp_tensor.layout=="":
            if len(inp_tensor.dims)==4:
                inp_tensor.layout = "NCHW"
            elif len(inp_tensor.dims)==3:
                inp_tensor.layout = "NCH"
            elif len(inp_tensor.dims)==2:
                inp_tensor.layout = "NC"
            elif len(inp_tensor.dims)==1:
                inp_tensor.layout = "N"
            else:
                print(f"[RELAY PARSER]: The input tensor {inp_name} has an unknown layout")
        odtype = call.checked_type.dtype
        out_dims, axeses = self.check_broadcasting_and_get_out_dims(inp_tensor=inp_tensor, w_tensor=w_tensor)
        out_tensor = MatchTensor(name=name,dims=out_dims,dtype=np.dtype(odtype),tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op=ops.MatchOpMultiply(
            out_arr=[out_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            axis=-1 if len(axeses)>1 or len(axeses)==0 else axeses[0],
            multiplier=int(w_tensor.data) if w_tensor.tensor_type=="const" and len(w_tensor.dims)==0 else 1,
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_dense(self, call, attrs, name):
        ishape = [int(v) for v in call.args[0].checked_type.shape]
        wshape = [int(v) for v in call.args[1].checked_type.shape]
        inp_features = wshape[1]
        out_features = wshape[0]
        odtype = call.checked_type.dtype
        if ishape[-1] != wshape[1]:
            raise NotImplementedError(f"[RELAY PARSER]: The weights shape in the dense operation are not correct")
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        # well consider the case where the multiplied dimension is the last one
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        if inp_tensor.layout=="":
            if len(inp_tensor.dims)==4:
                inp_tensor.layout = "NCHW"
            elif len(inp_tensor.dims)==3:
                inp_tensor.layout = "NCH"
            elif len(inp_tensor.dims)==2:
                inp_tensor.layout = "NC"
            elif len(inp_tensor.dims)==1:
                inp_tensor.layout = "N"
        w_tensor.layout = inp_tensor.layout
        self.update_all_dim_names_occurrences_with(old_dim_name=w_tensor.dims[-1].name,new_dim_name=inp_tensor.dims[-1].name)
        out_dims = inp_tensor.dims[:-1]+[w_tensor.dims[0]]
        out_tensor = MatchTensor(name=name,dims=out_dims,dtype=np.dtype(odtype),tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op = ops.MatchOpDense(
            out_arr=[out_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            inp_features=inp_features,
            out_features=out_features,
            out_dtype=np.dtype(attrs.out_dtype) if attrs.out_dtype!="" else np.dtype(odtype),
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_add(self,call,attrs,name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        if inp_tensor.layout=="":
            if len(inp_tensor.dims)==4:
                inp_tensor.layout = "NCHW"
            elif len(inp_tensor.dims)==3:
                inp_tensor.layout = "NCH"
            elif len(inp_tensor.dims)==2:
                inp_tensor.layout = "NC"
            elif len(inp_tensor.dims)==1:
                inp_tensor.layout = "N"

        odtype = call.checked_type.dtype
        out_dims, axeses = self.check_broadcasting_and_get_out_dims(inp_tensor=inp_tensor, w_tensor=w_tensor)
        out_tensor = MatchTensor(name=name,dims=out_dims,dtype=np.dtype(odtype),tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op=ops.MatchOpAdd(
            out_arr=[out_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            axis=-1 if len(axeses)>1 or len(axeses)==0 else axeses[0],
            adder=int(w_tensor.data) if w_tensor.tensor_type=="const" and len(w_tensor.dims)==0 else 0,
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_sqrt(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call, call.args[0], 0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)
        out_tensor = MatchTensor(name=name, dims=inp_tensor.dims, dtype=inp_tensor.dtype, 
                                tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name] = out_tensor
        op = ops.MatchOpSqrt(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
        )
        self.update_match_node(op=op, call=call, name=name)

    def visit_divide(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call, call.args[0], 0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call, call.args[1], 1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        self.infer_layout_if_empty(inp_tensor)
        self.infer_layout_if_empty(w_tensor)
        odtype = call.checked_type.dtype
        out_dims, axeses = self.check_broadcasting_and_get_out_dims(inp_tensor=inp_tensor, w_tensor=w_tensor)
        out_tensor = MatchTensor(name=name, dims=out_dims, dtype=np.dtype(odtype), 
                                tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name] = out_tensor
        op = ops.MatchOpDivide(
            out_arr=[out_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            axis=-1 if len(axeses) > 1 or len(axeses) == 0 else axeses[0],
        )
        self.update_match_node(op=op, call=call, name=name)

    def visit_repeat(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call, call.args[0], 0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)
        
        # Get repeat parameters
        repeats = int(attrs.repeats) if hasattr(attrs, 'repeats') else 1
        axis = int(attrs.axis) if hasattr(attrs, 'axis') else 0
        
        # Calculate output dimensions
        out_dims = inp_tensor.dims.copy()
        if 0 <= axis < len(out_dims) and repeats > 1:
            # Update the dimension size at the specified axis
            out_dims[axis] = MatchDim(
                name=f"{name}_repeat_dim_{axis}", 
                size=inp_tensor.dims[axis].size * repeats
            )
            inp_tensor.dims[axis].dim_dependency = DimDependency(
                size_dependencies=[(inp_tensor.dims[axis], 1/repeats)],
                idx_dependencies=[(inp_tensor.dims[axis], 1/repeats)]
            )
            self.node_all_dims[out_dims[axis].name] = out_dims[axis]
        
        out_tensor = MatchTensor(name=name, dims=out_dims, dtype=inp_tensor.dtype,
                                tensor_type="output", layout=inp_tensor.layout)
        self.calls_tensors[name] = out_tensor
        op = ops.MatchOpRepeat(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
            repeats=repeats,
            axis=axis,
        )
        self.update_match_node(op=op, call=call, name=name)

    def visit_instance_norm(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call, call.args[0], 0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)

        # scale (gamma) and shift (beta) are optional in relay.nn.instance_norm signature but
        # in our patterns we expect them (see fw_instance_norm pattern). Fetch them if present.
        gamma_tensor = beta_tensor = None
        const_arr = []
        if len(call.args) > 1:
            gamma_name, gamma_tensor, gamma_type = self.get_name_and_tensor_of_arg(call, call.args[1], 1)
            self.update_if_intermediate_tensor(tensor=gamma_tensor, name=gamma_name)
            const_arr.append(gamma_tensor)
        if len(call.args) > 2:
            beta_name, beta_tensor, beta_type = self.get_name_and_tensor_of_arg(call, call.args[2], 2)
            self.update_if_intermediate_tensor(tensor=beta_tensor, name=beta_name)
            const_arr.append(beta_tensor)

        # Get instance normalization parameters
        eps = float(attrs.epsilon) if hasattr(attrs, 'epsilon') else 1e-5
        momentum = float(attrs.momentum) if hasattr(attrs, 'momentum') else 0.9
        # NOTE: axis and layout could be extended here if needed

        # Create output tensor with the same dimensions and dtype as input
        out_tensor = MatchTensor(
            name=name, dims=inp_tensor.dims, dtype=inp_tensor.dtype,
            tensor_type="output", layout=inp_tensor.layout
        )
        self.calls_tensors[name] = out_tensor
        op = ops.MatchOpInstanceNorm(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
            const_arr=const_arr,
            epsilon=eps,
            momentum=momentum,
        )
        self.update_match_node(op=op, call=call, name=name)

    def visit_reshape(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call, call.args[0], 0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)
        
        # Get new shape from attributes or from constant argument
        if hasattr(attrs, 'newshape') and attrs.newshape:
            new_shape = [int(v) for v in attrs.newshape]
        elif len(call.args) > 1:
            # newshape might be passed as second argument
            new_shape = [int(v) for v in call.args[1].data.numpy()]
        else:
            # Get from checked_type
            new_shape = [int(v) for v in call.checked_type.shape]
        
        # Create new dimensions
        out_dims = inp_tensor.dims
        out_layout = inp_tensor.layout
        # TODO: Handle layout changes if needed, needs to find a way to make dependencies
        # if new_shape != [inp_tensor.dims[i].size for i in range(len(inp_tensor.dims))]:
        #     for i, size in enumerate(new_shape):
        #         dim = MatchDim(name=f"{name}_reshape_dim_{i}", size=size)
        #         out_dims.append(dim)
        #         self.node_all_dims[dim.name] = dim
            
        #     # Determine layout based on number of dimensions
        #     if len(out_dims) == 4:
        #         out_layout = "NCHW"
        #     elif len(out_dims) == 3:
        #         out_layout = "NCH"
        #     elif len(out_dims) == 2:
        #         out_layout = "NC"
        #     elif len(out_dims) == 1:
        #         out_layout = "N"
        
        out_tensor = MatchTensor(name=name, dims=out_dims, dtype=inp_tensor.dtype,
                                tensor_type="output", layout=out_layout)
        self.calls_tensors[name] = out_tensor
        op = ops.MatchOpReshape(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
            newshape=new_shape,
        )
        self.update_match_node(op=op, call=call, name=name)

    def visit_sum(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call, call.args[0], 0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        self.infer_layout_if_empty(inp_tensor)
        
        # Get sum parameters
        axis = attrs.axis if hasattr(attrs, 'axis') and attrs.axis is not None else None
        keepdims = bool(attrs.keepdims) if hasattr(attrs, 'keepdims') else False
        
        # Calculate output dimensions
        out_dims = inp_tensor.dims.copy()
        if axis is not None:
            if isinstance(axis, (list, tuple)):
                # Multiple axes - remove them in reverse order to maintain indices
                for ax in sorted(axis, reverse=True):
                    if keepdims:
                        out_dims[ax] = MatchDim(name=f"{name}_sum_dim_{ax}", size=1)
                    else:
                        out_dims.pop(ax)
            else:
                # Single axis
                ax = int(axis)
                if keepdims:
                    out_dims[ax] = MatchDim(name=f"{name}_sum_dim_{ax}", size=1)
                    self.node_all_dims[out_dims[ax].name] = out_dims[ax]
                else:
                    out_dims.pop(ax)
        else:
            # Sum over all axes
            if keepdims:
                out_dims = [MatchDim(name=f"{name}_sum_dim_{i}", size=1) for i in range(len(out_dims))]
                for dim in out_dims:
                    self.node_all_dims[dim.name] = dim
            else:
                out_dims = []  # Scalar result
        
        # Update layout for output
        out_layout = inp_tensor.layout
        if len(out_dims) == 4:
            out_layout = "NCHW"
        elif len(out_dims) == 3:
            out_layout = "NCH"
        elif len(out_dims) == 2:
            out_layout = "NC"
        elif len(out_dims) == 1:
            out_layout = "N"
        
        out_tensor = MatchTensor(name=name, dims=out_dims, dtype=inp_tensor.dtype,
                                tensor_type="output", layout=out_layout)
        self.calls_tensors[name] = out_tensor
        op = ops.MatchOpSum(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
            axis=axis,
            keepdims=keepdims,
        )
        self.update_match_node(op=op, call=call, name=name)

    def visit_conv_3d(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        # shapes etc.
        ishape = [int(v) for v in call.args[0].checked_type.shape]
        wshape = [int(v) for v in call.args[1].checked_type.shape]
        oshape = [int(v) for v in call.checked_type.shape]
        odtype = call.checked_type.dtype
        inp_tensor.layout = attrs.data_layout if attrs.data_layout!="" else "NCDHW"
        (i_n,i_n_dim), (i_c,i_c_dim), (i_d,i_d_dim), (i_h,i_h_dim), (i_w,i_w_dim) = self.get_io_from_layout(attrs.data_layout, ishape, inp_tensor.dims)
        w_tensor.layout = attrs.kernel_layout if attrs.kernel_layout!="" else "OIDHW"
        (w_cout,w_cout_dim), (w_cin,w_cin_dim), (w_ksd,w_ksd_dim), (w_ksh,w_ksh_dim), (w_ksw,w_ksw_dim) = self.get_io_from_layout(attrs.kernel_layout, wshape, w_tensor.dims)
        padding = [int(a_p) for a_p in attrs.padding]
        strides = [int(v) for v in attrs.strides]
        dilations = [int(a_d) for a_d in attrs.dilation]
        groups = int(attrs.groups)
        depthwise = groups != 1 and groups==i_c
        (o_n,o_n_dim), (o_c,o_c_dim), (o_d,o_d_dim), (o_h,o_h_dim), (o_w,o_w_dim) = self.get_io_from_layout(
            attrs.out_layout if attrs.out_layout != "" else attrs.data_layout, oshape, [None,None,None,None,None]
        )
        # manage dimensions dependencies
        if strides[0]!=1 or dilations[0]!=1 or w_ksd!=1 or padding[0]!=0:
            o_d_dim = MatchDim(name=name+"_out_d",size=o_d)
            self.node_all_dims[o_d_dim.name] = o_d_dim
            if i_d_dim.name==i_d_dim.original_name:
                i_d_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        (o_d_dim,strides[0]),
                        (w_ksd_dim,dilations[0]),
                        (padding[0],-1)
                    ],
                    size_dependencies=[
                        (o_d_dim,strides[0]),
                        (w_ksd_dim,dilations[0]),
                        (strides[0],-1)
                    ]
                )
            else:
                self.node_all_dims[i_d_dim.name].dependencies = DimDependency(
                    idx_dependencies=[
                        (w_cout_dim,strides[0]),
                        (w_ksd_dim,dilations[0]),
                        (padding[0],-1)
                    ],
                    size_dependencies=[
                        (w_cout_dim,strides[0]),
                        (w_ksd_dim,dilations[0]),
                        (strides[0],-1)
                    ]
                )
        else:
            o_d_dim = i_d_dim
        if strides[1]!=1 or dilations[1]!=1 or w_ksh!=1 or padding[1]!=0:
            o_h_dim = MatchDim(name=name+"_out_h",size=o_h)
            self.node_all_dims[o_h_dim.name] = o_h_dim
            if i_h_dim.name==i_h_dim.original_name:
                i_h_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        (o_h_dim,strides[1]),
                        (w_ksh_dim,dilations[1]),
                        (padding[1],-1)
                    ],
                    size_dependencies=[
                        (o_h_dim,strides[1]),
                        (w_ksh_dim,dilations[1]),
                        (strides[1],-1)
                    ]
                )
            else:
                self.node_all_dims[i_h_dim.name].dependencies = DimDependency(
                    idx_dependencies=[
                        (w_cout_dim,strides[1]),
                        (w_ksh_dim,dilations[1]),
                        (padding[1],-1)
                    ],
                    size_dependencies=[
                        (w_cout_dim,strides[1]),
                        (w_ksh_dim,dilations[1]),
                        (strides[1],-1)
                    ]
                )
        else:
            o_h_dim = i_h_dim
        if strides[2]!=1 or dilations[2]!=1 or w_ksw!=1 or padding[2]!=0:
            o_w_dim = MatchDim(name=name+"_out_w",size=o_w)
            self.node_all_dims[o_w_dim.name] = o_w_dim
            if i_w_dim.name==i_w_dim.original_name:
                i_w_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        (o_w_dim,strides[2]),
                        (w_ksw_dim,dilations[2]),
                        (padding[2],-1)
                    ],
                    size_dependencies=[
                        (o_w_dim,strides[2]),
                        (w_ksw_dim,dilations[2]),
                        (strides[2],-1)
                    ]
                )
            else:
                self.node_all_dims[i_w_dim.name].dependencies = DimDependency(
                    idx_dependencies=[
                        (w_cout_dim,strides[2]),
                        (w_ksw_dim,dilations[2]),
                        (padding[2],-1)
                    ],
                    size_dependencies=[
                        (w_cout_dim,strides[2]),
                        (w_ksw_dim,dilations[2]),
                        (strides[2],-1)
                    ]
                )
        else:
            o_w_dim = i_w_dim
        if not depthwise:
            self.update_all_dim_names_occurrences_with(old_dim_name=w_cin_dim.name,new_dim_name=i_c_dim.name)
        else:
            self.update_all_dim_names_occurrences_with(old_dim_name=i_c_dim.name,new_dim_name=w_cout_dim.name)
        o_tensor = MatchTensor(name=name,dims=self.get_dim_arr_from_layout_and_ncdhw_arr(
            layout=attrs.out_layout if attrs.out_layout != "" else attrs.data_layout,
            ncdhw_arr=[i_n_dim,w_cout_dim,o_d_dim,o_h_dim,o_w_dim]
        ),dtype=np.dtype(odtype),tensor_type="output", layout=attrs.out_layout if attrs.out_layout != "" else attrs.data_layout if attrs.data_layout!="" else inp_tensor.layout)
        self.calls_tensors[name]=o_tensor
        if i_n != o_n:
            raise NotImplementedError(
                f"[RELAY PARSER] Input batch size is {i_n}, while output batch size is {o_n}"
            )
        if not depthwise and groups>1:
            raise NotImplementedError(
                f"[PARSER] Grouped convolutions which are not completely depthwise aren't supported yet, groups set to {groups}"
            )
        op = ops.MatchOpConv3D(
            out_arr=[o_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            padding= padding,
            strides= strides,
            dilation= dilations,
            groups= groups,
            kernel_size=(w_ksd,w_ksh,w_ksw),
            depthwise= depthwise,
            data_layout= attrs.data_layout,
            kernel_layout= attrs.kernel_layout,
            out_dtype= np.dtype(attrs.out_dtype) if attrs.out_dtype!="" else np.dtype(odtype),
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_conv_2d_transpose(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        # shapes etc.
        ishape = [int(v) for v in call.args[0].checked_type.shape]
        wshape = [int(v) for v in call.args[1].checked_type.shape]
        oshape = [int(v) for v in call.checked_type.shape]
        odtype = call.checked_type.dtype
        inp_tensor.layout = attrs.data_layout if attrs.data_layout!="" else "NCHW"
        (i_n,i_n_dim), (i_c,i_c_dim), (i_h,i_h_dim), (i_w,i_w_dim) = self.get_io_from_layout(attrs.data_layout, ishape, inp_tensor.dims)
        w_tensor.layout = attrs.kernel_layout if attrs.kernel_layout!="" else "OIHW"
        (w_cout,w_cout_dim), (w_cin,w_cin_dim), (w_ksh,w_ksh_dim), (w_ksw,w_ksw_dim) = self.get_io_from_layout(attrs.kernel_layout, wshape, w_tensor.dims)
        padding = [int(a_p) for a_p in attrs.padding]
        output_padding = [int(a_p) for a_p in attrs.output_padding] if hasattr(attrs,"output_padding") else [0,0]
        strides = [int(v) for v in attrs.strides]
        dilations = [int(a_d) for a_d in attrs.dilation]
        groups = int(attrs.groups)
        depthwise = groups != 1 and groups==i_c
        (o_n,o_n_dim), (o_c,o_c_dim), (o_h,o_h_dim), (o_w,o_w_dim) = self.get_io_from_layout(
            attrs.out_layout if attrs.out_layout != "" else attrs.data_layout, oshape, [None,None,None,None]
        )
        CONV2D_TRANSPOSE_INPUT_IS_DEPENDENT = True
        # manage dimensions dependencies
        if strides[0]!=1 or dilations[0]!=1 or w_ksh!=1 or padding[0]!=0 or output_padding[0]!=0:
            o_h_dim = MatchDim(name=name+"_out_h",size=o_h)
            self.node_all_dims[o_h_dim.name] = o_h_dim
            if CONV2D_TRANSPOSE_INPUT_IS_DEPENDENT:
                i_h_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        # h_in = (h_out × 1/S_h) + (P_top × 1/S_h) + (-D_h × (K_h - 1) × 1/S_h) + (-OP_h × 1/S_h)
                        (o_h_dim, 1/strides[0]),  # + h_out * (1 / S_h)
                        (padding[0], -1/strides[0]),  # + P_top * (1 / S_h)
                        (w_ksh_dim, dilations[0]/strides[0]),  # + (-D_h * K_h * (1 / S_h))
                        # (dilations[0], 1/strides[0]),  # + (-D_h * - 1 * (1 / S_h))
                        # (output_padding[0], -1/strides[0]),  # + (-OP_h * (1 / S_h))
                    ],
                    size_dependencies=[
                        # H_in = (H_out * (1 / S_h)) + (-1 * (1 / S_h)) + (-D_h * (K_h - 1) * (1 / S_h)) + (-OP_h * (1 / S_h)) + (P_top * (1 / S_h)) + (P_bottom * (1 / S_h)) + 1
                        (1, 1),  # + 1
                        (o_h_dim, 1/strides[0]),  # + H_out * (1 / S_h)
                        (1, -1/strides[0]),  # + (-1 * (1 / S_h))
                        (w_ksh_dim, dilations[0]/strides[0]),  # + (-D_h * K_h * (1 / S_h))
                        (dilations[0], -1/strides[0]),  # + (-D_h * - 1 * (1 / S_h))
                        # (padding[0], 1/strides[0]),  # + (P_top * (1 / S_h))
                        # (padding[2], 1/strides[0]),  # + (P_bottom * (1 / S_h))
                        (output_padding[0], 1/strides[0]),  # + (-OP_h * (1 / S_h))
                    ]
                )
            else:
                o_h_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        # h_out = h_in * S_h - P_top
                        (i_h_dim, strides[0]),  # + H_in * S_h
                        (padding[0], -1),  # - P_top
                    ],
                    size_dependencies=[
                        # H_out = (H_in - 1) * S_h - P_top - P_bottom + D_h * (K_h - 1) + 1 + OP_h
                        (i_h_dim, strides[0]),  # + H_in * S_h
                        (1, -strides[0]),  # + (-1 * S_h)
                        (padding[0], -1),  # - P_top
                        (padding[2], -1),  # - P_bottom
                        (w_ksh_dim, dilations[0]),  # + D_h * K_h
                        (dilations[0], -1),  # + D_h * - 1
                        (output_padding[0], 1),  # + OP_h
                    ]
                )

        else:
            o_h_dim = i_h_dim
        if strides[1]!=1 or dilations[1]!=1 or w_ksw!=1 or padding[1]!=0 or output_padding[1]!=0:
            o_w_dim = MatchDim(name=name+"_out_w",size=o_w)
            self.node_all_dims[o_w_dim.name] = o_w_dim
            if CONV2D_TRANSPOSE_INPUT_IS_DEPENDENT:
                i_w_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        # w_in = (w_out × 1/S_w) + (P_left × 1/S_w) + (-D_w × (K_w - 1) × 1/S_w) + (-OP_w × 1/S_w)
                        (o_w_dim, 1/strides[1]),  # + w_out * (1 / S_w)
                        (padding[1], -1/strides[1]),  # + P_left * (1 / S_w)
                        (w_ksw_dim, dilations[1]/strides[1]),  # + (-D_w * K_w * (1 / S_w))
                        # (dilations[1], 1/strides[1]),  # + (-D_w * - 1 * (1 / S_w))
                        # (output_padding[1], -1/strides[1]),  # + (-OP_w * (1 / S_w))
                    ],
                    size_dependencies=[
                        # W_in = (W_out * (1 / S_w)) + (-1 * (1 / S_w)) + (-D_w * (K_w - 1) * (1 / S_w)) + (-OP_w * (1 / S_w)) + (P_left * (1 / S_w)) + (P_right * (1 / S_w)) + 1
                        (1, 1),  # + 1
                        (o_w_dim, 1/strides[1]),  # + W_out * (1 / S_w)
                        (1, -1/strides[1]),  # + (-1 * (1 / S_w))
                        (w_ksw_dim, dilations[1]/strides[1]),  # + (-D_w * K_w * (1 / S_w))
                        (dilations[1], -1/strides[1]),  # + (-D_w * - 1 * (1 / S_w))
                        # (padding[1], 1/strides[1]),  # + (P_left * (1 / S_w))
                        # (padding[3], 1/strides[1]),  # + (P_right * (1 / S_w))
                        (output_padding[1], 1/strides[1]),  # + (-OP_w * (1 / S_w))
                    ]
                )
            else:
                o_w_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        # w_out = w_in * S_w - P_left
                        (i_w_dim, strides[1]),  # + W_in * S_w
                        (padding[1], -1),  # - P_left
                    ],
                    size_dependencies=[
                        # W_out = (W_in - 1) * S_w - P_left - P_right + D_w * (K_w - 1) + 1 + OP_w
                        (i_w_dim, strides[1]),  # + W_in * S_w
                        (1, -strides[1]),  # + (-1 * S_w)
                        (padding[1], -1),  # - P_left
                        (padding[3], -1),  # - P_right
                        (w_ksw_dim, dilations[1]),  # + D_w * K_w
                        (dilations[1], -1),  # + D_w * - 1
                        (output_padding[1], 1),  # + OP_w
                    ]
                )
        else:
            o_w_dim = i_w_dim
        self.update_all_dim_names_occurrences_with(old_dim_name=w_cin_dim.name,new_dim_name=i_c_dim.name)
        o_tensor = MatchTensor(name=name,dims=self.get_dim_arr_from_layout_and_nchw_arr(
            layout=attrs.out_layout if attrs.out_layout != "" else attrs.data_layout,
            nchw_arr=[i_n_dim,w_cout_dim if not depthwise else i_c_dim,o_h_dim,o_w_dim]
        ),dtype=np.dtype(odtype),tensor_type="output", layout=attrs.out_layout if attrs.out_layout != "" else attrs.data_layout if attrs.data_layout!="" else inp_tensor.layout)
        self.calls_tensors[name]=o_tensor
        if i_n != o_n:
            raise NotImplementedError(
                f"[RELAY PARSER] Input batch size is {i_n}, while output batch size is {o_n}"
            )
        if not depthwise and groups>1:
            raise NotImplementedError(
                f"[PARSER] Grouped convolutions which are not completely depthwise aren't supported yet, groups set to {groups}"
            )
        op = ops.MatchOpConv2DTranspose(
            out_arr=[o_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            padding= padding,
            output_padding=output_padding,
            strides= strides,
            dilation= dilations,
            groups= groups,
            kernel_size=(w_ksh,w_ksw),
            depthwise= depthwise,
            data_layout= attrs.data_layout,
            kernel_layout= attrs.kernel_layout,
            out_dtype= np.dtype(attrs.out_dtype) if attrs.out_dtype!="" else np.dtype(odtype),
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_conv_2d(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        # shapes etc.
        ishape = [int(v) for v in call.args[0].checked_type.shape]
        wshape = [int(v) for v in call.args[1].checked_type.shape]
        oshape = [int(v) for v in call.checked_type.shape]
        odtype = call.checked_type.dtype
        inp_tensor.layout = attrs.data_layout if attrs.data_layout!="" else "NCHW"
        (i_n,i_n_dim), (i_c,i_c_dim), (i_h,i_h_dim), (i_w,i_w_dim) = self.get_io_from_layout(attrs.data_layout, ishape, inp_tensor.dims)
        w_tensor.layout = attrs.kernel_layout if attrs.kernel_layout!="" else "OIHW"
        (w_cout,w_cout_dim), (w_cin,w_cin_dim), (w_ksh,w_ksh_dim), (w_ksw,w_ksw_dim) = self.get_io_from_layout(attrs.kernel_layout, wshape, w_tensor.dims)
        padding = [int(a_p) for a_p in attrs.padding]
        strides = [int(v) for v in attrs.strides]
        dilations = [int(a_d) for a_d in attrs.dilation]
        groups = int(attrs.groups)
        depthwise = groups != 1 and groups==i_c
        (o_n,o_n_dim), (o_c,o_c_dim), (o_h,o_h_dim), (o_w,o_w_dim) = self.get_io_from_layout(
            attrs.out_layout if attrs.out_layout != "" else attrs.data_layout, oshape, [None,None,None,None]
        )
        # manage dimensions dependencies
        if strides[0]!=1 or dilations[0]!=1 or w_ksh!=1 or padding[0]!=0:
            o_h_dim = MatchDim(name=name+"_out_h",size=o_h)
            self.node_all_dims[o_h_dim.name] = o_h_dim
            if i_h_dim.name==i_h_dim.original_name:
                i_h_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        (o_h_dim,strides[0]),
                        (w_ksh_dim,dilations[0]),
                        (padding[0],-1)
                    ],
                    size_dependencies=[
                        (o_h_dim,strides[0]),
                        (w_ksh_dim,dilations[0]),
                        (strides[0],-1)
                    ]  
                )
            else:
                self.node_all_dims[i_h_dim.name].dependencies = DimDependency(
                    idx_dependencies=[
                        (w_cout_dim,strides[0]),
                        (w_ksh_dim,dilations[0]),
                        (padding[0],-1)
                    ],
                    size_dependencies=[
                        (w_cout_dim,strides[0]),
                        (w_ksh_dim,dilations[0]),
                        (strides[0],-1)
                    ]
                )
        else:
            o_h_dim = i_h_dim
        if strides[1]!=1 or dilations[1]!=1 or w_ksw!=1 or padding[1]!=0:
            o_w_dim = MatchDim(name=name+"_out_w",size=o_w)
            self.node_all_dims[o_w_dim.name] = o_w_dim
            if i_w_dim.name==i_w_dim.original_name:
                i_w_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        (o_w_dim,strides[1]),
                        (w_ksw_dim,dilations[1]),
                        (padding[1],-1)
                    ],
                    size_dependencies=[
                        (o_w_dim,strides[1]),
                        (w_ksw_dim,dilations[1]),
                        (strides[1],-1)
                    ]
                )
            else:
                self.node_all_dims[i_w_dim.name].dependencies = DimDependency(
                    idx_dependencies=[
                        (w_cout_dim,strides[1]),
                        (w_ksw_dim,dilations[1]),
                        (padding[1],-1)
                    ],
                    size_dependencies=[
                        (w_cout_dim,strides[1]),
                        (w_ksw_dim,dilations[1]),
                        (strides[1],-1)
                    ]
                )
        else:
            o_w_dim = i_w_dim
        if not depthwise:
            self.update_all_dim_names_occurrences_with(old_dim_name=w_cin_dim.name,new_dim_name=i_c_dim.name)
        else:
            self.update_all_dim_names_occurrences_with(old_dim_name=i_c_dim.name,new_dim_name=w_cout_dim.name)
        o_tensor = MatchTensor(name=name,dims=self.get_dim_arr_from_layout_and_nchw_arr(
            layout=attrs.out_layout if attrs.out_layout != "" else attrs.data_layout,
            nchw_arr=[i_n_dim,w_cout_dim,o_h_dim,o_w_dim]
        ),dtype=np.dtype(odtype),tensor_type="output", layout=attrs.out_layout if attrs.out_layout != "" else attrs.data_layout if attrs.data_layout!="" else inp_tensor.layout)
        self.calls_tensors[name]=o_tensor
        if i_n != o_n:
            raise NotImplementedError(
                f"[RELAY PARSER] Input batch size is {i_n}, while output batch size is {o_n}"
            )
        if not depthwise and groups>1:
            raise NotImplementedError(
                f"[PARSER] Grouped convolutions which are not completely depthwise aren't supported yet, groups set to {groups}"
            )
        op = ops.MatchOpConv2D(
            out_arr=[o_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            padding= padding,
            strides= strides,
            dilation= dilations,
            groups= groups,
            kernel_size=(w_ksh,w_ksw),
            depthwise= depthwise,
            data_layout= attrs.data_layout,
            kernel_layout= attrs.kernel_layout,
            out_dtype= np.dtype(attrs.out_dtype) if attrs.out_dtype!="" else np.dtype(odtype),
        )
        self.update_match_node(op=op,call=call,name=name)

    def visit_conv_1d(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call,call.args[1],1)
        self.update_if_intermediate_tensor(tensor=w_tensor, name=w_name)
        inp_name, inp_tensor, inp_type, w_name, w_tensor, weights_type = self.rearrange_if_const_first(
            inp_name=inp_name, inp_tensor=inp_tensor, inp_type=inp_type,
            w_name=w_name, w_tensor=w_tensor, weights_type=weights_type
        )
        # shapes etc.
        ishape = [int(v) for v in call.args[0].checked_type.shape]
        wshape = [int(v) for v in call.args[1].checked_type.shape]
        oshape = [int(v) for v in call.checked_type.shape]
        odtype = call.checked_type.dtype
        inp_tensor.layout = attrs.data_layout if attrs.data_layout!="" else "NCH"
        (i_n,i_n_dim), (i_c,i_c_dim), (i_spatial,i_spatial_dim)= self.get_io_from_layout(attrs.data_layout, ishape, inp_tensor.dims)
        inp_tensor.layout = attrs.kernel_layout if attrs.kernel_layout!="" else "OIH"
        (w_cout,w_cout_dim), (w_cin,w_cin_dim), (w_kernel,w_kernel_dim) = self.get_io_from_layout(attrs.data_layout, wshape, w_tensor.dims)
        padding = [int(a_p) for a_p in attrs.padding]
        strides = [int(v) for v in attrs.strides]
        dilations = [int(a_d) for a_d in attrs.dilation]
        groups = int(attrs.groups)
        depthwise = groups != 1 and groups==i_c
        (o_n,o_n_dim), (o_c,o_c_dim), (o_spatial,o_spatial_dim) = self.get_io_from_layout(
            attrs.out_layout if attrs.out_layout != "" else attrs.data_layout, oshape, [None,None,None,None]
        )
        # manage dimensions dependencies
        if strides[0]!=1 or dilations[0]!=1 or padding[0]!=0 or w_kernel!=1:
            o_spatial_dim = MatchDim(name=name+"_out_spatial",size=o_spatial)
            self.node_all_dims[o_spatial_dim.name] = o_spatial_dim
            if i_spatial_dim.name==i_spatial_dim.original_name:
                i_spatial_dim.dim_dependency = DimDependency(
                    idx_dependencies=[
                        (o_spatial_dim,strides[0]),
                        (w_kernel_dim,dilations[0]),
                        (padding[0],-1)
                    ],
                    size_dependencies=[
                        (o_spatial_dim,strides[0]),
                        (w_kernel_dim,dilations[0]),
                        (strides[0],-1)
                    ]
                )
            else:
                self.node_all_dims[i_spatial_dim.name].dependencies = DimDependency(
                    idx_dependencies=[
                        (w_cout_dim,strides[0]),
                        (w_kernel_dim,dilations[0]),
                        (padding[0],-1)
                    ],
                    size_dependencies=[
                        (w_cout_dim,strides[0]),
                        (w_kernel_dim,dilations[0]),
                        (strides[0],-1)
                    ]
                )
        else:
            o_spatial_dim = i_spatial_dim
        if not depthwise:
            self.update_all_dim_names_occurrences_with(old_dim_name=w_cin_dim.name,new_dim_name=i_c_dim.name)
        else:
            self.update_all_dim_names_occurrences_with(old_dim_name=i_c_dim.name,new_dim_name=w_cout_dim.name)
        o_tensor = MatchTensor(name=name,dims=self.get_dim_arr_from_layout_and_nchw_arr(
            layout=attrs.out_layout if attrs.out_layout != "" else attrs.data_layout,
            nchw_arr=[i_n_dim,w_cout_dim,o_spatial_dim]
        ),dtype=np.dtype(odtype),tensor_type="output",layout=inp_tensor.layout)
        self.calls_tensors[name]=o_tensor
        if i_n != o_n:
            raise NotImplementedError(
                f"Input batch size is {i_n}, while output batch size is {o_n}"
            )
        if not depthwise and groups>1:
            raise NotImplementedError(
                f"Grouped convolutions which are not completely depthwise aren't supported yet, groups set to {groups}"
            )
        op = ops.MatchOpConv1D(
            out_arr=[o_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            padding= padding,
            strides= strides,
            dilation= dilations,
            groups= groups,
            kernel_size=(w_kernel),
            depthwise= depthwise,
            data_layout= attrs.data_layout,
            kernel_layout= attrs.kernel_layout,
            out_dtype= np.dtype(attrs.out_dtype) if attrs.out_dtype!="" else np.dtype(odtype),
        )
        self.update_match_node(op=op,call=call,name=name)
        
        
    def visit_batch_matmul(self, call, attrs, name):
        x1_shape = [int(v) for v in call.args[0].checked_type.shape]
        x2_shape = [int(v) for v in call.args[1].checked_type.shape]

        dim_b = x1_shape[0]
        dim_m = x1_shape[1]
        dim_n = x2_shape[2]
        dim_k = x1_shape[2]
        
        if x2_shape[1] != dim_k or x1_shape[0] != x2_shape[0]:
            raise NotImplementedError(f"[RELAY PARSER]: batch_matmul shapes mismatch {x1_shape} and {x2_shape}.")
        
        out_dtype = call.checked_type.dtype
        
        x1_name, x1_tensor, x1_type = self.get_name_and_tensor_of_arg(call, call.args[0], 0)
        self.update_if_intermediate_tensor(tensor=x1_tensor, name=x1_name)
        
        x2_name, x2_tensor, x2_type = self.get_name_and_tensor_of_arg(call, call.args[1], 1)
        self.update_if_intermediate_tensor(tensor=x2_tensor, name=x2_name)
        
        x1_tensor.layout = "BMN"
        x2_tensor.layout = "BNK" if not attrs.transpose_b else "BKN"
        
        # update_all_dim_names_occurrences_with 
        
        out_dims = [*x1_tensor.dims[:-1], x2_tensor.dims[-1]]
        out_tensor = MatchTensor(name=name,dims=out_dims,dtype=np.dtype(out_dtype),tensor_type="output", layout=x1_tensor.layout)
        self.calls_tensors[name]=out_tensor
        op = ops.MatchOpBatchMatMul(
            out_arr=[out_tensor],
            var_arr=[x1_tensor, x2_tensor],
            const_arr=[],
            dim_b=dim_b,
            dim_m=dim_m,
            dim_n=dim_n,
            dim_k=dim_k,
            out_dtype=np.dtype(out_dtype),
        )
        self.update_match_node(op=op,call=call,name=name)