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
            "cast": self.visit_cast,
            "right_shift": self.visit_right_shift,
            "clip": self.visit_clip,
            "nn.bias_add": self.visit_bias_add,
            "nn.dense": self.visit_dense,
            "add": self.visit_add,
            "multiply": self.visit_multiply,
            "nn.relu": self.visit_relu,
        }
    
    def visit_relu(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call,call.args[0],0)
        self.update_if_intermediate_tensor(tensor=inp_tensor, name=inp_name)
        if inp_tensor.layout=="":
            if len(inp_tensor.dims)==4:
                inp_tensor.layout = "NCHW"
            elif len(inp_tensor.dims)==3:
                inp_tensor.layout = "NCH"
            elif len(inp_tensor.dims)==2:
                inp_tensor.layout = "NC"
            elif len(inp_tensor.dims)==1:
                inp_tensor.layout = "N"
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
        if inp_tensor.layout=="":
            if len(inp_tensor.dims)==4:
                inp_tensor.layout = "NCHW"
            elif len(inp_tensor.dims)==3:
                inp_tensor.layout = "NCH"
            elif len(inp_tensor.dims)==2:
                inp_tensor.layout = "NC"
            elif len(inp_tensor.dims)==1:
                inp_tensor.layout = "N"
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
        if inp_tensor.layout=="":
            if len(inp_tensor.dims)==4:
                inp_tensor.layout = "NCHW"
            elif len(inp_tensor.dims)==3:
                inp_tensor.layout = "NCH"
            elif len(inp_tensor.dims)==2:
                inp_tensor.layout = "NC"
            elif len(inp_tensor.dims)==1:
                inp_tensor.layout = "N"
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
                    idx_dependencies={o_h_dim:strides[0],w_ksh_dim:dilations[0],padding[0]:-1},
                    size_dependencies={o_h_dim:strides[0],w_ksh_dim:dilations[0],strides[0]:-1}    
                )
            else:
                self.node_all_dims[i_h_dim.name].dependencies = DimDependency(
                    idx_dependencies={w_cout_dim:strides[0],w_ksh_dim:dilations[0],padding[0]:-1},
                    size_dependencies={w_cout_dim:strides[0],w_ksh_dim:dilations[0],strides[0]:-1}
                )
        else:
            o_h_dim = i_h_dim
        if strides[1]!=1 or dilations[1]!=1 or w_ksw!=1 or padding[1]!=0:
            o_w_dim = MatchDim(name=name+"_out_w",size=o_w)
            self.node_all_dims[o_w_dim.name] = o_w_dim
            if i_w_dim.name==i_w_dim.original_name:
                i_w_dim.dim_dependency = DimDependency(
                    idx_dependencies={o_w_dim:strides[1],w_ksw_dim:dilations[1],padding[1]:-1},
                    size_dependencies={o_w_dim:strides[1],w_ksw_dim:dilations[1],strides[1]:-1}
                )
            else:
                self.node_all_dims[i_w_dim.name].dependencies = DimDependency(
                    idx_dependencies={w_cout_dim:strides[1],w_ksw_dim:dilations[1],padding[1]:-1},
                    size_dependencies={w_cout_dim:strides[1],w_ksw_dim:dilations[1],strides[1]:-1}    
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
                    idx_dependencies={o_spatial_dim:strides[0],w_kernel_dim:dilations[0],padding[0]:-1},
                    size_dependencies={o_spatial_dim:strides[0],w_kernel_dim:dilations[0],strides[0]:-1}    
                )
            else:
                self.node_all_dims[i_spatial_dim.name].dependencies = DimDependency(
                    idx_dependencies={w_cout_dim:strides[0],w_kernel_dim:dilations[0],padding[0]:-1},
                    size_dependencies={w_cout_dim:strides[0],w_kernel_dim:dilations[0],strides[0]:-1}    
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