# TVM imports
import numpy as np
from match import ops
from match.dim.dim import MatchDim, DimDependency
from match.parser.parser import MatchParser
from match.tensor.tensor import MatchTensor

import tvm
import tvm.relay

class MatchRelayParser(MatchParser):
    def __init__(self, node, args_list = ..., exec_module = None, pattern_name = "", partitioned = False, pattern_inst=None, match_node = None):
        super().__init__(node, args_list, exec_module, pattern_name, partitioned, pattern_inst, match_node)
        self.visit_router = {
            "nn.conv2d": self.visit_conv_2d,
            # "cast": self.visit_cast,
            # "right_shift": self.visit_right_shift,
            # "clip": self.visit_clip,
            "nn.bias_add": self.visit_bias_add,
            "nn.dense": self.visit_dense,
            # "add": self.visit_add,
            # "multiply": self.visit_multiply,
            "nn.relu": self.visit_relu,
        }
    
    def visit_relu(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call.args[0],0)
        if inp_name in self.name_to_calls:
            inp_tensor.tensor_type = "intermediate"
        out_tensor = MatchTensor(name=name,dims=inp_tensor.dims,dtype=inp_tensor.dtype,tensor_type="output")
        self.calls_tensors[name]=out_tensor
        op = ops.MatchOpReLU(
            var_arr=[inp_tensor],
            out_arr=[out_tensor],
        )
        self.update_match_node(op=op,call=call,name=name)

    # def visit_cast(self, call, attrs, name):
    #     otype = [int(v) for v in call.checked_type.shape]
    #     odtype = call.checked_type.dtype
    #     otype_dim = [Dim(name=name+f"_dim_{idx}",size=otype[idx]) for idx in range(len(otype)) if otype[idx]!=1]
    #     out = MatchOutputTensor(name,dims=otype_dim,dtype=odtype)
    #     op = ops.Cast(
    #         outs=[out],
    #         prec= self.get_bits(attrs.dtype),
    #         type= self.get_type(attrs.dtype),
    #     )
    #     self.match_node.ops[name]=op
    #     self.match_node.calls[name]=call

    # def visit_right_shift(self, call, attrs, name):
    #     # nothing to do actually, right shift has no attrs and the arg is already saved before
    #     otype = [int(v) for v in call.checked_type.shape]
    #     odtype = call.checked_type.dtype
    #     otype_dim = [Dim(name=name+f"_dim_{idx}",size=otype[idx]) for idx in range(len(otype)) if otype[idx]!=1]
    #     out = MatchOutputTensor(name,dims=otype_dim,dtype=odtype)
    #     op = ops.RightShift(
    #         outs=[out],
    #         prec= self.get_bits(call.checked_type.dtype),
    #         type= self.get_type(call.checked_type.dtype),
    #     )
    #     self.match_node.ops[name]=op
    #     self.match_node.calls[name]=call

    # def visit_clip(self, call, attrs, name):
    #     otype = [int(v) for v in call.checked_type.shape]
    #     odtype = call.checked_type.dtype
    #     otype_dim = [Dim(name=name+f"_dim_{idx}",size=otype[idx]) for idx in range(len(otype)) if otype[idx]!=1]
    #     out = MatchOutputTensor(name,dims=otype_dim,dtype=odtype)
    #     op = ops.Clip(
    #         outs=[out],
    #         min=int(attrs.a_min),
    #         max=int(attrs.a_max),
    #     )
    #     self.match_node.ops[name]=op
    #     self.match_node.calls[name]=call

    def visit_bias_add(self, call, attrs, name):
        axis = int(attrs.axis) if hasattr(attrs,"axis") else 0
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call.args[0],0)
        if inp_name in self.name_to_calls:
            inp_tensor.tensor_type = "intermediate"
        out_tensor = MatchTensor(name=name,dims=inp_tensor.dims,dtype=inp_tensor.dtype,tensor_type="output")
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call.args[1],1)
        if w_name in self.name_to_calls:
            w_tensor.tensor_type = "intermediate"
        for w_dim in w_tensor.dims:
            if w_dim.size!=1:
                self.update_all_dim_names_occurrences_with(old_dim_name=w_dim.name,new_dim_name=inp_tensor.dims[-1].name if axis>=len(inp_tensor.dims) or axis<0 else inp_tensor.dims[axis].name)
        self.calls_tensors[name]=out_tensor
        op=ops.MatchOpBiasAdd(
            out_arr=[out_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            axis=axis,
        )
        self.update_match_node(op=op,call=call,name=name)

    # def visit_multiply(self,call,atts,name):
    #     itype = [int(v) for v in call.args[0].args[0].checked_type.shape]
    #     iprec = call.args[0].args[0].checked_type.dtype
    #     wtype = [int(v) for v in call.args[1].args[0].checked_type.shape]
    #     wprec = call.args[1].args[0].checked_type.dtype
    #     otype = [int(v) for v in call.checked_type.shape]
    #     oprec = call.checked_type.dtype
    #     op=ops.Multiply(
    #         o_type=otype,
    #         o_prec=oprec,
    #         i_type=itype,
    #         i_prec=iprec,
    #         w_type=wtype,
    #         w_prec=wprec,
    #     )
    #     self.match_node.ops[name]=op
    #     self.match_node.calls[name]=call

    def visit_dense(self, call, attrs, name):
        ishape = [int(v) for v in call.args[0].checked_type.shape]
        wshape = [int(v) for v in call.args[1].checked_type.shape]
        inp_features = wshape[1]
        out_features = wshape[0]
        odtype = call.checked_type.dtype
        if ishape[1] != wshape[1]:
            raise NotImplementedError(f"[PARSER]: The weights shape in the dense operation are not correct")
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call.args[0],0,name)
        if inp_name in self.name_to_calls:
            inp_tensor.tensor_type = "intermediate"
        # well consider the case where the multiplied dimension is the last one
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call.args[1],1,name)
        if w_name in self.name_to_calls:
            w_tensor.tensor_type = "intermediate"
        self.update_all_dim_names_occurrences_with(old_dim_name=w_tensor.dims[-1].name,new_dim_name=inp_tensor.dims[-1].name)
        out_dims = inp_tensor.dims[:-1]+[w_tensor.dims[0]]
        out_tensor = MatchTensor(name=name,dims=out_dims,dtype=np.dtype(odtype),tensor_type="output")
        self.calls_tensors[name]=out_tensor
        op = ops.MatchOpDense(
            out_arr=[out_tensor],
            var_arr=[inp_tensor],
            const_arr=[w_tensor],
            inp_features=inp_features,
            out_features=out_features,
        )
        self.update_match_node(op=op,call=call,name=name)

    # def visit_add(self, call, attrs, name):
    #     axis = int(attrs.axis) if hasattr(attrs,"axis") else 0
    #     otype = [int(v) for v in call.checked_type.shape]
    #     odtype = call.checked_type.dtype
    #     otype_dim = [Dim(name=name+f"_dim_{idx}",size=otype[idx]) for idx in range(len(otype)) if otype[idx]!=1]
    #     out = MatchOutputTensor(name,dims=otype_dim,dtype=odtype)
    #     vars = []
    #     consts = []
    #     for arg in call.args:
    #         type = [int(v) for v in arg.checked_type.shape]
    #         dtype = arg.checked_type.dtype
    #         if isinstance(arg,tvm.relay.Var):
    #             vars.append(MatchVarTensor(name=arg.name_hint,dims=otype_dim,dtype=dtype))
    #         elif isinstance(arg,tvm.relay.Call):
    #             vars.append(MatchVarTensor(name=self.calls_to_name[arg],dims=otype_dim,dtype=dtype))
    #         else:
    #             consts.append(MatchConstTensor(name=call.op.name+ f".param.{int(name.split("_")[-1]) if name.split("_")[-1].isnumeric() else 0}"),
    #                           dims=[d for d in otype_dim if d.name==name+f"_dim_{axis}"],dtype=dtype)
    #     op = ops.MatchOpAdd(
    #         outs=[out],
    #         vars=vars,
    #         consts=consts,
    #         **attrs
    #     )
    #     self.match_node.ops[name]=op
    #     self.match_node.calls[name]=call

    def visit_conv_2d(self, call, attrs, name):
        inp_name, inp_tensor, inp_type = self.get_name_and_tensor_of_arg(call.args[0],0)
        if inp_name in self.name_to_calls:
            inp_tensor.tensor_type = "intermediate"
        w_name, w_tensor, weights_type = self.get_name_and_tensor_of_arg(call.args[1],1)
        if w_name in self.name_to_calls:
            w_tensor.tensor_type = "intermediate"
        # shapes etc.
        ishape = [int(v) for v in call.args[0].checked_type.shape]
        wshape = [int(v) for v in call.args[1].checked_type.shape]
        oshape = [int(v) for v in call.checked_type.shape]
        odtype = call.checked_type.dtype
        (i_n,i_n_dim), (i_c,i_c_dim), (i_h,i_h_dim), (i_w,i_w_dim) = self.get_io_from_layout(attrs.data_layout, ishape, inp_tensor.dims)
        (w_cout,w_cout_dim), (w_cin,w_cin_dim), (w_ksh,w_ksh_dim), (w_ksw,w_ksw_dim) = self.get_io_from_layout(attrs.data_layout, wshape, w_tensor.dims)
        padding = [int(a_p) for a_p in attrs.padding]
        strides = [int(v) for v in attrs.strides]
        dilations = [int(a_d) for a_d in attrs.dilation]
        groups = int(attrs.groups)
        depthwise = groups != 1 and groups==i_c
        (o_n,o_n_dim), (o_c,o_c_dim), (o_h,o_h_dim), (o_w,o_w_dim) = self.get_io_from_layout(
            attrs.out_layout if attrs.out_layout != "" else attrs.data_layout, oshape, [None,None,None,None]
        )
        # manage dimensions dependencies
        if strides[0]!=1 or dilations[0]!=1:
            o_h_dim = MatchDim(name=name+"_out_h",size=o_h)
            self.node_all_dims[o_h_dim.name] = o_h_dim
            if i_h_dim.name==i_h_dim.original_name:
                i_h_dim.dim_dependency = DimDependency(dependencies={o_h_dim:strides[0],w_ksh_dim:dilations[0],padding[0]:-1})
            else:
                self.node_all_dims[i_h_dim.name].dependencies = DimDependency(dependencies={w_cout_dim:strides[0],w_ksh_dim:dilations[0],padding[0]:-1})
        else:
            o_h_dim = i_h_dim
        if strides[1]!=1 or dilations[1]!=1:
            o_w_dim = MatchDim(name=name+"_out_w",size=o_w)
            self.node_all_dims[o_w_dim.name] = o_w_dim
            if i_w_dim.name==i_w_dim.original_name:
                i_w_dim.dim_dependency = DimDependency(dependencies={o_w_dim:strides[1],w_ksw_dim:dilations[1],padding[1]:-1})
            else:
                self.node_all_dims[i_w_dim.name].dependencies = DimDependency(dependencies={w_cout_dim:strides[1],w_ksw_dim:dilations[1],padding[1]:-1})
        else:
            o_w_dim = i_w_dim
        if not depthwise:
            self.update_all_dim_names_occurrences_with(old_dim_name=w_cin_dim.name,new_dim_name=i_c_dim.name)
        
        o_tensor = MatchTensor(name=name,dims=self.get_dim_arr_from_layout_and_nchw_arr(
            layout=attrs.out_layout if attrs.out_layout != "" else attrs.data_layout,
            nchw_arr=[i_n_dim,w_cout_dim,o_h_dim,o_w_dim]
        ),dtype=np.dtype(odtype),tensor_type="output")
        self.calls_tensors[name]=o_tensor
        if i_n != o_n:
            raise NotImplementedError(
                f"Input batch size is {i_n}, while output batch size is {o_n}"
            )
        if not depthwise and groups>1:
            raise NotImplementedError(
                f"Grouped convolutions which are not completely depthwise aren't supported yet, groups set to {groups}"
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
            out_dtype= np.dtype(attrs.out_dtype),
        )
        self.update_match_node(op=op,call=call,name=name)