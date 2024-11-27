# TVM imports
import tvm
from tvm.relay.transform import InferType
# utils imports
from math import ceil
from collections import OrderedDict
import copy
from typing import Any, Dict,List,Type

from match.target.exec_module import ExecModule
from match.codegen.layer_data import LayerData

def get_depth_arr_pattern(pattern_inst_):
    if isinstance(pattern_inst_,tvm.relay.dataflow_pattern.AttrPattern):
        pattern_inst_=pattern_inst_.pattern
    if isinstance(pattern_inst_,tvm.relay.dataflow_pattern.AltPattern):
        return [get_depth_arr_pattern(pattern_inst_=pattern_inst_.left),get_depth_arr_pattern(pattern_inst_=pattern_inst_.right)]
    elif isinstance(pattern_inst_,tvm.relay.dataflow_pattern.CallPattern):
        return [(pattern_inst_.op.expr.name,get_depth_arr_pattern(arg_)) for arg_ in pattern_inst_.args]
    else:
        return 0

class WorkloadParser:
    def __init__(self, node:tvm.ir.IRModule, args_list:List=[],exec_module:ExecModule=None, pattern_name:str="", partitioned:bool=False,pattern_inst=None):
        self.exec_module=exec_module
        self.args_list=args_list
        self.pattern_name=pattern_name
        self.node = node
        self.visit_router = {
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
        self.supported_operators = tuple(self.visit_router.keys())
        self.index = 0
        self.callsindexes = dict()
        self.calllist = []
        self.var_and_consts = dict()
        self.energy = 0
        self.latency = 0
        self.layer_data=LayerData()
        self.pattern_inst=pattern_inst
        #self.depth_limits=get_depth_arr_pattern(pattern_inst_=pattern_inst)
        if not partitioned:
            mod = tvm.ir.IRModule()
            self.node=InferType()(mod.from_expr(pattern_inst.pattern().partition(node)))
            self.node=self.node["main"].body.op.body
        
    def get_io_from_layout(self,layout, data):
        if layout == "NCHW":
            n = data[0]
            c = data[1]
            h = data[2]
            w = data[3]
        else:
            # layout is nhwc
            n = data[0]
            c = data[3]
            h = data[1]
            w = data[2]
        return n, c, h, w

    def get_type(self, dtype):
        if dtype[:3] == "int":
            return "int"
        elif dtype[:4] == "uint":
            return "uint"
        # put here other cases
        return "int"

    def get_bits(self, dtype):
        if dtype[:3] == "int":
            return int(dtype[3:])
        elif dtype[:4] == "uint":
            return int(dtype[4:])
        # put here other cases
        return 8

    def get_bits_type(self, dtype):
        if dtype[:3] == "int":
            return int(dtype[3:]), "int"
        elif dtype[:4] == "uint":
            return int(dtype[4:]), "uint"
        # put here other cases
        return 8, "int"

    def get_unique_name_and_update_occs(self,call):
        unique_name=call.op.name
        if unique_name in self.occurrences:
            unique_name+=f"_{self.occurrences[unique_name]}"
            self.occurrences[unique_name]+=1
        else:
            self.occurrences[unique_name]=1
        return unique_name
    def visit_relu(self,call,attrs):
        attrs = {
            "activation": "relu",
        }
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_cast(self, call, attrs):
        op = ops.Cast(
            prec= self.get_bits(attrs.dtype),
            type= self.get_type(attrs.dtype),
        )
        unique_name=self.get_unique_name_and_update_occs(call)
        self.layer_data.layer_attrs[unique_name]=op

    def visit_right_shift(self, call, attrs):
        # nothing to do actually, right shift has no attrs and the arg is already saved before
        op = ops.RightShift(
            prec= self.get_bits(call.checked_type.dtype),
            type= self.get_type(call.checked_type.dtype),
        )
        unique_name=self.get_unique_name_and_update_occs(call)
        self.layer_data.layer_attrs[unique_name]=op

    def visit_clip(self, call, attrs):
        op = ops.Clip(
            min=int(attrs.a_min),
            max=int(attrs.a_max),
        )
        unique_name=self.get_unique_name_and_update_occs(call)
        self.layer_data.layer_attrs[unique_name]=op

    def visit_bias_add(self, call, attrs):
        op=ops.BiasAdd(
            bias_axis=int(attrs.axis),
        )
        unique_name=self.get_unique_name_and_update_occs(call)
        self.layer_data.layer_attrs[unique_name]=op

    def visit_multiply(self,call,atts):
        itype = [int(v) for v in call.args[0].args[0].checked_type.shape]
        iprec = call.args[0].args[0].checked_type.dtype
        wtype = [int(v) for v in call.args[1].args[0].checked_type.shape]
        wprec = call.args[1].args[0].checked_type.dtype
        otype = [int(v) for v in call.checked_type.shape]
        oprec = call.checked_type.dtype
        op=ops.Multiply(
            o_type=otype,
            o_prec=oprec,
            i_type=itype,
            i_prec=iprec,
            w_type=wtype,
            w_prec=wprec,
        )
        unique_name=self.get_unique_name_and_update_occs(call)
        self.layer_data.layer_attrs[unique_name]=op

    def visit_dense(self, call, attrs):
        itype = [int(v) for v in call.args[0].checked_type.shape]
        iprec = call.args[0].checked_type.dtype
        wtype = [int(v) for v in call.args[1].checked_type.shape]
        wprec = call.args[1].checked_type.dtype
        inp_features = itype[1]
        out_features = wtype[0]
        otype = [int(v) for v in call.checked_type.shape]
        oprec = call.checked_type.dtype
        if itype[1] != wtype[1]:
            raise NotImplementedError(f"The weights shape is not correct")
        op = ops.Dense(
            o_type=otype,
            o_prec=oprec,
            i_type=itype,
            i_prec=iprec,
            w_type=wtype,
            w_prec=wprec,
            inp_features=inp_features,
            out_features=out_features,
        )
        unique_name=self.get_unique_name_and_update_occs(call)
        self.layer_data.layer_attrs[unique_name]=op

    def visit_add(self, call, attrs):
        itype = [int(v) for v in call.args[0].args[0].checked_type.shape]
        iprec = call.args[0].args[0].checked_type.dtype
        wtype = [int(v) for v in call.args[1].args[0].checked_type.shape]
        wprec = call.args[1].args[0].checked_type.dtype
        otype = call.checked_type.shape
        oprec = call.checked_type.dtype
        i_n, i_c, i_h, i_w = self.get_io_from_layout("NCHW", itype)
        w_cout, w_cin, w_ksh, w_ksw = self.get_io_from_layout("NCHW", wtype)
        o_n, o_c, o_h, o_w = self.get_io_from_layout("NCHW", otype)
        padding = [0, 0, 0, 0]
        self.layer_data.strides = [1, 1]
        self.layer_data.dilations = [1, 1]
        groups = None
        if itype[0] != otype[0]:
            raise NotImplementedError(
                f"Input batch size is {itype[0]}, while output batch size is {otype[0]}"
            )
        op = ops.Add(
            o_type=otype,
            o_prec=oprec,
            i_type=itype,
            i_prec=iprec,
            w_type=wtype,
            w_prec=wprec,
        )
        unique_name=self.get_unique_name_and_update_occs(call)
        self.layer_data.layer_attrs[unique_name]=op

    def visit_conv_2d(self, call, attrs):
        depthwise = False
        if attrs.groups > 1:
            depthwise = True
        itype = [int(v) for v in call.args[0].checked_type.shape]
        iprec = call.args[0].checked_type.dtype
        wtype = [int(v) for v in call.args[1].checked_type.shape]
        wprec = call.args[1].checked_type.dtype
        otype = [int(v) for v in call.checked_type.shape]
        oprec = call.checked_type.dtype
        i_n, i_c, i_h, i_w = self.get_io_from_layout(attrs.data_layout, itype)
        w_cout, w_cin, w_ksh, w_ksw = self.get_io_from_layout(attrs.data_layout, wtype)
        o_n, o_c, o_h, o_w = self.get_io_from_layout(
            attrs.out_layout if attrs.out_layout != "" else attrs.data_layout, otype
        )
        if attrs.groups == w_cout and w_cin==1 and w_cout>1:
            depthwise = True
        padding = [int(a_p) for a_p in attrs.padding]
        self.layer_data.strides = [int(v) for v in attrs.strides]
        self.layer_data.dilations = [int(a_d) for a_d in attrs.dilation]
        groups = attrs.groups
        if itype[0] != otype[0]:
            raise NotImplementedError(
                f"Input batch size is {i_n}, while output batch size is {o_n}"
            )
        self.layer_data.dimension_relations = [
            f"ix={int(self.layer_data.strides[1])}*ox+{int(self.layer_data.dilations[1])}*fx",
            f"iy={int(self.layer_data.strides[0])}*oy+{int(self.layer_data.dilations[0])}*fy",
        ]
        kernel_size = list()
        if "kernel_size" in dict(attrs) and attrs["kernel_size"]!=None:
            kernel_size = list(attrs["kernel_size"])
        else:
            kernel_size = list([int(v) for v in wtype][2:])
        
        op = ops.Conv2d(
            padding= padding,
            strides= strides,
            dilation= dilations,
            groups= groups,
            o_type=otype,
            o_prec=oprec,
            i_type=itype,
            i_prec=iprec,
            w_type=wtype,
            w_prec=wprec,
            kernel_size=kernel_size,
            depthwise= depthwise,
        }
        unique_name=self.get_unique_name_and_update_occs(call)
        self.layer_data.layer_attrs[unique_name]=op

    def visit_calls_with_depth(self,call,depth_limit=[]):
        if not isinstance(depth_limit,list):
            return
        if not isinstance(call, tvm.relay.Call):
            return
        # the call is just a function pre partitioned
        elif not isinstance(call.op,tvm.ir.Op):
            return
        self.calllist.append(call)
        for idx_arg_,arg_ in enumerate(call.args):
            self.visit_calls_with_depth(arg_,depth_limit=depth_limit[idx_arg_])

    def visit_calls(self,call):
        if not isinstance(call, tvm.relay.Call):
            return
        # the call is just a function pre partitioned
        elif not isinstance(call.op,tvm.ir.Op):
            return
        self.calllist.append(call)
        for arg_ in call.args:
            self.visit_calls(arg_)
    def visit(self):
        call = self.node
        self.calllist=[]
        #self.visit_calls_with_depth(call,[self.depth_limits])
        self.visit_calls(call)
        self.calllist.reverse()
        var_and_consts_not_unrolled = dict()
        var_and_consts_unrolled = dict()
        for c in self.calllist:
            # fit variables and constants
            for a in c.args:
                if isinstance(a, tvm.relay.Var):
                    if len(self.args_list)>len(var_and_consts_not_unrolled):
                        ## this can be either a constant or the input still so let's check the real type
                        if isinstance(
                            self.args_list[len(var_and_consts_not_unrolled)], tvm.relay.Var
                        ):
                            var_and_consts_not_unrolled[a.name_hint] = self.args_list[
                                len(var_and_consts_not_unrolled)
                            ]
                            self.layer_data.layer_arguments.append((a.name_hint,self.args_list[
                                len(var_and_consts_not_unrolled)-1
                            ]))
                        else:
                            v_name=c.op.name+ f".param.{sum([c.op.name in k for k in var_and_consts_not_unrolled.keys()])}"
                            var_and_consts_not_unrolled[v_name] = self.args_list[len(var_and_consts_not_unrolled)]
                            self.layer_data.layer_arguments.append((v_name,self.args_list[len(var_and_consts_not_unrolled)-1]))
                    else:
                        var_and_consts_not_unrolled[
                            a.name_hint
                        ] = a
                        self.layer_data.layer_arguments.append((a.name_hint,a))
                elif isinstance(a, tvm.relay.Constant):
                    v_name=c.op.name+ f".param.{sum([c.op.name in k for k in var_and_consts_not_unrolled.keys()])}"
                    var_and_consts_unrolled[v_name] = a
                    self.layer_data.layer_arguments.append((v_name,a))
            self.index += 1
            self.layer_data.pattern_operations.append(c.op.name)
            self.visit_call(c)
            self.callsindexes[c] = self.index

    def visit_call(self, call):
        if call.op.name not in self.supported_operators:
            # skip if not supported for testing purposes
            print("[WORKLOAD PARSER] The operator",call.op.name,"is not supported yet")
            raise NotImplementedError(
                f"Currently the operator {call.op.name} is not supported."
            )
        else:
            self.layer_data.visited_operations.append(call.op.name)
            self.visit_router[call.op.name](call, call.attrs)

    def get_layer_data(self):
        return self.layer_data