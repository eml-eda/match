# TVM imports
import tvm
# utils imports
from math import ceil
from collections import OrderedDict
import copy
from typing import Any, Dict,List,Type

from match.target.exec_module import ExecModule
from match.codegen.layer_data import LayerData

def get_depth_arr_pattern(pattern_inst_):
    if isinstance(pattern_inst_,tvm.relay.dataflow_pattern.CallPattern):
        return [get_depth_arr_pattern(arg_) for arg_ in pattern_inst_.args]
    else:
        return 0

class WorkloadParser:
    def __init__(self, node:tvm.ir.IRModule, args_list:List=[],exec_module:ExecModule=None, pattern_name:str="", pattern_inst=None):
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
        }
        self.supported_operators = tuple(self.visit_router.keys())
        self.index = 0
        self.callsindexes = dict()
        self.calllist = []
        self.var_and_consts = dict()
        self.energy = 0
        self.latency = 0
        self.layer_data=LayerData()
        self.depth_limits=get_depth_arr_pattern(pattern_inst_=pattern_inst)
        print(self.depth_limits)
        
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

    def visit_cast(self, call, attrs):
        attrs = {"cast.prec": self.get_bits(attrs.dtype), "cast.type": self.get_type(attrs.dtype)}
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_right_shift(self, call, attrs):
        # nothing to do actually, right shift has no attrs and the arg is already saved before
        attrs = {
            "right_shift.prec": self.get_bits(call.checked_type.dtype),
            "right_shift.type": self.get_type(call.checked_type.dtype),
        }
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_clip(self, call, attrs):
        attrs = {
            "clip.min": int(attrs.a_min),
            "clip.max": int(attrs.a_max),
            "activation": "relu" if attrs.a_min >= 0 else None,
        }
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_bias_add(self, call, attrs):
        attrs = {"bias.axis": int(attrs.axis),'bias_add':True}
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}
    def visit_multiply(self,call,atts):
        attrs = {"batchnorm":True}
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_dense(self, call, attrs):
        itype = call.args[0].checked_type.shape
        iprec = call.args[0].checked_type.dtype
        wtype = call.args[1].checked_type.shape
        wprec = call.args[1].checked_type.dtype
        inp_features = itype[1]
        out_features = wtype[0]
        otype = call.checked_type.shape
        padding = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        groups = None
        if itype[1] != wtype[1]:
            raise NotImplementedError(f"The weights shape is not correct")
        self.layer_data.loop_dim_size = {
            "B": 1,
            "K": int(out_features),
            "C": int(inp_features),
            "OY": 1,
            "OX": 1,
            "FY": 1,
            "FX": 1,
        }
        self.layer_data.dimension_relations = [
            f"ix={int(strides[0])}*ox+{int(dilations[0])}*fx",
            f"iy={int(strides[1])}*oy+{int(dilations[1])}*fy",
        ]
        self.layer_data.operand_precision = {
            "O": self.get_bits("int8"),
            "O_final": self.get_bits("int8"),
            "W": self.get_bits(wprec),
            "I": self.get_bits(iprec),
        }
        self.layer_data.padding = {
            "IY": (int(padding[0]), int(padding[2])),
            "IX": (int(padding[1]), int(padding[3])),
        }
        self.layer_data.pr_loop_dim_size = {"IY": 1, "IX": 1}
        (
            self.layer_data.loop_dim_size,
            self.layer_data.pr_loop_dim_size,
            operand_precision_zigzag,
            self.layer_data.operand_precision,
        ) = self.exec_module.adjust_dimensions_and_precision()(
            loop_dim_size=self.layer_data.loop_dim_size, pr_loop_dim_size=self.layer_data.pr_loop_dim_size, operand_precision=self.layer_data.operand_precision, strides=strides, pattern_name=self.pattern_name
        )
        attrs = {
            "padding": padding,
            "strides": {"IX": int(strides[0]), "IY": int(strides[1])},
            "dilation": dilations,
            "groups": groups,
            "loop_sizes": {**self.layer_data.loop_dim_size, **self.layer_data.pr_loop_dim_size},
            "dense_prec": self.layer_data.operand_precision,
        }
        self.layer_data.ordered_relevant_loops = {
            "I": ["C", "OY", "OX"],
            "O": ["K", "OY", "OX"],
            "W": ["K", "C", "FY", "FX"],
        }
        self.layer_data.input_dim_mapping = {"C": "C", "OY": "IY", "OX": "IX"}
        self.layer_data.operands = ["O", "W", "I"]
        self.layer_data.input_operands = ["I"]
        self.layer_data.padded_dims = []
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_add(self, call, attrs):
        #if len(self.layer_data.loop_dim_size)==0:
        #    return
        itype = call.args[0].args[0].checked_type.shape
        iprec = call.args[0].args[0].checked_type.dtype
        wtype = call.args[1].args[0].checked_type.shape
        wprec = call.args[1].args[0].checked_type.dtype
        otype = call.checked_type.shape
        i_n, i_c, i_h, i_w = self.get_io_from_layout("NCHW", itype)
        w_cout, w_cin, w_ksh, w_ksw = self.get_io_from_layout("NCHW", wtype)
        o_n, o_c, o_h, o_w = self.get_io_from_layout("NCHW", otype)
        padding = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        groups = None
        if itype[0] != otype[0]:
            raise NotImplementedError(
                f"Input batch size is {i_n}, while output batch size is {o_n}"
            )
        self.layer_data.loop_dim_size = {
            "B": int(o_n),
            "K": int(o_c),
            "OY": int(o_h),
            "OX": int(o_w),
            "C": 1,
            "FX": 1,
            "FY": 1,
        }
        self.layer_data.operand_precision = {
            "O": self.get_bits("int8"),
            "O_final": self.get_bits("int8"),
            "X": self.get_bits(wprec),
            "Y": self.get_bits(iprec),
        }
        self.layer_data.pr_loop_dim_size = {"IY": int(i_h), "IX": int(i_w)}
        (
            self.layer_data.loop_dim_size,
            self.layer_data.pr_loop_dim_size,
            operand_precision_zigzag,
            self.layer_data.operand_precision,
        ) = self.exec_module.adjust_dimensions_and_precision(
            loop_dim_size=self.layer_data.loop_dim_size, pr_loop_dim_size=self.layer_data.pr_loop_dim_size, operand_precision=self.layer_data.operand_precision, strides=strides, pattern_name=self.pattern_name
        )
        self.layer_data.equation = "O[b][k][oy][ox]+=X[b][k][oy][ox]*Y[b][k][oy][ox]"
        self.layer_data.ordered_relevant_loops = {
            "X": ["K", "OY", "OX"],
            "O": ["K", "OY", "OX"],
            "Y": ["K", "OY", "OX"],
        }
        self.layer_data.input_dim_mapping = {"K": "C", "OY": "IY", "OX": "IX"}
        self.layer_data.operands = ["O", "X", "Y"]
        self.layer_data.input_operands = ["X", "Y"]
        self.layer_data.padded_dims = []
        
        self.layer_data.pr_loop_dim_size["C"] = self.layer_data.loop_dim_size["K"]
        loop_dim_size_attrs = copy.deepcopy(self.layer_data.loop_dim_size)
        del loop_dim_size_attrs["C"]
        attrs = {
            "padding": padding,
            "strides": {"IX": int(strides[0]), "IY": int(strides[1])},
            "dilation": [1, 1],
            "groups": 1,
            "loop_sizes": {**loop_dim_size_attrs, **self.layer_data.pr_loop_dim_size},
            "add_prec": self.layer_data.operand_precision,
        }
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_conv_2d(self, call, attrs):
        depthwise = False
        if attrs.groups > 1:
            depthwise = True
        itype = call.args[0].checked_type.shape
        iprec = call.args[0].checked_type.dtype
        wtype = call.args[1].checked_type.shape
        wprec = call.args[1].checked_type.dtype
        otype = call.checked_type.shape
        i_n, i_c, i_h, i_w = self.get_io_from_layout(attrs.data_layout, itype)
        w_cout, w_cin, w_ksh, w_ksw = self.get_io_from_layout(attrs.data_layout, wtype)
        o_n, o_c, o_h, o_w = self.get_io_from_layout(
            attrs.out_layout if attrs.out_layout != "" else attrs.data_layout, otype
        )
        padding = [int(a_p) for a_p in attrs.padding]
        strides = [int(v) for v in attrs.strides]
        dilations = [int(a_d) for a_d in attrs.dilation]
        groups = attrs.groups
        if itype[0] != otype[0]:
            raise NotImplementedError(
                f"Input batch size is {i_n}, while output batch size is {o_n}"
            )
        self.layer_data.dimension_relations = [
            f"ix={int(strides[0])}*ox+{int(dilations[0])}*fx",
            f"iy={int(strides[1])}*oy+{int(dilations[1])}*fy",
        ]
        kernel_size = attrs.kernel_size
        self.layer_data.loop_dim_size = {
            "B": int(o_n),
            "K": int(o_c),
            "C": int(w_cin),
            "OY": int(o_h),
            "OX": int(o_w),
            "FY": int(kernel_size[0]),
            "FX": int(kernel_size[1]),
        }
        self.layer_data.operand_precision = {
            "O": self.get_bits("int8"),
            "O_final": self.get_bits("int8"),
            "W": self.get_bits(wprec),
            "I": self.get_bits(iprec),
        }
        self.layer_data.padding = {
            "IY": (int(padding[0]), int(padding[2])),
            "IX": (int(padding[1]), int(padding[3])),
        }
        self.layer_data.pr_loop_dim_size = {"IY": int(i_h), "IX": int(i_w)}
        (
            self.layer_data.loop_dim_size,
            self.layer_data.pr_loop_dim_size,
            operand_precision_zigzag,
            self.layer_data.operand_precision,
        ) = self.exec_module.adjust_dimensions_and_precision(
            loop_dim_size=self.layer_data.loop_dim_size, pr_loop_dim_size=self.layer_data.pr_loop_dim_size, operand_precision=self.layer_data.operand_precision, strides=strides, pattern_name=self.pattern_name
        )
        self.layer_data.equation = (
            "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][k][iy][ix]"
            if depthwise
            else "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]"
        )
        self.layer_data.constant_operands=["W"]
        self.layer_data.operand_source_dimension_mapping={'I': {'IX': 'OX', 'IY': 'OY','C':'K'}}
        self.layer_data.operand_source={"W": [], "I": []}
        self.layer_data.ordered_relevant_loops = {
            "I": [['OY','OX','C'],[("C" if not depthwise else "K"), "OY", "OX"]][1],
            "O": [['OY','OX','K'],["K", "OY", "OX"]][1],
            "W": [['K','C','FY','FX'],["K", "C", "FY", "FX"]][1],
        }
        self.layer_data.input_dim_mapping = {("C" if not depthwise else "K"): "C", "OY": "IY", "OX": "IX"}
        self.layer_data.operands = ["O", "W", "I"]
        self.layer_data.input_operands = ["I"]
        self.layer_data.padded_dims = ["OX", "OY"]
        attrs = {
            "padding": padding,
            "strides": {"IX": int(strides[0]), "IY": int(strides[1])},
            "dilation": dilations,
            "groups": groups,
            "loop_sizes": {**self.layer_data.loop_dim_size, **self.layer_data.pr_loop_dim_size},
            "nn.conv2d_prec": self.layer_data.operand_precision,
            "nn.conv2d_depthwise": depthwise,
        }
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_calls(self,call,depth_limit=[]):
        if not isinstance(depth_limit,list):
            return
        if not isinstance(call, tvm.relay.Call):
            return
        # the call is just a function pre partitioned
        elif not isinstance(call.op,tvm.ir.Op):
            return
        self.calllist.append(call)
        for idx_arg_,arg_ in enumerate(call.args):
            self.visit_calls(arg_,depth_limit=depth_limit[idx_arg_])

    def visit(self):
        call = self.node
        self.calllist=[]
        self.visit_calls(call,self.depth_limits)
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
                        else:
                            var_and_consts_not_unrolled[
                            c.op.name
                            + f".param.{sum([c.op.name in k for k in var_and_consts_not_unrolled.keys()])}"
                            ] = self.args_list[len(var_and_consts_not_unrolled)]
                    else:
                        var_and_consts_not_unrolled[
                            a.name_hint
                        ] = a
                elif isinstance(a, tvm.relay.Constant):
                    var_and_consts_unrolled[
                        c.op.name
                        + f".param.{sum([c.op.name in k for k in var_and_consts_not_unrolled.keys()])}"
                    ] = a
            self.index += 1
            self.layer_data.pattern_operations.append(c.op.name)
            self.visit_call(c)
            self.callsindexes[c] = self.index
        self.layer_data.layer_arguments = {**var_and_consts_not_unrolled, **var_and_consts_unrolled}

    def visit_call(self, call):
        #breakpoint()
        if call.op.name not in self.supported_operators:
            # skip if not supported for testing purposes
            raise NotImplementedError(
                f"Currently the operator {call.op.name} is not supported."
            )
        else:
            self.visit_router[call.op.name](call, call.attrs)

    def get_layer_data(self):
        return self.layer_data