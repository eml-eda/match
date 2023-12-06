# TVM imports
import tvm
# utils imports
from math import ceil
from collections import OrderedDict
import copy
from typing import Any, Dict,List,Type

from match.hwmodel.hwmodel import HwModel
from match.codegen.layer_data import LayerData

class WorkloadParser:
    def __init__(self, mod:tvm.ir.IRModule, hwmodel:HwModel=None, pattern_name:str=""):
        self.hwmodel=hwmodel
        self.pattern_name=pattern_name
        self.mod = mod
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
        self.workload = dict()
        self.index = 0
        self.callsindexes = dict()
        self.calllist = []
        self.var_and_consts = dict()
        self.cme = None
        self.energy = 0
        self.latency = 0
        self.layer_data=LayerData()

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
        #bnorm=False
        bnorm=True
        if not bnorm or self.layer_data.workload_name=="element_wise_sum":
            attrs = {"bias.axis": int(attrs.axis),'bias_add':True}
        else:
            attrs = {"bias.axis": int(attrs.axis),'bias_add':False,"batchnorm":True}
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}
    def visit_multiply(self,call,atts):
        attrs = {"batchnorm":True}
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}
    def visit_dense(self, call, attrs):
        itype = call.args[0].type_annotation.shape
        iprec = call.args[0].type_annotation.dtype
        wtype = call.args[1].type_annotation.shape
        wprec = call.args[1].type_annotation.dtype
        inp_features = itype[1]
        out_features = wtype[0]
        otype = call.checked_type.shape
        padding = [0, 0, 0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        groups = None
        if itype[1] != wtype[1]:
            raise NotImplementedError(f"The weights shape is not correct")
        self.layer_data.workload_name = "dense"
        loop_dim_size = {
            "B": 1,
            "K": int(out_features),
            "C": int(inp_features),
            "OY": 1,
            "OX": 1,
            "FY": 1,
            "FX": 1,
        }
        dimension_relations = [
            f"ix={int(strides[0])}*ox+{int(dilations[0])}*fx",
            f"iy={int(strides[1])}*oy+{int(dilations[1])}*fy",
        ]
        operand_precision = {
            "O": self.get_bits("int8"),
            "O_final": self.get_bits("int8"),
            "W": self.get_bits(wprec),
            "I": self.get_bits(iprec),
        }
        padding = {
            "IY": (int(padding[0]), int(padding[2])),
            "IX": (int(padding[1]), int(padding[3])),
        }
        pr_loop_dim_size = {"IY": 1, "IX": 1}
        (
            loop_dim_size,
            pr_loop_dim_size,
            operand_precision_zigzag,
            operand_precision,
        ) = self.hwmodel.adjust_dimensions_and_precision()(
            loop_dim_size=loop_dim_size, pr_loop_dim_size=pr_loop_dim_size, operand_precision=operand_precision, strides=strides, workload_name=self.layer_data.workload_name
        )
        self.workload = {
            "operator_type": "dense",
            "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
            "dimension_relations": dimension_relations,
            "loop_dim_size": loop_dim_size,
            "operand_precision": operand_precision_zigzag,
            "pr_loop_dim_size": pr_loop_dim_size,
            "padding": padding,
            "operand_source": {"W": [], "I": []},
            "constant_operands": ["W"],
            "operand_source_dimension_mapping": {"I": {"IX": "OX", "IY": "OY"}},
        }
        attrs = {
            "padding": padding,
            "strides": {"IX": int(strides[0]), "IY": int(strides[1])},
            "dilation": dilations,
            "groups": groups,
            "loop_sizes": {**loop_dim_size, **pr_loop_dim_size},
            "dense_prec": operand_precision,
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

    def visit_depthwise_conv_2d(self, call, attrs):
        return

    def visit_add(self, call, attrs):
        def get_io_from_layout(layout, data):
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
        if "batchnorm" in self.attrs and self.attrs['batchnorm']:
            return
        itype = call.args[0].args[0].type_annotation.shape
        iprec = call.args[0].args[0].type_annotation.dtype
        wtype = call.args[1].args[0].type_annotation.shape
        wprec = call.args[1].args[0].type_annotation.dtype
        otype = call.checked_type.shape
        i_n, i_c, i_h, i_w = get_io_from_layout("NCHW", itype)
        w_cout, w_cin, w_ksh, w_ksw = get_io_from_layout("NCHW", wtype)
        o_n, o_c, o_h, o_w = get_io_from_layout("NCHW", otype)
        if itype[0] != otype[0]:
            raise NotImplementedError(
                f"Input batch size is {i_n}, while output batch size is {o_n}"
            )
        dimension_relations = []
        self.layer_data.workload_name = "element_wise_sum"
        loop_dim_size = {
            "B": int(o_n),
            "K": int(o_c),
            "OY": int(o_h),
            "OX": int(o_w),
            "C": 1,
            "FX": 1,
            "FY": 1,
        }
        operand_precision = {
            "O": self.get_bits("int8"),
            "O_final": self.get_bits("int8"),
            "X": self.get_bits(wprec),
            "Y": self.get_bits(iprec),
        }
        padding = {"IY": (0, 0), "IX": (0, 0)}
        strides = [1, 1]
        pr_loop_dim_size = {"IY": int(i_h), "IX": int(i_w)}
        (
            loop_dim_size,
            pr_loop_dim_size,
            operand_precision_zigzag,
            operand_precision,
        ) = self.hwmodel.adjust_dimensions_and_precision(
            loop_dim_size=loop_dim_size, pr_loop_dim_size=pr_loop_dim_size, operand_precision=operand_precision, strides=strides, workload_name=self.layer_data.workload_name
        )
        equation = "O[b][k][oy][ox]+=X[b][k][oy][ox]*Y[b][k][oy][ox]"
        self.layer_data.ordered_relevant_loops = {
            "X": ["K", "OY", "OX"],
            "O": ["K", "OY", "OX"],
            "Y": ["K", "OY", "OX"],
        }
        self.layer_data.input_dim_mapping = {"K": "C", "OY": "IY", "OX": "IX"}
        self.layer_data.operands = ["O", "X", "Y"]
        self.layer_data.input_operands = ["X", "Y"]
        self.layer_data.padded_dims = []
        self.workload = {
            "operator_type": self.layer_data.workload_name,
            "equation": equation,
            "dimension_relations": dimension_relations,
            "loop_dim_size": loop_dim_size,
            "operand_precision": operand_precision_zigzag,
            "pr_loop_dim_size": pr_loop_dim_size,
            "padding": padding,
            "operand_source": {"X": [], "Y": []},
            "constant_operands": [],
            #'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY','C':'K'}},
        }
        pr_loop_dim_size["C"] = loop_dim_size["K"]
        loop_dim_size_attrs = copy.deepcopy(loop_dim_size)
        del loop_dim_size_attrs["C"]
        attrs = {
            "padding": padding,
            "strides": {"IX": int(strides[0]), "IY": int(strides[1])},
            "dilation": [1, 1],
            "groups": 1,
            "loop_sizes": {**loop_dim_size_attrs, **pr_loop_dim_size},
            f"{self.layer_data.workload_name}_prec": operand_precision,
        }
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit_conv_2d(self, call, attrs):
        depthwise = False
        if attrs.groups > 1:
            self.visit_depthwise_conv_2d(call, attrs)
            depthwise = True

        def get_io_from_layout(layout, data):
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

        itype = call.args[0].type_annotation.shape
        iprec = call.args[0].type_annotation.dtype
        wtype = call.args[1].type_annotation.shape
        wprec = call.args[1].type_annotation.dtype
        otype = call.checked_type.shape
        i_n, i_c, i_h, i_w = get_io_from_layout(attrs.data_layout, itype)
        w_cout, w_cin, w_ksh, w_ksw = get_io_from_layout(attrs.data_layout, wtype)
        o_n, o_c, o_h, o_w = get_io_from_layout(
            attrs.out_layout if attrs.out_layout != "" else attrs.data_layout, otype
        )
        padding = attrs.padding
        strides = [int(v) for v in attrs.strides]
        dilations = attrs.dilation
        groups = attrs.groups
        if itype[0] != otype[0]:
            raise NotImplementedError(
                f"Input batch size is {i_n}, while output batch size is {o_n}"
            )
        dimension_relations = [
            f"ix={int(strides[0])}*ox+{int(dilations[0])}*fx",
            f"iy={int(strides[1])}*oy+{int(dilations[1])}*fy",
        ]
        kernel_size = attrs.kernel_size
        self.layer_data.workload_name = "depthwise_conv_2d" if depthwise else "conv_2d"
        loop_dim_size = {
            "B": int(o_n),
            "K": int(o_c),
            "C": int(w_cin),
            "OY": int(o_h),
            "OX": int(o_w),
            "FY": int(kernel_size[0]),
            "FX": int(kernel_size[1]),
        }
        operand_precision = {
            "O": self.get_bits("int8"),
            "O_final": self.get_bits("int8"),
            "W": self.get_bits(wprec),
            "I": self.get_bits(iprec),
        }
        padding = {
            "IY": (int(padding[0]), int(padding[2])),
            "IX": (int(padding[1]), int(padding[3])),
        }
        pr_loop_dim_size = {"IY": int(i_h), "IX": int(i_w)}
        (
            loop_dim_size,
            pr_loop_dim_size,
            operand_precision_zigzag,
            operand_precision,
        ) = self.hwmodel.adjust_dimensions_and_precision(
            loop_dim_size=loop_dim_size, pr_loop_dim_size=pr_loop_dim_size, operand_precision=operand_precision, strides=strides, workload_name=self.layer_data.workload_name
        )
        equation = (
            "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][k][iy][ix]"
            if depthwise
            else "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]"
        )
        self.layer_data.ordered_relevant_loops = {
            "I": [['OY','OX','C'],[("C" if not depthwise else "K"), "OY", "OX"]][1],
            "O": [['OY','OX','K'],["K", "OY", "OX"]][1],
            "W": [['K','C','FY','FX'],["K", "C", "FY", "FX"]][1],
        }
        self.layer_data.input_dim_mapping = {("C" if not depthwise else "K"): "C", "OY": "IY", "OX": "IX"}
        self.layer_data.operands = ["O", "W", "I"]
        self.layer_data.input_operands = ["I"]
        self.layer_data.padded_dims = ["OX", "OY"]
        self.workload = {
            "operator_type": self.layer_data.workload_name,
            "equation": equation,
            "dimension_relations": dimension_relations,
            "loop_dim_size": loop_dim_size,
            "operand_precision": operand_precision_zigzag,
            "pr_loop_dim_size": pr_loop_dim_size,
            "padding": padding,
            "operand_source": {"W": [], "I": []},
            "constant_operands": ["W"],
            #'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY','C':'K'}},
        }
        attrs = {
            "padding": padding,
            "strides": {"IX": int(strides[0]), "IY": int(strides[1])},
            "dilation": dilations,
            "groups": groups,
            "loop_sizes": {**loop_dim_size, **pr_loop_dim_size},
            f"{self.layer_data.workload_name}_prec": operand_precision,
            "dephtwise": depthwise,
        }
        self.layer_data.layer_attrs = {**self.layer_data.layer_attrs, **attrs}

    def visit(self):
        call = self.mod.body.op.body
        while call is not None:
            self.calllist.append(call)
            if len(call.args) > 0 and isinstance(call.args[0], tvm.relay.Call):
                call = call.args[0]
            else:
                call = None
        self.calllist.reverse()
        var_and_consts_not_unrolled = dict()
        var_and_consts_unrolled = dict()
        for c in self.calllist:
            # fit variables and constants
            for a in c.args:
                if isinstance(a, tvm.relay.Var):
                    ## this can be either a constant or the input still so let's check the real type
                    if isinstance(
                        self.mod.body.args[len(var_and_consts_not_unrolled)], tvm.relay.Var
                    ):
                        var_and_consts_not_unrolled[a.name_hint] = self.mod.body.args[
                            len(var_and_consts_not_unrolled)
                        ]
                    else:
                        var_and_consts_not_unrolled[
                            c.op.name
                            + f".param.{sum([c.op.name in k for k in var_and_consts_not_unrolled.keys()])}"
                        ] = self.mod.body.args[len(var_and_consts_not_unrolled)]
                elif isinstance(a, tvm.relay.Constant):
                    var_and_consts_unrolled[
                        c.op.name
                        + f".param.{sum([c.op.name in k for k in var_and_consts_not_unrolled.keys()])}"
                    ] = a
            self.index += 1
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

    def get_workload(self):
        return self.workload

    def get_layer_data(self):
        return self.layer_data