import importlib
import numpy as np

# TVM imports
import tvm
from tvm.relay.dataflow_pattern import is_op
from tvm.relay.expr_functor import ExprVisitor

# ZigZag imports
from math import ceil
from zigzag import api
from mako.template import Template
from mako import exceptions
import functools
import operator
from collections import OrderedDict
import copy

class ZigZagWorkloadParser:
    def __init__(self, arch_def, mod, memory_sizes:dict[str,int]={}, lpf_limit: int = 6):
        self.arch_def = arch_def
        self.memory_sizes = dict() if memory_sizes is None else memory_sizes
        self.lpf_limit=lpf_limit
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
        self.attrs = dict()
        self.for_loops = []
        self.type = None
        self.accelerator = None
        self.mapping = None
        self.profiling = {
            "main": ["main"],
            "totals": ["mem_total", "kern_total"],
            "mem_total": ["mem_total"],
            "kern_total": ["kern_total"],
            "calls": ["mem_call", "kern_call"],
            "mem_call": ["mem_call"],
            "kern_call": ["kern_call"],
            "jobs": ["mem_job", "mem_job_on", "mem_job_off", "kern_job"],
            "mem_job": ["mem_job", "mem_job_on", "mem_job_off"],
            "mem_job_on": ["mem_job_on"],
            "mem_job_off": ["mem_job_off"],
            "kern_job": ["kern_job"],
            "nothing": [],
        }

    def visit_cast(self, call, attrs):
        attrs = {"cast.prec": self.get_bits(attrs.dtype), "cast.type": self.get_type(attrs.dtype)}
        self.attrs = {**self.attrs, **attrs}

    def visit_right_shift(self, call, attrs):
        # nothing to do actually, right shift has no attrs and the arg is already saved before
        attrs = {
            "right_shift.prec": self.get_bits(call.checked_type.dtype),
            "right_shift.type": self.get_type(call.checked_type.dtype),
        }
        self.attrs = {**self.attrs, **attrs}

    def visit_clip(self, call, attrs):
        attrs = {
            "clip.min": int(attrs.a_min),
            "clip.max": int(attrs.a_max),
            "activation": "relu" if attrs.a_min >= 0 else None,
        }
        self.attrs = {**self.attrs, **attrs}

    def visit_bias_add(self, call, attrs):
        #bnorm=False
        bnorm=True
        if not bnorm or self.type=="element_wise_sum":
            attrs = {"bias.axis": int(attrs.axis),'bias_add':True}
        else:
            attrs = {"bias.axis": int(attrs.axis),'bias_add':False,"batchnorm":True}
        self.attrs = {**self.attrs, **attrs}
    def visit_multiply(self,call,atts):
        attrs = {"batchnorm":True}
        self.attrs = {**self.attrs, **attrs}
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
        self.type = "dense"
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
        ) = self.arch_def["hw_adjust_dimensions_and_precision"](
            loop_dim_size, pr_loop_dim_size, operand_precision, strides, self.type
        )
        self.workload[self.index] = {
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
        self.ordered_relevant_loops = {
            "I": ["C", "OY", "OX"],
            "O": ["K", "OY", "OX"],
            "W": ["K", "C", "FY", "FX"],
        }
        self.inp_dim_mapping = {"C": "C", "OY": "IY", "OX": "IX"}
        self.operands = ["O", "W", "I"]
        self.input_operands = ["I"]
        self.padded_dims = []
        self.attrs = {**self.attrs, **attrs}
        self.dims = {
            "I": {
                "default_mem": "dram",
                "relevant_loops": {
                    "C": {"mapping": "C"},
                    "OY": {"mapping": "IY", "partial_relevancy": "FY"},
                    "OX": {"mapping": "IX", "partial_relevancy": "FX"},
                },
            },
            "W": {
                "default_mem": "dram",
                "relevant_loops": {
                    "K": {"mapping": "K"},
                    "C": {"mapping": "C"},
                    "FY": {"mapping": "FY"},
                    "FX": {"mapping": "FX"},
                },
            },
            "O": {
                "default_mem": "dram",
                "relevant_loops": {
                    "K": {"mapping": "K"},
                    "OY": {"mapping": "OY"},
                    "OX": {"mapping": "OX"},
                },
            },
        }

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
        self.type = "element_wise_sum"
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
        ) = self.arch_def["hw_adjust_dimensions_and_precision"](
            loop_dim_size, pr_loop_dim_size, operand_precision, strides, self.type
        )
        equation = "O[b][k][oy][ox]+=X[b][k][oy][ox]*Y[b][k][oy][ox]"
        self.ordered_relevant_loops = {
            "X": ["K", "OY", "OX"],
            "O": ["K", "OY", "OX"],
            "Y": ["K", "OY", "OX"],
        }
        self.inp_dim_mapping = {"K": "C", "OY": "IY", "OX": "IX"}
        self.operands = ["O", "X", "Y"]
        self.input_operands = ["X", "Y"]
        self.padded_dims = []
        self.workload[1] = {
            "operator_type": self.type,
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
            f"{self.type}_prec": operand_precision,
        }
        self.attrs = {**self.attrs, **attrs}
        self.dims = {
            "X": {
                "default_mem": "dram",
                "relevant_loops": {
                    "K": {"mapping": "C"},
                    "OY": {"mapping": "IY"},
                    "OX": {"mapping": "IX"},
                },
            },
            "Y": {
                "default_mem": "dram",
                "relevant_loops": {
                    "K": {"mapping": "C"},
                    "OY": {"mapping": "IY"},
                    "OX": {"mapping": "IX"},
                },
            },
            "O": {
                "default_mem": "dram",
                "relevant_loops": {
                    "K": {"mapping": "K"},
                    "OY": {"mapping": "OY"},
                    "OX": {"mapping": "OX"},
                },
            },
        }

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
        self.type = "depthwise_conv_2d" if depthwise else "conv_2d"
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
        ) = self.arch_def["hw_adjust_dimensions_and_precision"](
            loop_dim_size, pr_loop_dim_size, operand_precision, strides, self.type
        )
        equation = (
            "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][k][iy][ix]"
            if depthwise
            else "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]"
        )
        self.ordered_relevant_loops = {
            "I": [['OY','OX','C'],[("C" if not depthwise else "K"), "OY", "OX"]][1],
            "O": [['OY','OX','K'],["K", "OY", "OX"]][1],
            "W": [['K','C','FY','FX'],["K", "C", "FY", "FX"]][1],
        }
        self.inp_dim_mapping = {("C" if not depthwise else "K"): "C", "OY": "IY", "OX": "IX"}
        self.operands = ["O", "W", "I"]
        self.input_operands = ["I"]
        self.padded_dims = ["OX", "OY"]
        self.workload[self.index] = {
            "operator_type": self.type,
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
            f"{self.type}_prec": operand_precision,
            "dephtwise": depthwise,
        }
        self.attrs = {**self.attrs, **attrs}
        self.dims = {
            "I": {
                "default_mem": "dram",
                "relevant_loops": {
                    ("K" if depthwise else "C"): {"mapping": "C"},
                    "OY": {"mapping": "IY", "partial_relevancy": "FY"},
                    "OX": {"mapping": "IX", "partial_relevancy": "FX"},
                },
            },
            "W": {
                "default_mem": "dram",
                "relevant_loops": {
                    "K": {"mapping": "K"},
                    "C": {"mapping": "C"},
                    "FY": {"mapping": "FY"},
                    "FX": {"mapping": "FX"},
                },
            },
            "O": {
                "default_mem": "dram",
                "relevant_loops": {
                    "K": {"mapping": "K"},
                    "OY": {"mapping": "OY"},
                    "OX": {"mapping": "OX"},
                },
            },
        }

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
                            str(c.op)
                            + f".param.{sum([str(c.op) in k for k in var_and_consts_not_unrolled.keys()])}"
                        ] = self.mod.body.args[len(var_and_consts_not_unrolled)]
                elif isinstance(a, tvm.relay.Constant):
                    var_and_consts_unrolled[
                        str(c.op)
                        + f".param.{sum([str(c.op) in k for k in var_and_consts_not_unrolled.keys()])}"
                    ] = a
            self.index += 1
            self.visit_call(c)
            self.callsindexes[c] = self.index
        self.var_and_consts = {**var_and_consts_not_unrolled, **var_and_consts_unrolled}

    def visit_call(self, call):
        if str(call.op) not in self.supported_operators:
            # skip if not supported for testing purposes
            return
            # raise NotImplementedError(
            #    f"Currently the operator {str(call.op)} is not supported."
            # )
        else:
            self.visit_router[str(call.op)](call, call.attrs)

    def zigzag(
        self, dory_tmap: bool = False, std_cost_model: bool = True, temporal_ordering: list = []
    ):
        # skip if not supported for testing purposes
        if len(self.workload.keys()) == 0:
            return
        """needed_mem=(self.attrs['loop_sizes']['IX']*self.attrs['loop_sizes']['IY']*\
        self.attrs['loop_sizes']['C']*self.attrs['loop_sizes']['B']*(self.attrs[f'{self.type}_prec']['I']/8))\
        + (self.attrs['loop_sizes']['OX']*self.attrs['loop_sizes']['OY']*\
        self.attrs['loop_sizes']['K']*self.attrs['loop_sizes']['B']*(self.attrs[f'{self.type}_prec']['O_final']/8))\
         + (self.attrs['loop_sizes']['K']*self.attrs['loop_sizes']['C']*\
         self.attrs['loop_sizes']['FY']*self.attrs['loop_sizes']['FX']*(self.attrs[f'{self.type}_prec']['W']/8))
        if needed_mem>self.arch_def['limits']['max_memory']:
            raise Exception(f'This layer requires more memory than the one available, memory required {needed_mem} , memory available {diana_arch.max_memory}')"""
        print(f"Generating temporal mapping with following workload\n{self.workload}")
        self.mapping = self.arch_def["mapping"](self.attrs["loop_sizes"], self.type)
        # if any([val[1] != 16 for key, val in self.mapping[self.type]["spatial_mapping"].items()]):
        #     # not supported yet feed to DORY
        #     self.workload = dict()
        #     return
        self.accelerator = self.arch_def["accelerator"](self.input_operands,self.memory_sizes,self.type,self.attrs["loop_sizes"])
        self.workload[1]["cost_model"] = self.arch_def["cost_model"](std_cost_model)
        self.workload[1]["attrs"] = self.attrs
        print(self.lpf_limit)#+sum([self.attrs['loop_sizes'][fixedlp] for fixedlp in self.mapping[self.type]['fixed_loops']]))
        if not dory_tmap:
            energy, latency, cme = api.get_hardware_performance_zigzag(
                workload=self.workload,
                accelerator=self.accelerator,
                mapping=self.mapping,
                opt="latency",
                dump_filename_pattern=f"outputs/diana-{self.mod.attrs.global_symbol}-layer_?.json",
                pickle_filename=f"outputs/diana-{self.mod.attrs.global_symbol}-saved_list_of_cmes.pickle",
                lpf_limit=self.lpf_limit#+sum([self.attrs['loop_sizes'][fixedlp] for fixedlp in self.mapping[self.type]['fixed_loops']])
            )
        else:
            import copy

            dory_mapping = copy.deepcopy(self.mapping)
            dory_mapping[self.type]["temporal_ordering"] = temporal_ordering
            energy, latency, cme = api.get_hardware_performance_zigzag(
                workload=self.workload,
                accelerator=self.accelerator,
                mapping=dory_mapping,
                opt="latency",
                dump_filename_pattern=f"outputs/diana-dory-{self.mod.attrs.global_symbol}-layer_?.json",
                pickle_filename=f"outputs/diana-dory-{self.mod.attrs.global_symbol}-saved_list_of_cmes.pickle",
                temp_mapping=True,
                lpf_limit=self.lpf_limit#+sum([self.attrs['loop_sizes'][fixedlp] for fixedlp in self.mapping[self.type]['fixed_loops']])
            )
        self.cme = cme[0][0]
        print(f"\n\nOur result Latency was Comp {self.cme.latency_total0} total {self.cme.latency_total2}\n\n")
        self.energy = energy
        self.latency = latency
        print(f"Total network energy = {energy:.2e} pJ")
        print(f"Total network latency = {latency:.2e} cycles")
        from zigzag.visualization.results.print_mapping import print_mapping

        print("Mapping")
        print_mapping(self.cme)
        tm = self.cme.temporal_mapping.mapping_dic_stationary
        print(f"\nThe temporal mapping is the following:\n{tm}\n")
        mem_op_to_layer_op = self.cme.mem_op_to_layer_op
        layer_op_to_mem_op = self.cme.layer_op_to_mem_op
        mem_name = {}
        for mem_op, mems_all_levels in self.cme.accelerator.cores[0].mem_hierarchy_dict.items():
            layer_op = mem_op_to_layer_op[mem_op]
            mem_name[layer_op] = []
            for mem_a_level in mems_all_levels:
                mem_name[layer_op].append(mem_a_level.name)

        for layer_op, tm_layer_levels in tm.items():
            layerfound = []
            for idx, levels in enumerate(tm_layer_levels):
                for loop_name, loop_size in levels:
                    nameidx = sum([loop_name == el for el in layerfound])
                    fullname = f"{loop_name}_{nameidx}" if nameidx > 0 else loop_name
                    layerfound.append(loop_name)
                    if fullname not in [el["fullname"] for el in self.for_loops]:
                        self.for_loops.append(
                            {
                                "name": loop_name,
                                "index": nameidx,
                                "fullname": fullname,
                                "size": loop_size,
                                f"mem_{layer_op}": mem_name[layer_op][idx],
                            }
                        )
                    else:
                        self.for_loops[[el["fullname"] for el in self.for_loops].index(fullname)][
                            f"mem_{layer_op}"
                        ] = mem_name[layer_op][idx]

        self.for_loops = self.arch_def["hw_dependent_temporal_mapping"](
            self.for_loops, self.attrs["loop_sizes"], self.mapping, self.type
        )
        print(f"\nGot these for loops:\n{self.for_loops[::-1]}\n")

    def template_data(self):
        temp_info = OrderedDict([])
        temp_info["attrs"] = self.attrs
        temp_info["weights"] = self.arch_def["template_data"]["weights"](
            self.var_and_consts, self.attrs["loop_sizes"], self.type, temp_info, self.mapping
        )
        temp_info["func_name"] = self.mod.attrs.global_symbol
        temp_info["func_number"] = int(temp_info["func_name"].split("_")[::-1][0])
        temp_info["dims"] = self.dims
        temp_info=self.arch_def["template_data"]["hw_dependent_template_data"](
                temp_info, self.mapping, self.type
        )
        if any([temp_info['ordered_operand_memories'][op_][::-1][0]!=self.for_loops[0][f'mem_{op_}'] for op_ in self.operands]):
            onesizedlooptoadd=dict()
            onesizedlooptoadd['name']='K'
            onesizedlooptoadd['fullname']='K'
            onesizedlooptoadd['size']=1
            onesizedlooptoadd['index']=0
            for lp in self.for_loops:
                if lp['name']=='K':
                    lp['index']+=1
                    lp['fullname']='K_'+str(lp['index'])
            for op_ in self.operands:
                onesizedlooptoadd[f'mem_{op_}']=temp_info['ordered_operand_memories'][op_][::-1][0]
            self.for_loops=[onesizedlooptoadd]+self.for_loops
        temp_info["for_loops"] = self.for_loops[::-1]
        temp_info["layer_name"] = self.type
        spatial_dims = [
            val[0] for val in self.mapping[temp_info["layer_name"]]["spatial_mapping"].values()
        ]
        my_for_loops = copy.deepcopy(temp_info["for_loops"])
        for spatial_dim in spatial_dims:
            sp_val = np.prod(
                [
                    spat_loop["size"]
                    for spat_loop in temp_info["for_loops"]
                    if spat_loop["name"] == spatial_dim
                ]
            )
            if temp_info["attrs"]["loop_sizes"][spatial_dim] != sp_val:
                for idxox in range(len(my_for_loops)):
                    if my_for_loops[idxox]["name"] == spatial_dim:
                        my_for_loops[idxox]["index"] += 1
                        my_for_loops[idxox][
                            "fullname"
                        ] = f'{spatial_dim}_{my_for_loops[idxox]["index"]}'
                obj = {
                    "name": spatial_dim,
                    "fullname": spatial_dim,
                    "size": temp_info["attrs"]["loop_sizes"][spatial_dim] // sp_val,
                    "index": 0,
                }
                for dimname, dimval in temp_info["dims"].items():
                    obj[f"mem_{dimname}"] = my_for_loops[len(my_for_loops) - 1][f"mem_{dimname}"]
                my_for_loops.append(obj)

        temp_info["my_for_loops"] = my_for_loops
        temp_info["for_loops_fullname_set"] = set([mlp["fullname"] for mlp in my_for_loops])
        temp_info["for_loops_lens"] = {
            name: len([val for val in my_for_loops if val["name"] == name])
            for name in set([mlp["name"] for mlp in my_for_loops])
        }
        temp_info["tilable_dimensions"] = set(
            [
                mfl["fullname"]
                for mfl in my_for_loops
                if mfl["index"] > 0 or temp_info["for_loops_lens"][mfl["name"]] == 1
            ]
        )
        tiling_sizes = dict()
        current_dims = copy.deepcopy(temp_info["attrs"]["loop_sizes"])
        for fl in my_for_loops:
            if fl["fullname"] in temp_info["tilable_dimensions"]:
                for dimtrad in set(
                    [
                        temp_info["dims"][k]["relevant_loops"][fl["name"]]["mapping"]
                        for k, v in temp_info["dims"].items()
                        if fl["name"] in temp_info["dims"][k]["relevant_loops"]
                    ]
                ):
                    fl_fullname = dimtrad if fl["index"] == 0 else f'{dimtrad}_{fl["index"]}'
                    size_ = current_dims[dimtrad] / fl["size"]
                    tiling_sizes[fl_fullname] = size_
                    current_dims[dimtrad] = size_

        temp_info["tiling_sizes"] = tiling_sizes
        temp_info["for_loops_name_set"] = set([lp["name"] for lp in my_for_loops])
        temp_info["default_mem"] = {operand: "dram" for operand in self.operands}
        temp_info["operands"] = self.operands
        temp_info["input_operands"] = self.input_operands
        temp_info["padded_dims"] = self.padded_dims
        temp_info["inp_dim_mapping"] = self.inp_dim_mapping
        # added
        temp_info["ordered_relevant_loops"] = self.ordered_relevant_loops
        temp_info["kernel_loops"] = set(
            [
                lp["fullname"]
                for idx, lp in enumerate(my_for_loops)
                if all(
                    [
                        lp[f"mem_{op_}"]==temp_info['ordered_operand_memories'][op_][::-1][0]
                        for op_ in temp_info['operands']
                    ]
                )
            ]
        )
        temp_info["kernel_loops_dict_names_indexes"] = {
            fullname: [
                {"name": fl["name"], "index": fl["index"]}
                for fl in my_for_loops
                if fl["fullname"] == fullname
            ][0]
            for fullname in temp_info["kernel_loops"]
        }
        temp_info["software_loops"] = (
            temp_info["for_loops_fullname_set"] - temp_info["kernel_loops"]
        )
        temp_info["onesizeddims"] = set(
            [k for k, v in temp_info["attrs"]["loop_sizes"].items() if v == 1]
        )
        temp_info["temp_loops"] = [
            {**fl, "idx": idx}
            for idx, fl in enumerate(my_for_loops)
            if fl["fullname"] in temp_info["software_loops"]
        ]
        temp_info["last_movements"] = {}
        temp_info["kernel_loops_lps"] = [
            (
                k,
                np.prod(
                    [
                        kl["size"]
                        for kl in temp_info["my_for_loops"]
                        if kl["name"] == k and kl["fullname"] in temp_info["kernel_loops"]
                    ]
                ),
            )
            for k in temp_info["kernel_loops"] & temp_info["for_loops_name_set"]
        ]
        temp_info["copy_out"] = False
        temp_info["print_debug"] = False
        temp_info["profiling"] = self.profiling["nothing"]
        temp_info["sw_for_loops"] = [
            lp for lp in temp_info["my_for_loops"] if lp["fullname"] in temp_info["software_loops"]
        ] + [
            [lp for lp in temp_info["my_for_loops"] if lp["fullname"] in temp_info["kernel_loops"]][
                0
            ]
        ]
        temp_info["sw_for_loops_dict"] = {
            lp["fullname"]: {**lp, "idx_loops": idx}
            for idx, lp in enumerate(temp_info["sw_for_loops"])
        }
        temp_info["for_loop_mem_transfers"] = {
            sl["fullname"]: [
                operand
                for operand in temp_info["operands"]
                if (idx == 0 and sl[f"mem_{operand}"] != temp_info["default_mem"][operand])
                or (
                    idx > 0
                    and sl[f"mem_{operand}"] != temp_info["sw_for_loops"][idx - 1][f"mem_{operand}"]
                )
            ]
            for idx, sl in enumerate(temp_info["sw_for_loops"])
        }
        temp_info["last_movements"] = {operand: 0 for operand in self.operands}
        temp_info["size_loops_mem"] = {
            operand: {
                rel_dim: {
                    temp_info["default_mem"][operand]: temp_info["attrs"]["loop_sizes"][
                        rel_dim
                        if self.type == "depthwise_conv_2d"
                        or operand not in temp_info["input_operands"]
                        else temp_info["inp_dim_mapping"][rel_dim]
                    ],
                    **{
                        temp_info["sw_for_loops_dict"][fl_mem_t_fn][f"mem_{operand}"]: int(
                            temp_info["attrs"]["loop_sizes"][
                                rel_dim
                                if self.type == "depthwise_conv_2d"
                                or operand not in temp_info["input_operands"]
                                else temp_info["inp_dim_mapping"][rel_dim]
                            ]
                            / np.prod(
                                []
                                + [
                                    tilp["size"]
                                    for tilp in temp_info["sw_for_loops"][
                                        : temp_info["sw_for_loops_dict"][fl_mem_t_fn]["idx_loops"]
                                    ]
                                    if tilp["name"] == rel_dim
                                ]
                            )
                        )
                        for fl_mem_t_fn, ops in temp_info["for_loop_mem_transfers"].items()
                        if operand in ops
                    },
                }
                for rel_dim in temp_info["ordered_relevant_loops"][operand]
            }
            for operand in temp_info["operands"]
        }
        """
        #useless because there's always a one sized loop in case no transfers happens
        for operand in operands:
            for rel_dim in temp_info["ordered_relevant_loops"][operand]:
                for mem in set(temp_info['ordered_operand_memories'][operand])-set(temp_info["size_loops_mem"][operand][rel_dim].keys()):
                    temp_info["size_loops_mem"][operand][rel_dim][mem]=1
                    if rel_dim in self.mapping[temp_info["layer_name"]]["spatial_mapping"].keys():
                        temp_info["size_loops_mem"][operand][rel_dim][mem]*=self.mapping[temp_info["layer_name"]]["spatial_mapping"][rel_dim]
                    if operand in input_operands and inp_dim_mapping[rel_dim] in self.attrs['strides'].keys():
                        temp_info["size_loops_mem"][operand][rel_dim][mem]*=self.attrs['strides'][inp_dim_mapping[rel_dim]]
        """
        self.temp_data = temp_info

    def get_code(self):
        try:
            # skip if not supported for testing purposes
            if len(self.workload.keys()) == 0:
                return dory_gap9_compiler(self.mod)[1], False, 0, None
            self.template_data()
            temp = Template(filename="./templates/gap9_template.c")
            return temp.render(**self.temp_data), False, self.latency, self.cme
        except:
            return exceptions.html_error_template().render(), True, self.latency, self.cme


def generate_code(arch_def, mod, std_cost_model: bool = False, temp_mapping: bool=False, forced_temporal_mapping: list[tuple[str,int]]=[],
                    memory_sizes: dict[str,int]={},profiling: bool = False, lpf_limit: int = 6):
    parser = ZigZagWorkloadParser(arch_def, mod, memory_sizes, lpf_limit)
    parser.visit()
    parser.zigzag(temp_mapping, std_cost_model, forced_temporal_mapping)
    code, error_codegen, latency, cme = parser.get_code()
    if profiling:
        import hashlib

        layer = hashlib.sha256(str(mod).encode("UTF-8")).hexdigest()
        tm_shape = f"\nO cycles {cme.temporal_mapping.mapping_dic_stationary['O']}\nI cycles {cme.temporal_mapping.mapping_dic_stationary['I']}"
        cme_data = f"\ntransfer_calls_per_time_from_to_l2 {cme.transfer_calls_per_time_from_to_l2}\nrelmap {cme.relmap}\nmultiplicity_l2{cme.multiplicity_l2}\nmultiplicity_rel_L2{cme.multiplicity_rel_L2}"
        if not std_cost_model:
            tm_data = f"and tm data like\ntotal cycles {cme.temporal_mapping.total_cycle}\ncontrib {cme.temporal_mapping.contrib}\nspatial_mapping_sizes {cme.temporal_mapping.spatial_mapping_sizes}"
        else:
            tm_data = f"and tm data like\ntotal cycles {cme.temporal_mapping.total_cycle}"
        with open(
            f"./outputs/zigzag-data/{mod.attrs.global_symbol}_data_{layer}.txt", "w"
        ) as fw:
            fw.write(
                f"Latency for {layer} is {latency}\nwith latency divided like {cme.__jsonrepr__()['outputs']['latency']}\nworkload is {cme.layer.loop_dim_size}\n{tm_data} {tm_shape} {cme_data}"
            )

    return code, error_codegen


def codegen(mod: tvm.ir.IRModule):
    device_name = mod.attrs.global_symbol.split("_")[1]
    code, error_codegen = generate_code(device_name, mod)
    with open(
        f'./outputs/{mod.attrs.global_symbol}.{"html" if error_codegen else "c"}',
        "wb" if error_codegen else "w",
    ) as fw:
        fw.write(code)
    if error_codegen:
        raise Exception("Couldn't generate output")
    return code