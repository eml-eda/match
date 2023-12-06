import copy
from math import ceil
from typing import Any, Dict, List

import numpy as np
import tvm
from match.codegen.layer_data import LayerData

from match.hwmodel.hwmodel import HwModel

REQUIRED_HW_DEPENDENT_PARAMS=("weights")

class TemplateDataGenerator:
    def __init__(self,mod:tvm.ir.IRModule,temporal_mapping:List=[],layer_data:LayerData=None,hwmodel:HwModel=None,pattern_name:str=""):
        self.mod=mod
        self.temporal_mapping=temporal_mapping
        self.layer_data=layer_data
        self.hwmodel=hwmodel
        self.pattern_name=pattern_name
        self.template_data=dict()
            
    def generate_hw_dependent_template_data(self):
        hw_dependent_template_data = dict()
        hw_dependent_template_data["weights"] = self.hwmodel.weights_and_constants(self.layer_data.layer_arguments)
        hw_dependent_template_data["apis"] = self.hwmodel.apis_names()
        hw_dependent_template_data["kernel_params"] = self.hwmodel.additional_kernel_parameters()
        hw_dependent_template_data["ordered_operand_memories"] = self.hwmodel.operand_memories(self.layer_data.operands)
        self.template_data={**self.template_data,**hw_dependent_template_data}

    def generate_general_template_data(self):
        general_template_data=dict()
        ## layer data
        general_template_data["operands"] = self.layer_data.operands
        general_template_data["input_operands"] = self.layer_data.input_operands
        general_template_data["padded_dims"] = self.layer_data.padded_dims
        general_template_data["input_dim_mapping"] = self.layer_data.input_dim_mapping
        general_template_data["ordered_relevant_loops"] = self.layer_data.ordered_relevant_loops
        general_template_data["layer_attrs"] = self.layer_data.layer_attrs
        general_template_data["default_mem"] = {operand: self.template_data["ordered_operand_memories"][operand][0] for operand in self.layer_data.operands}
        ## function details
        general_template_data["func_name"] = self.mod.attrs.global_symbol
        general_template_data["func_number"] = int(general_template_data["func_name"].split("_")[::-1][0])
        ## add 1 sized loop if the lower level memories are not used at all
        if any([self.template_data['ordered_operand_memories'][op_][::-1][0]!=self.temporal_mapping[::-1][0][f'mem_{op_}'] for op_ in self.layer_data.operands]):
            one_index=sum([lp['name']=='K' for lp in self.temporal_mapping])
            one_sized_loop_to_add={
                "name":"K",
                "fullname":f"K_{one_index}",
                "size":1,
                "index":one_index
            }
            for op_ in self.layer_data.operands:
                one_sized_loop_to_add[f'mem_{op_}']=self.template_data["ordered_operand_memories"][op_][::-1][0]
            self.temporal_mapping=[one_sized_loop_to_add]+self.temporal_mapping
        
        general_template_data["for_loops"] = self.temporal_mapping
        general_template_data["workload_name"] = self.layer_data.workload_name

        ## get loops that are managed by back-end kernels, so lower level memory ones
        general_template_data["kernel_loops"] = set(
            [
                lp["fullname"]
                for lp in self.temporal_mapping
                if all(
                    [
                        lp[f"mem_{op_}"]==self.template_data["ordered_operand_memories"][op_][::-1][0]
                        for op_ in general_template_data['operands']
                    ]
                )
            ]
        )
        general_template_data["sw_for_loops"] = [
            lp for lp in self.temporal_mapping if lp["fullname"] not in general_template_data["kernel_loops"]
        ] + [
            [lp for lp in self.temporal_mapping if lp["fullname"] in general_template_data["kernel_loops"]][
                0
            ]
        ]
        general_template_data["sw_for_loops_dict"] = {
            lp["fullname"]: {**lp, "idx_loops": idx}
            for idx, lp in enumerate(general_template_data["sw_for_loops"])
        }
        tiling_sizes = dict()
        tiled_dimensions = set(
            [
                mfl["fullname"]
                for mfl in self.temporal_mapping
                if mfl["index"] > 0
            ]
        )
        for fl in self.temporal_mapping:
            if fl["index"]==0:
                for dim_name in set(
                    [
                        fl["name"] if operand not in general_template_data["input_operands"] else general_template_data["input_dim_mapping"][fl["name"]]
                        for operand in general_template_data["operands"]
                        if fl["name"] in general_template_data["ordered_relevant_loops"][operand]
                    ]
                ):
                    tiling_sizes[dim_name]={"name":fl["name"],"size":1,"index":fl["index"]}
        
        current_dims = copy.deepcopy(general_template_data["layer_attrs"]["loop_sizes"])
        for fl in self.temporal_mapping:
            if fl["fullname"] in tiled_dimensions:
                for dim_name in set(
                    [
                        fl["name"] if operand not in general_template_data["input_operands"] else general_template_data["input_dim_mapping"][fl["name"]]
                        for operand in general_template_data["operands"]
                        if fl["name"] in general_template_data["ordered_relevant_loops"][operand]
                    ]
                ):
                    fl_fullname = dim_name if fl["index"] == 0 else f'{dim_name}_{fl["index"]}'
                    size_ = current_dims[dim_name] / fl["size"]
                    tiling_sizes[fl_fullname] = {"name":fl["name"],"size":size_,"index":fl["index"]}
                    current_dims[dim_name] = size_
        general_template_data["tiling_sizes"] = tiling_sizes
        general_template_data["memory_transfers"] = {
            sl["fullname"]: [
                operand
                for operand in general_template_data["operands"]
                if (idx == 0 and sl[f"mem_{operand}"] != general_template_data["default_mem"][operand])
                or (
                    idx > 0
                    and sl[f"mem_{operand}"] != general_template_data["sw_for_loops"][idx - 1][f"mem_{operand}"]
                )
            ]
            for idx, sl in enumerate(general_template_data["sw_for_loops"])
        }
        general_template_data["last_movements"] = {operand: 0 for operand in self.layer_data.operands}
        general_template_data["size_loops_mem"] = {
            operand: {
                rel_dim: {
                    general_template_data["default_mem"][operand]: general_template_data["layer_attrs"]["loop_sizes"][
                        rel_dim
                        if self.layer_data.workload_name == "depthwise_conv_2d"
                        or operand not in general_template_data["input_operands"]
                        else general_template_data["input_dim_mapping"][rel_dim]
                    ],
                    **{
                        general_template_data["sw_for_loops_dict"][fl_mem_t_fn][f"mem_{operand}"]: int(
                            general_template_data["layer_attrs"]["loop_sizes"][
                                rel_dim
                                if self.layer_data.workload_name == "depthwise_conv_2d"
                                or operand not in general_template_data["input_operands"]
                                else general_template_data["input_dim_mapping"][rel_dim]
                            ]
                            / np.prod(
                                []
                                + [
                                    tilp["size"]
                                    for tilp in general_template_data["sw_for_loops"][
                                        : general_template_data["sw_for_loops_dict"][fl_mem_t_fn]["idx_loops"]
                                    ]
                                    if tilp["name"] == rel_dim
                                ]
                            )
                        )
                        for fl_mem_t_fn, ops in general_template_data["memory_transfers"].items()
                        if operand in ops
                    },
                }
                for rel_dim in general_template_data["ordered_relevant_loops"][operand]
            }
            for operand in general_template_data["operands"]
        }

        def calc_overlap():
            ##{(attrs['loop_sizes'][ordim]+attrs['loop_sizes'][trdim['partial_relevancy']]-1-(attrs['loop_sizes'][trdim['mapping']]//attrs['strides'][trdim['mapping']]))//2};
            pass

        calc_overlap()
        self.template_data={**self.template_data,**general_template_data}

    def get_template_data(self):
        return self.template_data