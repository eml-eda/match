import copy
from math import ceil
from typing import Any, Dict, List

import numpy as np
import tvm
from match.codegen.layer_data import LayerData

from match.target.exec_module import ExecModule

REQUIRED_HW_DEPENDENT_PARAMS=("weights")

class TemplateDataGenerator:
    def __init__(self,mod:tvm.ir.IRModule,temporal_mapping:List=[],
            layer_data:LayerData=None,exec_module:ExecModule=None,pattern_name:str=""):
        self.mod=mod
        self.temporal_mapping=temporal_mapping
        self.layer_data=layer_data
        self.exec_module=exec_module
        self.pattern_name=pattern_name
        self.template_data=dict()
        self.template_data["debug_level"]=["start_codegen","end_codegen"]
            
    def generate_hw_dependent_template_data(self):
        hw_dependent_template_data = dict()
        hw_dependent_template_data["weights_and_constants"] = self.exec_module.weights_and_constants(self.pattern_name,self.layer_data,self.layer_data.layer_arguments)
        hw_dependent_template_data["mem_apis"] = self.exec_module.match_mem_apis(pattern_name=self.pattern_name)
        hw_dependent_template_data["comp_apis"] = self.exec_module.match_comp_apis(pattern_name=self.pattern_name)
        hw_dependent_template_data["platform_apis"] = self.exec_module.match_platform_apis(pattern_name=self.pattern_name)
        hw_dependent_template_data["sync_apis"] = self.exec_module.match_sync_apis(pattern_name=self.pattern_name)
        hw_dependent_template_data["types"] = self.exec_module.match_types(pattern_name=self.pattern_name)
        hw_dependent_template_data["include_list"] = self.exec_module.match_include_list(pattern_name=self.pattern_name)
        hw_dependent_template_data["kernel_params"] = self.exec_module.additional_kernel_parameters(pattern_name=self.pattern_name)
        hw_dependent_template_data["ordered_operand_memories"] = self.exec_module.operand_memories(self.layer_data.operands)
        self.template_data={**self.template_data,**hw_dependent_template_data}

    def generate_general_template_data(self):
        general_template_data=dict()
        ## layer data
        general_template_data["layer_data"] = self.layer_data
        general_template_data["operands"] = self.layer_data.operands
        general_template_data["input_operands"] = self.layer_data.input_operands
        general_template_data["padded_dims"] = self.layer_data.padded_dims
        general_template_data["input_dim_mapping"] = self.layer_data.input_dim_mapping
        general_template_data["ordered_relevant_loops"] = self.layer_data.ordered_relevant_loops
        general_template_data["pattern_operations"]=self.layer_data.pattern_operations
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
        general_template_data["pattern_name"] = self.pattern_name

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
        tiling_sizes = {operand:dict() for operand in general_template_data["operands"]}
        tiled_dimensions = set(
            [
                mfl["fullname"]
                for mfl in self.temporal_mapping
                if mfl["index"] > 0 and (mfl["fullname"] in general_template_data["sw_for_loops_dict"])
            ]
        )
        # we don't need "virtual" tiles of size 1 I think
        #for fl in self.temporal_mapping:
        #    if fl["index"]==0:
        #        for operand in [op for op in general_template_data["operands"] if fl["name"] in general_template_data["ordered_relevant_loops"][op]]:
        #            dim_name=fl["name"] if operand not in general_template_data["input_operands"] else general_template_data["input_dim_mapping"][fl["name"]]
        #            tiling_sizes[operand][dim_name]={"name":fl["name"],"size":1,"index":fl["index"]}
        # TODO: calc overlap
        ##{(attrs['loop_sizes'][ordim]+attrs['loop_sizes'][trdim['partial_relevancy']]-1-(attrs['loop_sizes'][trdim['mapping']]//attrs['strides'][trdim['mapping']]))//2};
        general_template_data["overlaps"]=copy.deepcopy(self.layer_data.padding)

        current_dims = {operand:copy.deepcopy(general_template_data["layer_attrs"]["loop_sizes"]) for operand in general_template_data["operands"]}

        # DEPTHWISE
        if "nn.conv2d_depthwise" in general_template_data["layer_data"].layer_attrs and general_template_data["layer_data"].layer_attrs["nn.conv2d_depthwise"]:
            for input_op in general_template_data["input_operands"]:
                current_dims[input_op]["C"]=current_dims[input_op]["K"]
        # PADS
        for input_op in general_template_data["input_operands"]:
            for overlapped_dim,overlapped_val in general_template_data["overlaps"].items():
                current_dims[input_op][overlapped_dim]+=max(overlapped_val)
        #print(f"For {general_template_data['func_name']} Dims {current_dims} overlap {general_template_data['overlaps']}\n\n")
        for fl in general_template_data["sw_for_loops"]:
            if fl["fullname"] in tiled_dimensions:
                for operand in [op for op in general_template_data["operands"] if fl["name"] in general_template_data["ordered_relevant_loops"][op]]:
                    dim_name=fl["name"] if operand not in general_template_data["input_operands"] else general_template_data["input_dim_mapping"][fl["name"]]
                    fl_fullname = dim_name if fl["index"] == 0 else f'{dim_name}_{fl["index"]}'
                    size_ = current_dims[operand][dim_name] / fl["size"]
                    tiling_sizes[operand][fl_fullname] = {"name":fl["name"],"size":size_,"index":fl["index"]}
                    current_dims[operand][dim_name] = size_
        
        #print(f"For {general_template_data['func_name']} tiling sizes {tiling_sizes}\n\n")

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
        general_template_data["last_movements"] = {operand: -1 for operand in self.layer_data.operands}
        general_template_data["size_loops_mem"] = {
            operand: {
                rel_dim: {
                    general_template_data["default_mem"][operand]: general_template_data["layer_attrs"]["loop_sizes"][
                        rel_dim
                        if "nn.conv2d" in self.layer_data.pattern_operations and self.layer_data.layer_attrs["nn.conv2d_depthwise"]
                        or operand not in general_template_data["input_operands"]
                        else general_template_data["input_dim_mapping"][rel_dim]
                    ],
                    **{
                        general_template_data["sw_for_loops_dict"][fl_mem_t_fn][f"mem_{operand}"]: int(
                            general_template_data["layer_attrs"]["loop_sizes"][
                                rel_dim
                                if ("nn.conv2d" in self.layer_data.pattern_operations and
                                     self.layer_data.layer_attrs["nn.conv2d_depthwise"]
                                     and rel_dim in ["K","C"])
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

        # calc db opportunites
        general_template_data["db_opportunities"]={
            operand:any([p.double_buffering_support for p in self.exec_module.platform_memories]) and general_template_data["sw_for_loops"][::-1][0][f'mem_{operand}']!=\
            general_template_data["sw_for_loops"][0][f'mem_{operand}']
            for operand in general_template_data["operands"]
        }
        # calc size at each level
        #% for init_mem_idx_reldim,init_mem_reldim in enumerate(ordered_relevant_loops[init_mem_op]):
        #% if init_mem_idx_reldim>0:
        #*
        #% endif
        #% if init_mem_op in input_operands and init_mem_reldim in padded_dims:
        #(${size_loops_mem[init_mem_op][init_mem_reldim]['shared_l1']}+2*dim_${init_mem_op}.accel_dim.addsize_${inp_dim_mapping[init_mem_reldim]})
        #% else:
        #${size_loops_mem[init_mem_op][init_mem_reldim]['shared_l1']}
        #% endif
        #% endfor
        #% if init_mem_op=='W' and 'batchnorm' in attrs and attrs['batchnorm']:
        #+8*dim_${init_mem_op}.accel_dim.size_K[shared_l1]
        #% endif
        def c_friendly_npvalue(arr):
            # params: arr is expected to be a numpy version of the value, it should be an array but it may be also just a single value
            if len(arr.shape)>0:
                # this is actually an array and not a single value
                arr=arr.reshape([arr.shape[0]])
                return {
                    "value":f'{{{str(list(arr))[1:len(str(list(arr)))-1]}}}',
                    "shape":f"[{ceil(arr.shape[0])}]"
                }
            else:
                return {
                    "value":str(arr),
                    "shape":"[1]",
                }
        
        general_template_data["size_each_level"]={
            operand:c_friendly_npvalue(
                np.array(
                    [np.prod(
                        [general_template_data["size_loops_mem"][operand][dim][op_mem] + 
                        (np.sum(general_template_data["overlaps"][general_template_data["input_dim_mapping"][dim]]) if operand in general_template_data["input_operands"] and dim in general_template_data["padded_dims"] else 0)
                         for dim in general_template_data["size_loops_mem"][operand].keys()]
                    )
                for op_mem in self.template_data['ordered_operand_memories'][operand]
                ]))
            for operand in general_template_data["operands"]
        }
        #calc_size_at_each_memory_level()
        #calc_db_opportunities()
        #calc_overlap()
        general_template_data["layer_has_padding"]=any([v!=0 for v in np.array([v for v in general_template_data["layer_data"].padding.values()]).flatten().tolist()])
        general_template_data["layer_has_weights"]=self.template_data["weights_and_constants"]["len"]>0
        general_template_data["padding_c_array"]=c_friendly_npvalue(np.array([v for v in general_template_data["layer_data"].padding.values()]).flatten())
        general_template_data["strides_c_array"]=c_friendly_npvalue(np.array([v for v in general_template_data["layer_attrs"]["strides"].values()]).flatten())
        
        self.template_data={**self.template_data,**general_template_data}

    def get_template_data(self):
        return self.template_data