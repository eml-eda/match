    

import copy

from match.dim.dim import MatchTiledDim
from match.schedule.block import MatchBlock
from match.schedule.loop import MatchLoop
from match.schedule.mem_transfer import MatchMemTransfer
from match.schedule.schedule import MatchSchedule
from match.tensor.tensor import MatchTensorTile


class ZigZagTransformMappingToSchedule:

    def __init__(self,match_node,mem_hierarchy,mem_hierarchy_dict,workload,zigzag_parser,cme,
                 zigzag_operands, zigzag_temporal_mapping, spatial_mapping, zigzag_operands_to_tensors,
                 platform_memories):
        self.match_node = match_node
        self.mem_hierarchy = mem_hierarchy
        self.mem_hierarchy_dict = mem_hierarchy_dict
        self.workload = workload
        self.zigzag_parser = zigzag_parser
        self.cme = cme
        self.zigzag_operands = zigzag_operands
        self.zigzag_temporal_mapping = zigzag_temporal_mapping
        self.spatial_mapping = spatial_mapping
        self.mem_hierarchy = mem_hierarchy
        self.mem_hierarchy_dict = mem_hierarchy_dict
        self.mem_name = dict()
        self.zigzag_operands_to_tensors = zigzag_operands_to_tensors
        self.platform_memories = platform_memories
        self.temporal_mapping = []
        
    def generate_memories_names(self):
        mem_op_to_layer_op = self.cme.mem_op_to_layer_op
        for mem_op, mems_all_levels in self.cme.accelerator.cores[0].mem_hierarchy_dict.items():
            layer_op = mem_op_to_layer_op[mem_op]
            self.mem_name[layer_op] = []
            for mem_a_level in mems_all_levels:
                self.mem_name[layer_op].append(mem_a_level.name)

    def add_spatial_mapping_to_temporal_mapping(self):
        # reverse it and add spatial dimensions
        self.temporal_mapping=self.temporal_mapping[::-1]
        for (spatial_dim,spatial_val) in self.spatial_mapping.values():
            for idxox in range(len(self.temporal_mapping)):
                if self.temporal_mapping[idxox]["name"] == spatial_dim:
                    self.temporal_mapping[idxox]["index"] += 1
                    self.temporal_mapping[idxox][
                        "fullname"
                    ] = f'{spatial_dim}_{self.temporal_mapping[idxox]["index"]}'
            obj = {
                "name": spatial_dim,
                "fullname": spatial_dim,
                "size": spatial_val,
                "new_size": spatial_val,
                "index": 0,
            }
            for operand in self.zigzag_operands:
                obj[f"mem_{operand}"] = self.temporal_mapping[len(self.temporal_mapping) - 1][f"mem_{operand}"]
            self.temporal_mapping.append(obj)
    
    def constraint_on_tiling_output_at_last(self):
        for idx in reversed(range(len(self.temporal_mapping))):
            for op in set(self.zigzag_operands)-set("O"):
                if self.temporal_mapping[idx][f"mem_{op}"]!=self.mem_name[op][0] and self.temporal_mapping[idx]["mem_O"]==self.mem_name["O"][0]:
                    self.temporal_mapping[idx]["mem_O"]=self.mem_name["O"][1 if len(self.mem_name["O"])>1 else 0]
                    break

    def apply_constraints_to_temporal_mapping(self):
        self.constraint_on_tiling_output_at_last()

    def adjust_temporal_mapping(self):
        new_temporal_mapping = []
        dim_step = copy.deepcopy(self.workload[1]["loop_dim_size"])
        for idx,t_map in enumerate(self.temporal_mapping):
            dim_step[t_map["name"]] /= t_map["size"]
            t_map["step"] = dim_step[t_map["name"]]
            if t_map["size"] > 1 or idx==0:
                if idx>0:
                    curr_t_map = {k: v for k, v in t_map.items() if k not in ("fullname","index","step","new_size")}
                    last_t_map = {k: v for k, v in new_temporal_mapping[-1].items() if k not in ("fullname","index","step","new_size")}
                    if curr_t_map == last_t_map:
                        t_map["new_size"] *= new_temporal_mapping[-1]["new_size"]
                        new_temporal_mapping[-1] = t_map
                    else:
                        new_temporal_mapping.append(t_map)
                else:
                    new_temporal_mapping.append(t_map)
        self.temporal_mapping = new_temporal_mapping

    def declare_temporal_mapping(self):
        for layer_op, tm_layer_levels in self.zigzag_temporal_mapping.items():
            layerfound = []
            for idx, levels in enumerate(tm_layer_levels):
                for loop_name, loop_size in levels:
                    nameidx = sum([loop_name == el for el in layerfound])
                    fullname = f"{loop_name}_{nameidx}" if nameidx > 0 else loop_name
                    layerfound.append(loop_name)
                    if fullname not in [el["fullname"] for el in self.temporal_mapping]:
                        self.temporal_mapping.append(
                            {
                                "name": loop_name,
                                "index": nameidx,
                                "fullname": fullname,
                                "size": loop_size,
                                "new_size": loop_size,
                                f"mem_{layer_op}": self.mem_name[layer_op][idx],
                            }
                        )
                    else:
                        self.temporal_mapping[[el["fullname"] for el in self.temporal_mapping].index(fullname)][
                            f"mem_{layer_op}"
                        ] = self.mem_name[layer_op][idx]

    def generate_temporal_mapping(self):
        self.declare_temporal_mapping()
        self.add_spatial_mapping_to_temporal_mapping()
        self.apply_constraints_to_temporal_mapping()
        self.adjust_temporal_mapping()


    def add_loads_immediately_for_constants_to_schedule(self):
        # ZigZag expects all the constants to be loaded immediately to the inner memory of weights...
        if len(self.mem_hierarchy["const"])>1:
            for const_tensor in self.schedule.tensors.values():
                if const_tensor.tensor_type=="const":
                    if const_tensor!=self.zigzag_parser.w_tensor:
                        self.schedule.blocks[0].loops[0].mem_transfers.append(
                            MatchMemTransfer(
                                tensor=const_tensor,
                                top_mem=self.mem_hierarchy["const"][-1].name,
                                mem=self.mem_hierarchy["const"][0].name,
                                sw_controlled=self.mem_hierarchy["const"][0].sw_controlled
                            )
                        )
    def add_tensor_tiles_to_schedule(self):
        for tensor in self.schedule.tensors.values():
            self.schedule.tensor_tiles[tensor.name] = [MatchTensorTile(tensor=tensor,
                                            tiled_dims=[MatchTiledDim(dim=dim, size=dim.size, max_size=dim.max_size) for dim in tensor.dims]) for mem in self.platform_memories[::-1]]
            t_type = tensor.tensor_type
            if t_type=="const" and tensor!=self.zigzag_parser.w_tensor:
                continue
            for mem_idx,mem_inst in enumerate(self.platform_memories[::-1]):
                # is a memory of the tensor and its not the top memory so we can get the tiling size of it
                if mem_inst.name in [mem_inst_.name for mem_inst_ in self.mem_hierarchy[t_type]] \
                    and mem_inst.name!=self.mem_hierarchy[t_type][-1].name:
                    steps = {dim.name:dim.size for dim in self.match_node.dims.values()}
                    for loop in self.schedule.blocks[0].loops:
                        if any([mem_trans.tensor==tensor \
                                and mem_trans.mem==mem_inst.name for mem_trans in loop.mem_transfers]):
                            for dim_idx,dim in enumerate(tensor.dims):
                                if dim.name=="default" and dim.size==1:
                                    continue
                                if dim.dim_dependency:
                                    new_size = 0
                                    for ind_dim,mult in dim.dim_dependency.size_dependencies.items():
                                        new_size += (mult*(ind_dim if not hasattr(ind_dim,"name") else steps[ind_dim.name]))
                                    new_size = int(new_size)
                                    self.schedule.tensor_tiles[tensor.name][mem_idx].tiled_dims[dim_idx].max_size = new_size
                                    if new_size>dim.size:
                                        new_size = dim.size
                                    self.schedule.tensor_tiles[tensor.name][mem_idx].tiled_dims[dim_idx].size = new_size
                                else:
                                    self.schedule.tensor_tiles[tensor.name][mem_idx].tiled_dims[dim_idx].size = int(steps[dim.name])
                                    self.schedule.tensor_tiles[tensor.name][mem_idx].tiled_dims[dim_idx].max_size = int(steps[dim.name])
                        steps[loop.dim.name] = loop.step
    
    def temporal_mapping_to_schedule(self):
        top_memories = {op:self.mem_name[op][-1] for op in self.zigzag_operands}
        self.schedule = MatchSchedule(
            [
                MatchBlock(
                    [
                        MatchLoop(
                            name=tm["fullname"],
                            dim=self.zigzag_parser.get_dim_name_by_name(tm["name"]),
                            size=tm["new_size"],
                            step=tm["step"],
                            mem_transfers=[
                                MatchMemTransfer(
                                    tensor= self.zigzag_operands_to_tensors[op_type],
                                    top_mem=top_memories[op_type] if idx==0 else self.temporal_mapping[idx-1][f"mem_{op_type}"],
                                    mem=tm[f"mem_{op_type}"],
                                    sw_controlled=self.mem_hierarchy_dict[tm[f"mem_{op_type}"]],
                                )
                                for op_type in self.zigzag_operands
                                if (idx==0 and tm[f"mem_{op_type}"]!=top_memories[op_type]) or (idx>0 and tm[f"mem_{op_type}"]!=self.temporal_mapping[idx-1][f"mem_{op_type}"]) 
                            ],
                        ) for idx,tm in enumerate(self.temporal_mapping)
                    ],
                    backend="ZigZag"
                )
            ],
            # ZigZag schedule shouldnt use intermediate tensors
            tensors={tens_name:tens for tens_name,tens in self.match_node.tensors.items() if tens.tensor_type!="intermediate" and len(tens.dims)>0},
            tensor_tiles=dict(),
            buffers=[],
        )

    def transform_schedule(self):
        self.generate_memories_names()
        self.generate_temporal_mapping()
        self.temporal_mapping_to_schedule()
        self.add_loads_immediately_for_constants_to_schedule()
        self.add_tensor_tiles_to_schedule()
    
    def get_schedule(self):
        self.transform_schedule()
        return self.schedule