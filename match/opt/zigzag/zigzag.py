import copy
from math import floor
from typing import Any, Dict, List

from match.cost_model.zigzag import ZigZagMatchCostModel
from match.dim.dim import MatchTiledDim
from match.node.node import MatchNode
from match.opt.engine import ScheduleEngine
from match.opt.zigzag.architecture import ZigZagSoC
from match.opt.zigzag.parser import MatchNodeToZigZagParser
from match.schedule.block import MatchBlock
from match.schedule.mem_transfer import MatchMemTransfer
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ExecModule
# zigzag imports
from match.target.memory_inst import MemoryInst
from match.tensor.tensor import MatchTensor, MatchTensorTile
from zigzag import api
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.classes.opt.temporal.loma.engine import NoValidLoopOrderingFoundException

DEBUG_MODE_MATCH = False

class ZigZagEngine(ScheduleEngine):
    def __init__(self, target=None, exec_module: ExecModule=None,
                 pattern_name: str="", match_node: MatchNode=None
                 ):
        super(ZigZagEngine, self).__init__(target=target, exec_module=exec_module,
                                           pattern_name=pattern_name, match_node=match_node)
        self.lpf_limit=13
        self.zigzag_temporal_mapping=dict()
        self.zigzag_parser = MatchNodeToZigZagParser(match_node=self.match_node, pattern_name=self.pattern_name)
        self.workload = dict()
        self.mem_name = dict()
        self.temporal_mapping = list()
        self.platform_memories: List[MemoryInst] = self.target.memory_list_for_pt(exec_module=self.exec_module, pattern_name=self.pattern_name)
        self.mem_hierarchy = self.target.memory_hierarchy_for_pt(exec_module=self.exec_module, pattern_name=self.pattern_name)
        self.mem_hierarchy_dict = {mem.name:mem for mem in set([mem_ for k,v in self.mem_hierarchy.items() for mem_ in v])}
        self.cme: ZigZagMatchCostModel = None
        self.zigzag_operands: List[str] = []
        self.zigzag_operands_to_tensors: Dict[str, MatchTensor] = {
            "I": None,
            "X": None,
            "Y": None,
            "W": None,
            "O": None,
        }
        self.spatial_mapping: Dict[str, Dict[str, Any]] = dict()
        self.cost_model = self.exec_module.zigzag_cost_model()
        self.optimal_spatial_mapping = dict()

    def transform_schedule_for_engine(self):
        self.zigzag_parser.parse()
        self.workload = self.zigzag_parser.generate_workload()
    
    def zigzag_set_exec_module(self):
        # self.exec_module.set_match_node(self.match_node)
        # set spatial mapping and other known stuff
        
        self.zigzag_operands = self.zigzag_parser.get_operands()
        self.zigzag_operands_to_tensors = {
            "I":self.zigzag_parser.i_tensor,
            "W":self.zigzag_parser.w_tensor,
            "O":self.zigzag_parser.o_tensor,
            "X":self.zigzag_parser.x_tensor,
            "Y":self.zigzag_parser.y_tensor
        }
        self.exec_module.zigzag_set_optimal_spatial_mapping(match_node=self.match_node,pattern_name = self.pattern_name)
        
        def op_link_name(operand: str="O"):
            if operand=="O":
                return "O"
            elif operand in ["I","X"]:
                return "I1"
            else:
                return "I2"
        
        self.spatial_mapping[self.pattern_name] = {
            "core_allocation": 1,
            "spatial_mapping":  {f"D{opt_idx+1}":(opt_sptmap[0],self.exec_module.limit_spatial_mapping_to(\
                dim_size=self.zigzag_parser.get_dim_name_by_name(opt_sptmap[0]).size,optimal_spat=self.exec_module.get_optimal_spat_size(opt_sptmap[1],self.zigzag_parser.get_dim_name_by_name(opt_sptmap[0]))))\
                                for opt_idx,opt_sptmap in enumerate(self.exec_module.zigzag_optimal_spatial_mapping)},
            "memory_operand_links": {op:op_link_name(op) for op in self.zigzag_operands},
            "unordered_loops": self.zigzag_parser.get_spatially_unrolled_dimensions(),
        }
        # self.exec_module.match_specific_pattern(match_node=self.match_node,pattern_name=self.pattern_name)
        self.optimal_spatial_mapping = self.exec_module.zigzag_optimal_spatial_mapping

    def generate_zigzag_soc(self):
        self.accelerator = self.exec_module.zigzag_architecture(optimal_spatial_mapping=self.optimal_spatial_mapping,
                                                                platform_memories=self.platform_memories,match_node=self.match_node)
        if self.accelerator is None:
            zigzag_soc = ZigZagSoC()
            self.accelerator = zigzag_soc.get_accelerator(optimal_spatial_mapping=self.optimal_spatial_mapping,
                                                          platform_memories=self.platform_memories)

    def temporal_mapping_search(self):
        current_spatial_mapping = self.spatial_mapping
        found_valid_temporal_mapping = False
        while not found_valid_temporal_mapping:
            try:
                print("Looking for temporal mapping with following spatial mapping",current_spatial_mapping)
                self.energy, self.latency, cme = api.get_hardware_performance_zigzag(
                    workload=self.workload,
                    accelerator=self.accelerator,
                    mapping=current_spatial_mapping,
                    opt="latency",
                    dump_filename_pattern=f"tmp/match-layer_?.json",
                    pickle_filename=f"tmp/match-saved_list_of_cmes.pickle",
                    lpf_limit=self.lpf_limit,
                    cost_model_class= self.cost_model
                )
                if hasattr(cme[0][0],"is_tm_valid"):
                    found_valid_temporal_mapping = cme[0][0].is_tm_valid
            except NoValidLoopOrderingFoundException as exc:
                found_valid_temporal_mapping = False
            if not found_valid_temporal_mapping and all([v[1]==1 for v in list(current_spatial_mapping[self.pattern_name]["spatial_mapping"].values())]):
                raise NoValidLoopOrderingFoundException(
                    f"No valid loop ordering was found for layer {self.workload}."
                )
            if not found_valid_temporal_mapping:
                max_spatial_size = max(list(current_spatial_mapping[self.pattern_name]["spatial_mapping"].values()),key= lambda a: a[1])
                curr_ = current_spatial_mapping[self.pattern_name]["spatial_mapping"]
                for dim_ in curr_.keys():
                    dim_spat_size = current_spatial_mapping[self.pattern_name]["spatial_mapping"][dim_]
                    if dim_spat_size==max_spatial_size:
                        current_spatial_mapping[self.pattern_name]["spatial_mapping"][dim_] = (max_spatial_size[0],floor(max_spatial_size[1]/2))
                        break
        self.cme = cme[0][0]

    def generate_schedule(self): 
        self.zigzag_set_exec_module()
        self.generate_zigzag_soc()
        try:
            self.temporal_mapping_search()
        except Exception as exc:
            self.energy=-1
            self.latency=-1
            self.cme=None
            print(f"[ZIGZAG_ENGINE] No valid loop ordering found: {exc}")
            raise Exception(f"[ZIGZAG_ENGINE] No valid loop ordering found: {exc}")
        self.zigzag_temporal_mapping = self.cme.temporal_mapping.mapping_dic_stationary
        if DEBUG_MODE_MATCH:
            print(f"[ZIGZAG_ENGINE] Total node energy = {self.energy} pJ")
            print(f"[ZIGZAG_ENGINE] Total node latency = {self.latency} cycles")
            print("[ZIGZAG_ENGINE] ZigZag Schedule: ")
            print_mapping(self.cme)

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
        for (spatial_dim,spatial_val) in self.spatial_mapping[self.pattern_name]["spatial_mapping"].values():
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
                                if dim.dim_dependency:
                                    new_size = 0
                                    for ind_dim,mult in dim.dim_dependency.dependencies.items():
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