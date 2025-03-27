import copy
from math import floor
from typing import Any, Dict, List

from match.cost_model.zigzag import ZigZagMatchCostModel
from match.dim.dim import MatchTiledDim
from match.node.node import MatchNode
from match.opt.engine import ScheduleEngine
from match.opt.zigzag.architecture import ZigZagSoC
from match.opt.zigzag.parser import MatchNodeToZigZagParser
from match.opt.zigzag.transform import ZigZagTransformMappingToSchedule
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
        self.lpf_limit=11
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
        self.workload[1]["get_match_schedule"] = self.transform_schedule_
        self.workload[1]["exec_module"] = self.exec_module
        self.workload[1]["w_tensor"] = self.zigzag_operands_to_tensors["W"]
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
                self.spatial_mapping = current_spatial_mapping
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

    def transform_schedule_(self, cme):
        transformer = ZigZagTransformMappingToSchedule(
            match_node=self.match_node,
            mem_hierarchy=self.mem_hierarchy,
            mem_hierarchy_dict=self.mem_hierarchy_dict,
            workload=self.workload,
            zigzag_parser=self.zigzag_parser,
            cme=cme,
            zigzag_operands=self.zigzag_operands,
            zigzag_temporal_mapping=cme.temporal_mapping.mapping_dic_stationary,
            spatial_mapping=cme.layer.user_spatial_mapping if cme.layer is not None else self.spatial_mapping[self.pattern_name]["spatial_mapping"],
            zigzag_operands_to_tensors=self.zigzag_operands_to_tensors,
            platform_memories=self.platform_memories,
        )
        self.schedule = transformer.get_schedule()
        return self.schedule
    
    def transform_schedule(self):
        return self.transform_schedule_(self.cme)