from math import ceil, prod
from typing import Callable
from copy import deepcopy
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.classes.mapping.temporal.temporal_mapping import TemporalMapping
from zigzag.classes.opt.temporal.loma.engine import NoValidLoopOrderingFoundException


class ZigZagMatchCostModel(CostModelEvaluation):
    """MATCH implementation of the cost model that will be used by ZigZag"""

    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
        compute_constants_allocation=True,
        has_any_additional_buffer=False,
    ):
        # MATCH cost model params
        self.MATCH_ITERATION_LATENCY = 300  # TODO: profile MATCH latency
        self.MATCH_EXECUTIONAL_MODEL_ITERATION_LATENCY = 200  # default value
        self.COMPUTE_CONSTANTS_ALLOCATION = compute_constants_allocation
        self.HAS_ANY_ADDITIONAL_BUFFER = has_any_additional_buffer
        self.allocated_buffers = []
        self.max_num_buffers = 0
        self.set_max_num_buffers = False
        temporal_mapping_dict=temporal_mapping.mapping_dic_origin
        operands_=temporal_mapping.operand_list
        constrained_temporal_mapping_dict,valid=self.adjust_temporal_mapping(temporal_mapping_dict,operands_,layer)
        self.layer = layer
        self.accelerator = accelerator
        self.core_id = layer.core_allocation
        self.mem_hierarchy_dict = accelerator.get_core(
            self.core_id
        ).get_memory_hierarchy_dict()
        self.layer_op_to_mem_op = layer.memory_operand_links
        self.mem_op_to_layer_op = dict(
            [(value, key) for key, value in self.layer_op_to_mem_op.items()]
        )
        self.spatial_mapping = spatial_mapping
        constrained_temporal_mapping=TemporalMapping(temporal_mapping_dict=constrained_temporal_mapping_dict,
                                                     layer_node=temporal_mapping.layer_node)
        self.temporal_mapping = constrained_temporal_mapping
        tm_valid = valid
        self.is_tm_valid = tm_valid
        self.run_match_cost_model()
        # check that constants and buffers can actually fit in memory
        if self.COMPUTE_CONSTANTS_ALLOCATION and tm_valid and len(self.allocated_buffers)<self.max_num_buffers:
            self.add_constraints_on_temp_mapping_to_fit_buffers(constrained_temporal_mapping_dict)
        
        super(ZigZagMatchCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=self.temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access)
        
        if self.COMPUTE_CONSTANTS_ALLOCATION:
            self.final_cleanup()

    def is_temporal_mapping_valid(self, temporal_mapping_dict, unordered_loops):
        loops_at_outer_level = [lp[0] for lp in temporal_mapping_dict["O"][1]]
        return sum([lp in loops_at_outer_level for lp in unordered_loops]) == 0

    def adjust_temporal_mapping(self, temporal_mapping_dict, operand_list, layer):
        """Fix the temporal mapping of a schedule to match the requirements of the platform, the default implementation will
        move loops of the output to permit the computation to happen as soon as the output has been allocated

        Args:
            temporal_mapping_dict (Dict[List[List[Tuple]]]): dictionary containing per each operator the list of memories with the loops assigned
            operand_list (List[Str]): operands used for the specific pattern

        Returns:
            Dict[List[List[Tuple]]]: the new temporal mapping satisfying each constraint
        """
        min_innermost_loops = min([len(temporal_mapping_dict[operand][0]) for operand in operand_list])
        temporal_mapping_dict["O"][1] = temporal_mapping_dict["O"][0][min_innermost_loops:] + temporal_mapping_dict["O"][1]
        temporal_mapping_dict["O"][0] = temporal_mapping_dict["O"][0][:min_innermost_loops]
        return temporal_mapping_dict, self.is_temporal_mapping_valid(
            temporal_mapping_dict, layer.layer_attrs["unordered_loops"]
        )

    def set_match_params(self):
        self.temp_mapping = self.temporal_mapping.mapping_dic_origin
        self.loop_sizes = self.layer.loop_dim_size
        self.partial_relevant_loop_sizes = self.layer.pr_loop_dim_size
        self.operands = self.temporal_mapping.operand_list
        self.input_operands = [op for op in self.operands if op != "O"]
        self.operand_loops = self.layer.operand_loop_dim
        self.spatial_sizes = self.spatial_mapping.spatial_loop_dim_size
        self.pattern_name = self.layer.layer_attrs["operator_type"]
        self.match_node = self.layer.layer_attrs["match_node"]
        self.precision = {
            key if "final" not in key else key.split("_")[0]: val
            for key, val in self.layer.operand_precision.items()
            if key + "_final" not in self.layer.operand_precision
        }

    def def_innermost_loops_cost(self):
        """This function computes the cost of each single iteration of the kernel

        Returns:
            number: The cost of each iteration of the inner computation
        """
        return prod([self.loop_iters_per_mem_level[operand][0] for operand in self.operands])

    def calc_innermost_loops_cost(self):
        self.innermost_loops_cost_per_it = self.def_innermost_loops_cost()
        self.computational_cost = self.innermost_loops_cost_per_it * self.computational_iters

    def calc_loop_iters_per_mem_level(self):
        self.loop_iters_per_mem_level = {
            operand: [prod([v[1] for v in val[mem_level]]) for mem_level in range(len(val))]
            for (operand, val) in self.temp_mapping.items()
        }
        self.outermost_loop_iters = {
            operand: prod(
                [
                    self.loop_iters_per_mem_level[operand][idx + 1]
                    for idx in range(len(self.loop_iters_per_mem_level[operand]) - 1)
                ]
            )
            for operand in self.operands
        }
        self.sorted_multiplicities=sorted(set([self.outermost_loop_iters[operand] for operand in self.operands]))
        self.computational_iters=self.sorted_multiplicities[-1]

    def add_constraints_on_temp_mapping_to_fit_buffers(self, constrained_temporal_mapping_dict):
        valid = False
        original_temporal_mapping_dict = deepcopy(constrained_temporal_mapping_dict)
        # print(f"Trying to fit buffers in temporal mapping {constrained_temporal_mapping_dict}")
        for op in ["O"] + sorted([op_ for op_ in self.operands], reverse=True):
            lps_in_inner_level = constrained_temporal_mapping_dict[op][0]
            len_lps_in_inner_level = len(lps_in_inner_level)
            for _ in range(len_lps_in_inner_level - len(self.layer.layer_attrs["unordered_loops"])):
                self.is_tm_valid = True
                lp = constrained_temporal_mapping_dict[op][0].pop()
                constrained_temporal_mapping_dict[op][1] = [lp] + constrained_temporal_mapping_dict[op][1]
                constrained_temporal_mapping_dict_, tm_valid = self.adjust_temporal_mapping(constrained_temporal_mapping_dict, self.operands, self.layer)
                # print(f"Adjusted temporal mapping to: {constrained_temporal_mapping_dict_}")
                if tm_valid:
                    self.temporal_mapping = TemporalMapping(
                        temporal_mapping_dict=constrained_temporal_mapping_dict_,
                        layer_node=self.layer
                    )
                    self.run_match_cost_model()
                    if self.is_tm_valid and len(self.allocated_buffers) == self.max_num_buffers:
                        # print(f"Found valid temporal mapping that fits all the extra-buffers: {constrained_temporal_mapping_dict_}")
                        valid = True
                        break
            if valid:
                break
        if not valid:
            original_temporal_mapping_dict_, tm_valid = self.adjust_temporal_mapping(original_temporal_mapping_dict, self.operands, self.layer)
            if tm_valid:
                self.temporal_mapping = TemporalMapping(
                    temporal_mapping_dict=original_temporal_mapping_dict_,
                    layer_node=self.layer
                )
    
    def calc_sizes_per_mem_level(self):
        self.size_per_mem_level = dict()
        for operand in self.operands:
            self.size_per_mem_level[operand] = {
                dim: [1 for _ in range(len(self.temp_mapping[operand]))]
                for dim in self.layer.operand_dimensionality_order[operand]
            }
            all_dimensions = self.layer.loop_dim_list
            all_dim_sizes = {dim: 1 for dim in all_dimensions}
            for loops in self.spatial_mapping.mapping_dict_origin[operand]:
                for loop_dim, loop_size in loops:
                    if loop_dim in all_dim_sizes:
                        all_dim_sizes[loop_dim] *= loop_size
            for memory_level in range(len(self.temp_mapping[operand])):
                for loop_dim, loop_size in self.temp_mapping[operand][memory_level]:
                    if loop_dim in all_dim_sizes:
                        all_dim_sizes[loop_dim] *= loop_size
                for dim in self.layer.operand_loop_dim[operand]["r"]:
                    self.size_per_mem_level[operand][dim][memory_level] = all_dim_sizes[dim]
                for dim in self.layer.operand_loop_dim[operand]["pr"].keys():
                    self.size_per_mem_level[operand][dim][memory_level] = self.layer.calc_tensor_dim(all_dim_sizes, dim)

    def def_transfer_cost(self):
        """This function computes the cost of an iteration of memory transfer per each operand

        Returns:
            Dict[Str,Number]: Cost of transfer per each iteration for every single operand
        """
        return {
            operand: sum(self.input_transfer_costs[operand])
            if operand in self.input_operands
            else sum(self.output_transfer_costs)
            for operand in self.operands
        }

    def calc_transfer_costs(self):
        self.input_transfer_costs=0 if not hasattr(self,"data_loading_cc_pair_combined_per_op") else self.data_loading_cc_pair_combined_per_op
        self.output_transfer_costs=0 if not hasattr(self,"data_offloading_cc_pair_combined") else self.data_offloading_cc_pair_combined
        self.transfer_costs=self.def_transfer_cost()

    def overall_latency_single_buffer_match_no_comp(self):
        input_overall_transfers = max([self.transfer_costs[operand] * self.outermost_loop_iters[operand] for operand in self.operands if operand!='O']+[0])
        output_overall_transfers = self.transfer_costs["O"] * self.computational_iters

        self.match_overall_latency=self.computational_iters * (self.MATCH_EXECUTIONAL_MODEL_ITERATION_LATENCY + self.MATCH_ITERATION_LATENCY)
        self.match_overall_latency+=input_overall_transfers + output_overall_transfers
        # self.match_overall_latency+=self.computational_cost
    
    def overall_latency_sync(self):
        input_overall_transfers = sum(
            [self.transfer_costs[operand] * self.outermost_loop_iters[operand] for operand in self.operands if operand != "O"]
        )
        output_overall_transfers = self.transfer_costs["O"] * self.computational_iters

        self.match_overall_latency = self.computational_iters * (
            self.MATCH_EXECUTIONAL_MODEL_ITERATION_LATENCY + self.MATCH_ITERATION_LATENCY
        )
        self.match_overall_latency += input_overall_transfers + output_overall_transfers
        self.match_overall_latency += self.computational_cost

    def overall_latency_async(self):
        cycles = 0
        prev_mult_ = 0
        # print(f"Cost model multiplicities {sorted_multiplicities}")
        for idx, mult_ in enumerate(self.sorted_multiplicities):
            cycles += (mult_ - prev_mult_) * (self.MATCH_EXECUTIONAL_MODEL_ITERATION_LATENCY + self.MATCH_ITERATION_LATENCY)
            if idx == 0:
                cycles += max(
                    [0]
                    + [
                        self.transfer_costs[operand]
                        for operand in self.operands
                        if operand != "O" and self.outermost_loop_iters[operand] >= mult_
                    ]
                )
                prev_mult_ = 1
            cycles += (mult_ - prev_mult_) * max(
                [
                    self.innermost_loops_cost_per_it,
                    max(
                        [
                            self.transfer_costs[operand]
                            for operand in self.operands
                            if self.outermost_loop_iters[operand] >= mult_
                        ]
                    ),
                ]
            )
            prev_mult_ = mult_
        self.match_overall_latency = cycles + self.innermost_loops_cost_per_it + self.transfer_costs["O"]

    def def_overall_execution(self):
        self.overall_latency_async()

    def calc_match_overall_latency(self, compute_estimation: bool = False):
        self.set_match_params()
        self.calc_loop_iters_per_mem_level()
        self.calc_sizes_per_mem_level()
        if compute_estimation:
            self.calc_transfer_costs()
            self.calc_innermost_loops_cost()
            self.def_overall_execution()
    
    def check_constants_and_buffer_alloc(self):
        constant_mem_key = "I2"
        if constant_mem_key not in self.mem_hierarchy_dict:
            constant_mem_key = "I1"
        lowest_const_mem = self.mem_hierarchy_dict[constant_mem_key][0]
        mem_bytes = lowest_const_mem.memory_instance.size // 8
        sizes_per_mem_level = self.size_per_mem_level
        
        for operand in self.operands:
            if self.layer.memory_operand_links[operand] in lowest_const_mem.operands:
                mem_bytes -= prod([val[0] for val in sizes_per_mem_level[operand].values()]) * self.precision[operand] // 8

        for w_tensor in self.match_node.const_tensors.values():
            if self.layer.layer_attrs["w_tensor"] is not None and w_tensor != self.layer.layer_attrs["w_tensor"]:
                mem_bytes -= w_tensor.prod_shape_int * w_tensor.dtype.itemsize

        if mem_bytes < 0:
            return False

        var_mem_key = "I1"
        lowest_var_mem = self.mem_hierarchy_dict[var_mem_key][0]
        var_mem_bytes = lowest_var_mem.memory_instance.size // 8
        if lowest_const_mem == lowest_var_mem:
            var_mem_bytes = mem_bytes

        if self.HAS_ANY_ADDITIONAL_BUFFER:
            schedule = self.layer.layer_attrs["get_match_schedule"](self)
            schedule.buffers = []
            self.layer.layer_attrs["exec_module"].set_buffers_for_schedule(
                match_node=self.match_node, schedule=schedule, pattern_name=self.pattern_name, engine="ZigZag"
            )
            new_buffers = []
            if not self.set_max_num_buffers:
                self.max_num_buffers = len(schedule.buffers)
                self.set_max_num_buffers = True
            for buff_tensor in sorted(schedule.buffers, key=lambda buff: (-buff.required, -buff.num_bytes)):
                var_mem_bytes-=buff_tensor.num_bytes
                if var_mem_bytes>=0:
                    new_buffers.append(buff_tensor)
                else:
                    if not buff_tensor.required:
                        # if the buffer is not required we can remove it from the schedule
                        # and print a warning
                        var_mem_bytes+=buff_tensor.num_bytes
                        # print(f"WARNING: Buffer {buff_tensor.name} is not required but it cannot fit in memory, removing it from the schedule")
                    else:
                        # print(f"ERROR: Buffer {buff_tensor.name} is required but it cannot fit in memory, cannot continue")
                        break
            schedule.buffers = new_buffers
            self.allocated_buffers = new_buffers
        
        if var_mem_bytes<0:
            return False

        return True

    def final_cleanup(self):
        self.layer = None
        self.temporal_mapping.layer_node = None
        self.spatial_mapping.layer_node = None
        self.mapping.layer_node = None
        self.mapping_int.layer_node = None
        self.mapping_int.spatial_mapping = None
        # pass

    def run_match_cost_model(self, compute_estimation: bool = False):
        # call user defined latency function
        self.calc_match_overall_latency(compute_estimation=compute_estimation)
        # set overall latency
        if compute_estimation:
            self.latency_total2 = self.match_overall_latency
        # check that constants and buffers can actually fit in memory
        if self.COMPUTE_CONSTANTS_ALLOCATION:
            self.is_tm_valid = self.is_tm_valid and self.check_constants_and_buffer_alloc()
            # add latency penalty if the scheduler could not allocate all the buffers
            if compute_estimation and self.is_tm_valid:
                self.latency_total2 += (self.max_num_buffers - len(self.allocated_buffers)) * 100000000
    def calc_overall_latency(self):
        # use default ZigZag implementation (just to compute some necessary parameters)
        super().calc_overall_latency()
        self.run_match_cost_model(compute_estimation=True)

class ZigZagMatchNoTilingCostModel(ZigZagMatchCostModel):
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        super(ZigZagMatchNoTilingCostModel, self).__init__(
            accelerator=accelerator,
            layer=layer,
            spatial_mapping=spatial_mapping,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access,
        )
        # we consider no cost at all for no tiling schedules
        self.match_overall_latency = 0
        self.energy_total = 0
        self.MAC_energy = 0
        self.mem_energy = 0
        self.latency_total2 = 0
        self.latency_total1 = 0
        self.latency_total0 = 0

    def adjust_temporal_mapping(self, temporal_mapping_dict, operand_list, layer):
        return temporal_mapping_dict, True
