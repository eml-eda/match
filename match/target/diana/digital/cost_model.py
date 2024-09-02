from match.target.cost_model import ZigZagMatchCostModel
from math import prod,ceil,floor
from math import prod,ceil


class DigitalAcceleratorCostModel(ZigZagMatchCostModel):
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        super(DigitalAcceleratorCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access)

    def adjust_temporal_mapping(self,temporal_mapping_dict,operand_list,layer):
        """Fix the temporal mapping of a schedule to match the requirements of the platform, the default implementation will
        move loops of the output to permit the computation to happen as soon as the output has been allocated

        Args:
            temporal_mapping_dict (Dict[List[List[Tuple]]]): dictionary containing per each operator the list of memories with the loops assigned
            operand_list (List[Str]): operands used for the specific pattern

        Returns:
            Dict[List[List[Tuple]]]: the new temporal mapping satisfying each constraint
        """
        min_innermost_loops=min([len(temporal_mapping_dict[operand][0]) for operand in operand_list])
        temporal_mapping_dict["O"][1]=temporal_mapping_dict["O"][0][min_innermost_loops:]+temporal_mapping_dict["O"][1]
        temporal_mapping_dict["O"][0]=temporal_mapping_dict["O"][0][:min_innermost_loops]
        # FIX DENSE OUT CHS TO BE 16
        if layer.layer_attrs["match_layer_data"].specific_pattern=="dense":
            temporal_mapping_dict["O"][1]=temporal_mapping_dict["O"][0][1:]+temporal_mapping_dict["O"][1]
            temporal_mapping_dict["O"][0]=temporal_mapping_dict["O"][0][:1]
            #temporal_mapping_dict["I"][1]=temporal_mapping_dict["I"][0][1:]+temporal_mapping_dict["I"][1]
            #temporal_mapping_dict["I"][0]=temporal_mapping_dict["I"][0][:1]
            #temporal_mapping_dict["W"][1]=temporal_mapping_dict["W"][0][1:]+temporal_mapping_dict["W"][1]
            #temporal_mapping_dict["W"][0]=temporal_mapping_dict["W"][0][:1]
        return temporal_mapping_dict,self.is_temporal_mapping_valid(temporal_mapping_dict,layer.layer_attrs["unordered_loops"])

    def def_transfer_cost(self):
        multiplicity_l2 = {
            key: prod([v[1] for v in val[len(val) - 1]])
            for (key, val) in self.temporal_mapping.mapping_dic_stationary.items()
        }
        # diana contrib
        multiplicity_l2 = {
            key: (max(multiplicity_l2.values()) if key == "O" or key == "W" else val)
            for key, val in multiplicity_l2.items()
        }
        tmap = self.temporal_mapping.mapping_dic_stationary
        lsize = self.temporal_mapping.layer_node.loop_dim_size
        relmap = {
            key: {
                "r": val["r"] + sorted([v_[0] for k_, v_ in val["pr"].items()])[::-1],
                "ir": val["ir"],
            }
            for (
                key,
                val,
            ) in self.temporal_mapping.layer_node.operand_loop_dim.items()
        }
        multiplicity_rel_L2 = {
            operand: {
                reldim: prod(
                    [val[1] for val in tmap[operand][len(tmap[operand]) - 1] if val[0] == reldim]
                )
                for reldim in relmap[operand]["r"]
            }
            for operand in self.operands
        }
        for comm in set(relmap["O"]["r"]).intersection(
            set(
                [
                    val
                    for key, dictval in relmap.items()
                    if key not in ["W", "O"]
                    for val in dictval["r"]
                ]
            )
        ):
            multiplicity_rel_L2["O"][comm] = max(
                [dictval[comm] for key, dictval in multiplicity_rel_L2.items() if comm in dictval]
            )

        def get_transfer_calls_per_time_from_to_l2(operand):
            if operand == "W":
                return 1
            len_rel_map_operand = len(relmap[operand]["r"])
            for ind in range(len_rel_map_operand)[::-1]:
                if ind == 0:
                    return 1
                if multiplicity_rel_L2[operand][relmap[operand]["r"][ind]] != 1:
                    return prod(
                        [
                            lsize[relmap[operand]["r"][prod_lp]]
                            / multiplicity_rel_L2[operand][relmap[operand]["r"][prod_lp]]
                            for prod_lp in range(ind)
                        ]
                    )

        transfer_calls_per_time_from_to_l2 = {
            operand: get_transfer_calls_per_time_from_to_l2(operand)
            for operand in self.operands
        }
        
        def input_cost(operand):
            return multiplicity_l2[operand]\
            * (
                self.data_loading_cc_pair_combined_per_op[operand][-1]
                + transfer_calls_per_time_from_to_l2[operand] * 70
            )
        
        def output_cost():
            return multiplicity_l2["O"]\
            * (
                self.data_offloading_cc_pair_combined[-1]
                + transfer_calls_per_time_from_to_l2["O"] * 70
            )
        #breakpoint()
        return {
            operand: input_cost(operand=operand) if operand != "O" else output_cost()
            for operand in self.operands
        }

    
    def def_innermost_loops_cost(self):
        spatial_mapping_sizes = prod(
            [
                dim
                for (
                    key,
                    dim,
                ) in self.spatial_sizes
            ]
        )
        no_pad_size = (
            self.loop_sizes["K"]
            * (
                self.loop_sizes["OY"]
                - 2 * self.layer_data.padding["IY"][0]
            )
            * self.loop_sizes["OX"]
        )
        pad_size = (
            self.loop_sizes["K"]
            * 2
            * self.loop_sizes["OX"]
        )
        contrib = [
            (
                (
                    self.loop_sizes["C"]
                    * (self.loop_sizes["FY"] - pad)
                    * self.loop_sizes["FX"]
                )
                + (
                    self.loop_sizes["C"]
                    * (self.loop_sizes["FY"] - pad)
                    * 2
                )
                + 23
            )
            for pad in range(
                self.layer_data.padding["IY"][0] + 1
            )
        ]
        comp_cost = sum(
            [
                ceil((no_pad_size if pad == 0 else pad_size) / spatial_mapping_sizes) * contrib[pad]
                for pad in range(
                    self.layer_data.padding["IY"][0] + 1
                )
            ]
        )
        return comp_cost/self.computational_iters
    
    def def_overall_execution(self):
        return self.overall_latency_sync()