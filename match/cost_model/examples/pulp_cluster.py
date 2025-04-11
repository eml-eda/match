from match.cost_model.zigzag import ZigZagMatchCostModel
from math import prod,ceil,floor

 
def depthwise_generic_latency(
    input_tile_dim, 
    filter_dim, 
    output_tile_dim, 
    padding, 
    stride, 
    num_cores=8,
    t_mem_access=2,
    t_mac=2,
    t_sync=50):
    """
    Estimate the latency (in cycles) for the pulp_nn_depthwise_generic function.
    Parameters:
    - input_tile_dim (tuple): (input_channels, input_height, input_width)
    - filter_dim (tuple): (filter_channels, filter_height, filter_width)
    - output_tile_dim (tuple): (output_channels, output_height, output_width)
    - padding (tuple): (padding_top, padding_bottom, padding_left, padding_right)
    - stride (tuple): (stride_height, stride_width)
    - num_cores (int): Number of processing cores (default is 8)
    - t_mem_access (int): Latency cost for memory access operations (default is 2 cycles)
    - t_mac (int): Latency cost for a single MAC operation (default is 2 cycles)
    - t_sync (int): Latency cost for synchronization (default is 50 cycles)
    Returns:
    - estimated_latency (int): Estimated latency in cycles
    """
    # Unpacking the parameters
    input_channels, input_height, input_width = input_tile_dim
    filter_channels, filter_height, filter_width = filter_dim
    output_channels, output_height, output_width = output_tile_dim
    padding_top, padding_bottom, padding_left, padding_right = padding
    stride_height, stride_width = stride
    # Effective input dimensions after padding
    effective_input_height = input_height + padding_top + padding_bottom
    effective_input_width = input_width + padding_left + padding_right
    # Adjusted dimensions considering the stride
    adjusted_output_height = (effective_input_height - filter_height) // stride_height + 1
    adjusted_output_width = (effective_input_width - filter_width) // stride_width + 1
    
    # Ensure the model uses the provided output dimensions (this could vary depending on specific padding/stride behavior)
    adjusted_output_height = min(adjusted_output_height, output_height)
    adjusted_output_width = min(adjusted_output_width, output_width)
    # Calculate total MAC operations, considering the stride
    mac_operations_per_output_pixel = filter_height * filter_width
    total_output_pixels = adjusted_output_height * adjusted_output_width * output_channels
    total_mac_operations = mac_operations_per_output_pixel * total_output_pixels
    # Memory accesses: consider multiple passes through the data (input, weights, output)
    input_mem_accesses = effective_input_height * effective_input_width * input_channels
    weight_mem_accesses = filter_height * filter_width * filter_channels
    output_mem_accesses = adjusted_output_height * adjusted_output_width * output_channels
    total_mem_accesses = (input_mem_accesses + weight_mem_accesses + output_mem_accesses)
    # Estimating latency
    # Adjust t_mac to account for potential loop unrolling and SIMD operations.
    adjusted_t_mac = t_mac * 0.25  # Empirical adjustment factor
    # Add empirical adjustment factors for memory accesses and synchronization
    adjusted_t_mem_access = t_mem_access * 1.15
    # Total latency components
    total_mac_latency = total_mac_operations * adjusted_t_mac
    total_mem_latency = total_mem_accesses * adjusted_t_mem_access
    total_sync_latency = t_sync * num_cores
    # Sum all latencies
    estimated_latency = total_mac_latency + total_mem_latency + total_sync_latency
    # Empirical correction factor based on observed data
    correction_factor = 1.05  # Adjust this based on more empirical data
    estimated_latency *= correction_factor
    return int(estimated_latency)



class PulpClusterCostModel(ZigZagMatchCostModel):
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        super(PulpClusterCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access,
            has_any_additional_buffer=True
        )
    
    def adjust_temporal_mapping(self, temporal_mapping_dict, operand_list, layer):
        temporal_mapping_dict,valid = super().adjust_temporal_mapping(temporal_mapping_dict, operand_list, layer)
        if valid and "I" in operand_list:
            min_innermost_loops=min([len(temporal_mapping_dict[operand][0]) for operand in operand_list])
            temporal_mapping_dict["I"][1]=temporal_mapping_dict["I"][0][min_innermost_loops:]+temporal_mapping_dict["I"][1]
            temporal_mapping_dict["I"][0]=temporal_mapping_dict["I"][0][:min_innermost_loops]
            return temporal_mapping_dict,valid
        else:
            return temporal_mapping_dict,valid

    def def_transfer_cost(self):
        USE_SIMPLER_MODEL = False
        if USE_SIMPLER_MODEL:
            def get_stride_2_op(operand):
                if operand in ['I','X','Y']:
                    return self.loop_sizes['C' if 'C' in self.size_per_mem_level[operand] else 'K']*self.partial_relevant_loop_sizes['IX']
                elif operand=='W':
                    return self.loop_sizes['C']*self.loop_sizes['FY']*self.loop_sizes['FX']
                elif operand=='O':
                    return self.loop_sizes['K']*self.loop_sizes['OX']
            def get_stride_1_op(operand):
                if operand in ['I','X','Y']:
                    return self.loop_sizes['C' if 'C' in self.size_per_mem_level[operand] else 'K']
                elif operand=='W':
                    return self.loop_sizes['C']
                elif operand=='O':
                    return self.loop_sizes['K']
            def get_num_2d_copies_op(operand):
                if operand in ['I','X','Y']:
                    return self.size_per_mem_level[operand]["OY"][0]
                elif operand=='W':
                    return self.size_per_mem_level["W"]["K"][0] if self.pattern_name!='depthwise_conv_2d' else 1
                elif operand=='O':
                    return self.size_per_mem_level["O"]["OY"][0]
            def get_num_1d_copies_op(operand):
                if operand in ['I','X','Y']:
                    return self.size_per_mem_level[operand]["OX"][0]
                elif operand=='W':
                    return self.loop_sizes['FY']*self.loop_sizes['FX']*self.loop_sizes['C'] if self.pattern_name!='depthwise_conv_2d' else 1
                elif operand=='O':
                    return self.size_per_mem_level["O"]["OX"][0]
            def get_len_1d_copy_op(operand):
                if operand in ['I','X','Y']:
                    return self.size_per_mem_level[operand]['C' if 'C' in self.size_per_mem_level[operand] else 'K'][0]
                elif operand=='W':
                    return self.loop_sizes['C'] if self.pattern_name!='depthwise_conv_2d' else (self.size_per_mem_level["W"]["K"][0])*self.loop_sizes['FY']*self.loop_sizes['FX']
                elif operand=='O':
                    return self.size_per_mem_level["O"]["K"][0]
            
            dmaconfstruct={
                operand:{
                    'hwc_to_cwh':operand=='I' and self.pattern_name=='depthwise_conv_2d',
                    'stride_2d':get_stride_2_op(operand),
                    'stride_1d':get_stride_1_op(operand),
                    'num_2d_copies':get_num_2d_copies_op(operand),
                    'num_1d_copies':get_num_1d_copies_op(operand),
                    'len_1d_copy':get_len_1d_copy_op(operand),
                } for operand in self.operands
            }
            def calc_overhead(operand):
                if dmaconfstruct[operand]['hwc_to_cwh']:
                    return (27*dmaconfstruct[operand]['len_1d_copy'])+1000
                elif dmaconfstruct[operand]['num_2d_copies']==1 and dmaconfstruct[operand]['num_1d_copies']==1:
                    return 100+300
                else:
                    return (27*dmaconfstruct[operand]['num_2d_copies'])+300
                
            overhead_per_op={operand:calc_overhead(operand) for operand in self.operands}

            def calc_total_transfer_cost_per_op(operand):
                if operand=='O':
                    return self.output_transfer_costs[0]+overhead_per_op['O']
                else:
                    return (self.input_transfer_costs[operand][0]*(2 if dmaconfstruct[operand]['hwc_to_cwh'] else 1))+overhead_per_op[operand]
            
            self.dmaconfstruct=dmaconfstruct
            self.overhead_per_op=overhead_per_op
            return {operand:calc_total_transfer_cost_per_op(operand) for operand in self.operands}
        else:
            def calc_transfer_costs(operand):
                BYTES_PER_CYCLE = 8
                API_OVERHEAD = 550
                MATCH_SETUP_OVERHEAD = 920 - API_OVERHEAD
                OVERHEAD_2D_TRANSFER = 1
                OVERHEAD_3D_TRANSFER = 10
                
                TRANS_CYCLES = 0
                if operand in self.input_operands and operand!="W":
                    IN_HEIGHT_L1 = self.size_per_mem_level[operand]["OY"][0]
                    IN_HEIGHT_L2 = self.size_per_mem_level[operand]["OY"][1]

                    IN_WIDTH_L1 = self.size_per_mem_level[operand]["OX"][0]
                    IN_WIDTH_L2 = self.size_per_mem_level[operand]["OX"][1]

                    IN_CHANNELS_L1 = self.size_per_mem_level[operand]['C' if 'C' in self.size_per_mem_level[operand] else 'K'][0]
                    IN_CHANNELS_L2 = self.size_per_mem_level[operand]['C' if 'C' in self.size_per_mem_level[operand] else 'K'][1]

                    # HWC TO CHW
                    if self.pattern_name in ['depthwise_conv2d','depthwise_conv2d_less_4']:
                        BYTES_PER_CYCLE = 1
                        OVERHEAD_BETWEEN_TRANSFERS = 12
                        NUM_TRANSFERS = IN_CHANNELS_L1
                        COST_PER_SINGLE_TRANSFER =  IN_HEIGHT_L1 * IN_WIDTH_L1 / BYTES_PER_CYCLE
                        
                        TRANS_CYCLES = NUM_TRANSFERS * (OVERHEAD_BETWEEN_TRANSFERS + COST_PER_SINGLE_TRANSFER)
                    # 1D transfers
                    elif IN_WIDTH_L1 == IN_WIDTH_L2 and IN_CHANNELS_L1 == IN_CHANNELS_L2:
                        TRANS_CYCLES = IN_HEIGHT_L1 * IN_WIDTH_L1 * IN_CHANNELS_L1 / BYTES_PER_CYCLE
                    # 2D transfer
                    elif IN_CHANNELS_L1 == IN_CHANNELS_L2:
                        TRANS_CYCLES = (IN_HEIGHT_L1 * IN_WIDTH_L1 * IN_CHANNELS_L1 / BYTES_PER_CYCLE) + IN_HEIGHT_L1 * OVERHEAD_2D_TRANSFER
                    # 3D transfer
                    else:
                        TRANS_CYCLES = (((IN_WIDTH_L1 * IN_CHANNELS_L1 / BYTES_PER_CYCLE) + IN_WIDTH_L1 * OVERHEAD_2D_TRANSFER) * IN_HEIGHT_L1) * IN_HEIGHT_L1 * OVERHEAD_3D_TRANSFER
                elif operand=="W":
                    FILTER_HEIGHT = self.loop_sizes["FY"]
                    FILTER_WIDTH = self.loop_sizes["FX"]

                    WEIGHTS_CH_IN = self.loop_sizes["C"]
                    WEIGHTS_CH_OUT = self.size_per_mem_level["W"]["K"][0]
                    # 1D transfer
                    TRANS_CYCLES = WEIGHTS_CH_OUT * WEIGHTS_CH_IN * FILTER_HEIGHT * FILTER_WIDTH / BYTES_PER_CYCLE
                # output
                else:
                    OUT_HEIGHT_L1 = self.size_per_mem_level[operand]["OY"][0]
                    OUT_HEIGHT_L2 = self.size_per_mem_level[operand]["OY"][1]

                    OUT_WIDTH_L1 = self.size_per_mem_level[operand]["OX"][0]
                    OUT_WIDTH_L2 = self.size_per_mem_level[operand]["OX"][1]

                    OUT_CHANNELS_L1 = self.size_per_mem_level[operand]['K'][0]
                    OUT_CHANNELS_L2 = self.size_per_mem_level[operand]['K'][1]
                    # 1D transfers
                    if OUT_WIDTH_L1 == OUT_WIDTH_L2 and OUT_CHANNELS_L1 == OUT_CHANNELS_L2:
                        TRANS_CYCLES = OUT_HEIGHT_L1 * OUT_WIDTH_L1 * OUT_CHANNELS_L1 / BYTES_PER_CYCLE
                    # 2D transfer
                    elif OUT_CHANNELS_L1 == OUT_CHANNELS_L2:
                        TRANS_CYCLES = (OUT_HEIGHT_L1 * OUT_WIDTH_L1 * OUT_CHANNELS_L1 / BYTES_PER_CYCLE) + OUT_HEIGHT_L1 * OVERHEAD_2D_TRANSFER
                    # 3D transfer
                    else:
                        TRANS_CYCLES = (((OUT_WIDTH_L1 * OUT_CHANNELS_L1 / BYTES_PER_CYCLE) + OUT_WIDTH_L1 * OVERHEAD_2D_TRANSFER) * OUT_HEIGHT_L1) + OUT_HEIGHT_L1 * OVERHEAD_3D_TRANSFER
                # add API and MATCH overhead
                return MATCH_SETUP_OVERHEAD + API_OVERHEAD + TRANS_CYCLES
            return {operand:calc_transfer_costs(operand) for operand in self.operands}
        
    def def_innermost_loops_cost(self):
        def _floor(ch, N):
            return floor((ch + N - 1) / N)
        latency=0
        ch_in = self.loop_sizes["C"]
        ch_out = self.size_per_mem_level["O"]["K"][0]
        kernel_size_x = self.loop_sizes['FX']
        kernel_size_y = self.loop_sizes['FY']
        output_shape=[1,ch_out,self.size_per_mem_level["O"]["OY"][0],self.size_per_mem_level["O"]["OX"][0]]
        if self.pattern_name in ["conv2d","pointwise_conv2d"]:
            IS_POINTWISE = self.pattern_name=="pointwise_conv2d"
            # define scalar costs
            COST_SCALAR_MAC = 14
            COST_SCALAR_LOAD = 3
            # iterations in ho_parallel that lead to matmul 8
            NUM_CORES = 8
            INPUT_PATCHES_PER_IM2COL = 2
            iterations = _floor(output_shape[2], NUM_CORES) * _floor(output_shape[3], INPUT_PATCHES_PER_IM2COL)
            # im2col is num patches by patch size
            im2col = kernel_size_x * kernel_size_y * ch_in * INPUT_PATCHES_PER_IM2COL
            # parallelized by 4 over the im2col, 6 loads 8 macs
            matmul = (5 + _floor(kernel_size_x * kernel_size_y * ch_in, 4) * (6 + 8) + 10)
            # im2col parallelized by 4
            leftover_im2col = (kernel_size_x * kernel_size_y * ch_in) % 4
            # for leftover use scalar ops
            leftover_matmul = (5 + leftover_im2col * ((6*COST_SCALAR_LOAD) + (8*COST_SCALAR_MAC)) + 10)
            # ch out parallelized by 4
            leftover_out_ch_channels = ch_out % 4
            # for ch out leftover only 3 loads and 2 macs
            leftover_out_ch_matmul = (5 + _floor(kernel_size_x * kernel_size_y * ch_in, 4) * (3 + 2) + 10)
            # for additional leftover use scalar ops
            leftover_out_ch_matmul_im2col = (5 + leftover_im2col * ((3*COST_SCALAR_LOAD) + (2*COST_SCALAR_MAC)) + 10)
            # leftover of ho_parallel
            leftover_width_hoparallel = output_shape[3] % 2
            # 2 loads and 1 mac
            leftover_width_matmul = (5 + _floor(kernel_size_x * kernel_size_y * ch_in, 4) * (2 + 1) + 10)
            # leftover of width + im2col in hoparallel, use scalar
            leftover_width_matmul_im2col = (5 + leftover_im2col * ((2*COST_SCALAR_LOAD) + (1*COST_SCALAR_MAC)) + 10)
            # NORMAL CONV PUT
            if not IS_POINTWISE:
                # put all together
                latency = iterations * im2col + \
                    iterations * ((_floor(int(ch_out), 4) * (matmul+leftover_matmul)) + (leftover_out_ch_channels * (leftover_out_ch_matmul+leftover_out_ch_matmul_im2col))) +\
                    leftover_width_hoparallel * (ch_out * (leftover_width_matmul + leftover_width_matmul_im2col) )
            else:
                # put all together
                latency = iterations * ((_floor(int(ch_out), 4) * (matmul+leftover_matmul)) + (leftover_out_ch_channels * (leftover_out_ch_matmul+leftover_out_ch_matmul_im2col))) +\
                    iterations * leftover_width_hoparallel * (ch_out * (leftover_width_matmul + leftover_width_matmul_im2col) )

        elif self.pattern_name in ['depthwise_conv2d','depthwise_conv2d_less_4']:
            # define scalar costs
            COST_SCALAR_MAC = 2
            COST_SCALAR_LOAD = 2
            COST_QUANT = 9
            # iterations in ho_parallel that lead to matmul 8
            NUM_CORES = 8
            iterations = _floor(output_shape[1], NUM_CORES) * output_shape[3]
            # parallelized by 4 over the im2col, 6 loads 8 macs
            vec_matmul = (5 + _floor(kernel_size_x * kernel_size_y, 4) * (2*3 + 1) + 10)
            scalar_matmul = (5 + ((kernel_size_x * kernel_size_y) % 4) * ((2*COST_SCALAR_LOAD) + (1*COST_SCALAR_MAC)) + 10)
            im2col = self.size_per_mem_level["I"]["OY"][0] * kernel_size_y
            latency = iterations * (im2col + output_shape[2] * (scalar_matmul + vec_matmul + COST_QUANT))
        elif self.pattern_name=='dense':
            latency += _floor(ch_in, 2) * _floor(ch_out, 4)
        else:
            latency += _floor(ch_in, 2) * _floor(ch_out, 4)
        return latency