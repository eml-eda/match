from match.target.cost_model import ZigZagMatchCostModel
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



class Gap9ClusterCostModel(ZigZagMatchCostModel):
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        super(Gap9ClusterCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access)
    
    def def_transfer_cost(self):
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
    
    def def_innermost_loops_cost(self):
        def _floor(ch, N):
            return floor((ch + N - 1) / N)
        latency=0
        ch_in = self.dmaconfstruct[self.input_operands[0]]['len_1d_copy']
        ch_out = self.dmaconfstruct['O']['len_1d_copy']
        kernel_size_x = self.loop_sizes['FX']
        kernel_size_y = self.loop_sizes['FY']
        output_shape=[1,self.dmaconfstruct['O']['len_1d_copy'],self.dmaconfstruct['O']['num_2d_copies'],self.dmaconfstruct['O']['num_1d_copies']]
        strides=self.layer_data.strides
        padding=self.layer_data.padding
        if self.layer_data.specific_pattern in ["conv2d","pointwise_conv2d"]:
            iterations = _floor(int(output_shape[2]*strides[0]), 8)* _floor(int(output_shape[3]*strides[1]), 2) * _floor(int(ch_out), 4)
            im2col = kernel_size_x * kernel_size_y * ch_in * 2
            matmul = (5 + _floor(kernel_size_x * kernel_size_y * ch_in, 4) * (6 + 8) + 10)
            latency += iterations * (im2col + matmul)
        elif self.layer_data.specific_pattern in ['depthwise_conv2d','depthwise_conv2d_less_4']:
            # more accurate model of depthwise generic
            input_tile_dim = (ch_out, int(output_shape[2]*strides[0]), int(output_shape[3]*strides[1]))    # (input_height, input_width, input_channels)
            filter_dim = (ch_out, kernel_size_y, kernel_size_x)          # (filter_channels, filter_height, filter_width)
            output_tile_dim = (ch_out, output_shape[2], output_shape[3])   # (output_channels, output_height, output_width)
            padding = (padding["IY"][0], padding["IY"][1], padding["IX"][0], padding["IX"][1])           # (padding_top, padding_bottom, padding_left, padding_right)
            stride = (strides[0], strides[1])                  # (stride_height, stride_width)
            latency = depthwise_generic_latency(input_tile_dim, filter_dim, output_tile_dim, padding, stride)
            # easier model
            #latency = 4 * _floor(ch_out, 8)  * _floor(output_shape[3]*strides[1],4) * kernel_size_x * kernel_size_y * int(output_shape[2]*strides[0])
        elif self.layer_data.specific_pattern=='dense':
            latency += _floor(ch_in, 2) * _floor(ch_out, 4)
        else:
            latency += _floor(ch_in, 2) * _floor(ch_out, 4)
        return latency