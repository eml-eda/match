from match.cost_model.zigzag import ZigZagMatchCostModel
from math import prod,ceil,floor

import numpy as np
from numpy import prod


def div_and_ceil(a, b):
    return ((a - 1) // b) + 1


def FloorSTE(ch,N):
    return np.floor_divide(ch,N)

def DivAndCeilSTE(a,b):
    return ((a - 1) // b) + 1


def ModuloSTE(a,b):
    return a % b


class Ne16PerfModel:
    INPUT_BUFFER_SHAPE = (5, 5, 16)
    OUTPUT_BUFFER_SHAPE = (3, 3, 32)
    FIFO_LATENCY = 6
    SHIFTER_COUNT = 4
    ADDER_COUNT = 8
    MULTIPLIER_COUNT = 4
    MEMORY_THROUGHPUT = 256  # bits per cycle

    def __init__(self, operation, kernel_shape, depthwise=False, nq_shift=False, nq_bias=False, nq_bits=32, WEIGHTS_BITWIDTH = 8):
        """
        print(f'Latency breakdown:\n'
              f'  - load: {total_load_latency} ({total_load_latency / total_latency:.2%})\n'
              f'  - weight offset: {total_weight_offset_latency} ({total_weight_offset_latency / total_latency:.2%})\n'
              f'  - matrixvec: {total_matrixvec_latency} ({total_matrixvec_latency / total_latency:.2%})\n'
              f'  - update idx: {total_update_idx_latency} ({total_update_idx_latency / total_latency:.2%})\n'
              f'  - normquant: {total_normquant_latency} ({total_normquant_latency / total_latency:.2%})\n'
              f'  - streamout: {total_streamout_latency} ({total_streamout_latency / total_latency:.2%})\n'
              f'TOTAL: {total_latency}')
        """
        self.operation = operation
        self.kernel_shape = kernel_shape
        self.depthwise = depthwise
        self.nq_shift = nq_shift
        self.nq_bias = nq_bias
        self.nq_bits = nq_bits
        self.WEIGHTS_BITWIDTH = WEIGHTS_BITWIDTH
        self.INPUT_BITWIDTH = 8
        self.OUTPUT_BITWIDTH = 8
        self.layer = (
                self.OUTPUT_BUFFER_SHAPE[0],
                self.OUTPUT_BUFFER_SHAPE[1],
                self.OUTPUT_BUFFER_SHAPE[2] if not depthwise else self.INPUT_BUFFER_SHAPE[2],
                self.INPUT_BUFFER_SHAPE[2])

    def set_layer(self, layer):
        self.layer = layer
        return self

    def set_subtile(self, h_out=None, w_out=None, k_out=None, k_in=None):
        #print(f"Setting subtile as {h_out} {w_out} {k_out} {k_in}")
        h_out = h_out if h_out is not None else self.OUTPUT_BUFFER_SHAPE[0]
        w_out = w_out if w_out is not None else self.OUTPUT_BUFFER_SHAPE[1]
        k_out = k_out if k_out is not None else self.OUTPUT_BUFFER_SHAPE[2]
        k_in  = k_in  if k_in  is not None else self.INPUT_BUFFER_SHAPE[2]
        self.INPUT_BUFFER_SHAPE = (h_out + 2, w_out + 2, k_in)
        self.OUTPUT_BUFFER_SHAPE = (h_out, w_out, k_out)

    @property
    def is_3x3(self):
        return self.operation == 'conv' and self.kernel_shape == (3, 3) and not self.depthwise

    @property
    def is_1x1(self):
        return self.operation == 'conv' and self.kernel_shape == (1, 1) and not self.depthwise

    @property
    def is_dw(self):
        return self.operation == 'conv' and self.kernel_shape == (3, 3) and self.depthwise

    @property
    def load_latency(self):
        return 10 + self.OUTPUT_BUFFER_SHAPE[0] * self.OUTPUT_BUFFER_SHAPE[1] * DivAndCeilSTE(self.INPUT_BUFFER_SHAPE[2] * self.INPUT_BITWIDTH, self.MEMORY_THROUGHPUT) if self.is_1x1 \
            else self.FIFO_LATENCY + self.INPUT_BUFFER_SHAPE[0] * self.INPUT_BUFFER_SHAPE[1] * DivAndCeilSTE(self.INPUT_BUFFER_SHAPE[2] * self.INPUT_BITWIDTH, self.MEMORY_THROUGHPUT)

    def weight_offset_latency(self, k):
        return (self.FIFO_LATENCY + k) if self.is_dw else self.FIFO_LATENCY

    def matrixvec_latency(self, k):
        return (self.FIFO_LATENCY + k) if self.is_1x1 else (self.FIFO_LATENCY + k * self.WEIGHTS_BITWIDTH)

    @property
    def update_idx_latency(self):
        return 2

    @property
    def nq_shift_latency(self):
        return 0 if not self.nq_shift else DivAndCeilSTE(self.OUTPUT_BUFFER_SHAPE[2], self.SHIFTER_COUNT)

    def nq_bias_latency(self, k):
        return 0 if not self.nq_bias else 8 + DivAndCeilSTE(k, self.ADDER_COUNT)

    def nq_scale_latency(self, k):
        return 9 + DivAndCeilSTE(k * (self.nq_bits // 8), self.MULTIPLIER_COUNT)

    def normquant_latency(self, k):
        return self.nq_shift_latency + self.nq_scale_latency(k) + self.nq_bias_latency(k)

    @property
    def streamout_latency(self):
        return 3 + self.OUTPUT_BUFFER_SHAPE[0] * self.OUTPUT_BUFFER_SHAPE[1] * DivAndCeilSTE(self.OUTPUT_BUFFER_SHAPE[2] * self.OUTPUT_BITWIDTH, self.MEMORY_THROUGHPUT) + 1  # + end
    
    @property
    def weight_load_latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorSTE(self.layer[2], k_out_body)

        # nothing depends on k_in so no need for remainder
        n_in = DivAndCeilSTE(self.layer[3], self.INPUT_BUFFER_SHAPE[2])

        # depthwise doesn't care about spatial remainder, it just fetches the same
        if self.is_dw:
            return n_out_body*self.weight_offset_latency(k_out_body)
        else:
            return n_out_body*n_in*self.weight_offset_latency(k_out_body)

    @property
    def iter_normquant_latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorSTE(self.layer[2], k_out_body)

        return n_out_body * self.normquant_latency(k_out_body)

    @property
    def out_store_latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorSTE(self.layer[2], k_out_body)

        return n_out_body * self.streamout_latency

    @property
    def update_indexes_latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorSTE(self.layer[2], k_out_body)

        return n_out_body * self.update_idx_latency

    @property
    def input_load_latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorSTE(self.layer[2], k_out_body)
        return n_out_body * self.load_latency
    
    @property
    def computation_iteration_latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorSTE(self.layer[2], k_out_body)

        # nothing depends on k_in so no need for remainder
        n_in = DivAndCeilSTE(self.layer[3], self.INPUT_BUFFER_SHAPE[2])
        
        if self.is_dw:
            return n_out_body * self.matrixvec_latency(k_out_body)
        else:
            return n_out_body * n_in * self.matrixvec_latency(k_out_body)

    @property
    def layer_shape_in(self):
        return (self.layer[0] + self.kernel_shape[0] - 1, self.layer[1] + self.kernel_shape[1] - 1, self.layer[3])

    def dma_latency(self, dma_stall=8, bandwidth=4):
        h_out, w_out, k_out, _ = self.layer
        h_in, w_in, k_in = self.layer_shape_in
        mem = h_in * w_in * k_in + h_out * w_out * k_out + self.kernel_shape[0] * self.kernel_shape[1] * k_out * k_in
        return (mem / bandwidth) * dma_stall
    
    
    @property
    def latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorSTE(self.layer[2], k_out_body)
        k_out_rem = ModuloSTE(self.layer[2], k_out_body)

        # nothing depends on k_in so no need for remainder
        n_in = DivAndCeilSTE(self.layer[3], self.INPUT_BUFFER_SHAPE[2])

        # depthwise doesn't care about spatial remainder, it just fetches the same
        n_spatial = DivAndCeilSTE(self.layer[0], self.OUTPUT_BUFFER_SHAPE[0]) * DivAndCeilSTE(self.layer[1], self.OUTPUT_BUFFER_SHAPE[1])

        if self.is_dw:
            def iteration_latency(k):
                return self.load_latency + self.weight_offset_latency(k) + self.matrixvec_latency(k) + self.update_idx_latency +\
                       self.normquant_latency(k) + self.streamout_latency
        else:
            def iteration_latency(k):
                return n_in * (self.load_latency + self.weight_offset_latency(None) + self.matrixvec_latency(k) + self.update_idx_latency) +\
                       self.normquant_latency(k) + self.streamout_latency

        total_latency = n_spatial * (n_out_body * iteration_latency(k_out_body) + (iteration_latency(k_out_rem) if k_out_rem != 0 else 0))

        if self.is_dw:
            total_weight_offset_latency = n_spatial * (n_out_body * self.weight_offset_latency(k_out_body) + (self.weight_offset_latency(k_out_rem) if k_out_rem != 0 else 0))
            total_matrixvec_latency     = n_spatial * (n_out_body * self.matrixvec_latency(k_out_body)     + (self.matrixvec_latency(k_out_rem) if k_out_rem != 0 else 0))
            total_load_latency          = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.load_latency
            total_update_idx_latency    = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.update_idx_latency

            total_normquant_latency = n_spatial * (n_out_body * self.normquant_latency(k_out_body) + (self.normquant_latency(k_out_rem) if k_out_rem != 0 else 0))
            total_streamout_latency = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.streamout_latency
        else:
            total_weight_offset_latency = n_spatial * (n_out_body * n_in * self.weight_offset_latency(k_out_body) + ((n_in * self.weight_offset_latency(k_out_rem)) if k_out_rem != 0 else 0))
            total_matrixvec_latency     = n_spatial * (n_out_body * n_in * self.matrixvec_latency(k_out_body)     + ((n_in * self.matrixvec_latency(k_out_rem)) if k_out_rem != 0 else 0))
            total_load_latency          = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * n_in * self.load_latency
            total_update_idx_latency    = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * n_in * self.update_idx_latency

            total_normquant_latency = n_spatial * (n_out_body * self.normquant_latency(k_out_body) + (self.normquant_latency(k_out_rem) if k_out_rem != 0 else 0))
            total_streamout_latency = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.streamout_latency

        total_component_wise_latency = total_weight_offset_latency + total_matrixvec_latency + total_load_latency + total_update_idx_latency + total_normquant_latency + total_streamout_latency

        # assert total_latency == total_component_wise_latency, f"total latencies don't match: {total_latency} vs. {total_component_wise_latency}"

        """
        print(f'Latency breakdown:\n'
              f'  - load: {total_load_latency} ({total_load_latency / total_latency:.2%})\n'
              f'  - weight offset: {total_weight_offset_latency} ({total_weight_offset_latency / total_latency:.2%})\n'
              f'  - matrixvec: {total_matrixvec_latency} ({total_matrixvec_latency / total_latency:.2%})\n'
              f'  - update idx: {total_update_idx_latency} ({total_update_idx_latency / total_latency:.2%})\n'
              f'  - normquant: {total_normquant_latency} ({total_normquant_latency / total_latency:.2%})\n'
              f'  - streamout: {total_streamout_latency} ({total_streamout_latency / total_latency:.2%})\n'
              f'TOTAL: {total_latency}')
        """

        #return total_latency
        return total_component_wise_latency
    
    def tiled_layer_latency(self, layer_shape_in, layer_shape_out, tile_shape_out):
        body_h_count = FloorSTE(layer_shape_out[0], tile_shape_out[0])
        body_w_count = FloorSTE(layer_shape_out[1], tile_shape_out[1])
        body_k_count = FloorSTE(layer_shape_out[2], tile_shape_out[2])
        rem_h = ModuloSTE(layer_shape_out[0], tile_shape_out[0])
        rem_w = ModuloSTE(layer_shape_out[1], tile_shape_out[1])
        rem_k = ModuloSTE(layer_shape_out[2], tile_shape_out[2])
        layers = [
            (tile_shape_out[0], tile_shape_out[1], tile_shape_out[2], layer_shape_in[2]),
            (tile_shape_out[0], tile_shape_out[1], rem_k, layer_shape_in[2]),
            (tile_shape_out[0], rem_w, tile_shape_out[2], layer_shape_in[2]),
            (tile_shape_out[0], rem_w, rem_k, layer_shape_in[2]),
            (rem_h, tile_shape_out[1], tile_shape_out[2], layer_shape_in[2]),
            (rem_h, tile_shape_out[1], rem_k, layer_shape_in[2]),
            (rem_h, rem_w, tile_shape_out[2], layer_shape_in[2]),
            (rem_h, rem_w, rem_k, layer_shape_in[2])
        ]
        n_tiles = [
            body_h_count * body_w_count * body_k_count,
            body_h_count * body_w_count * (1 if rem_k > 0 else 0),
            body_h_count * (1 if rem_w > 0 else 0) * body_k_count,
            body_h_count * (1 if rem_w > 0 else 0) * (1 if rem_k > 0 else 0),
            (1 if rem_h > 0 else 0) * body_w_count * body_k_count,
            (1 if rem_h > 0 else 0) * body_w_count * (1 if rem_k > 0 else 0),
            (1 if rem_h > 0 else 0) * (1 if rem_w > 0 else 0) * body_k_count,
            (1 if rem_h > 0 else 0) * (1 if rem_w > 0 else 0) * (1 if rem_k > 0 else 0)
        ]

        latency = 0
        for layer, n in zip(layers, n_tiles):
            self.set_layer(layer)
            latency += n * self.latency

        return latency

def Ne16PerfModel_generalized(name, ks, depthwise, WEIGHTS_BITWIDTH, layer):
    if ks[0]==3:
        ne16 = Ne16PerfModel(name, (3,3), depthwise=depthwise, WEIGHTS_BITWIDTH = WEIGHTS_BITWIDTH)
        ne16.set_layer(layer)
        return ne16
    else:
        ne16 = Ne16PerfModel(name, (1,1), depthwise=depthwise, WEIGHTS_BITWIDTH = WEIGHTS_BITWIDTH)
        ne16.set_layer(layer)
        return ne16

IS_PADDING_COUNTED = True

class NE16AcceleratorCostModel(ZigZagMatchCostModel):
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        super(NE16AcceleratorCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access)
    
    def def_transfer_cost(self):
        USE_PLINIO_TRANSFER_MODEL = False
        if USE_PLINIO_TRANSFER_MODEL:
            is_dw = self.pattern_name=='depthwise_conv2d'
            layer_params = (
                self.loop_sizes["OY"],
                self.loop_sizes["OX"],
                self.loop_sizes["K"] if not is_dw else self.loop_sizes["K"] + (16-(self.loop_sizes["K"]%16)),
                self.loop_sizes["C"] + (16-(self.loop_sizes["C"]%16)) if not is_dw else self.loop_sizes["K"] + (16-(self.loop_sizes["K"]%16))
            )
            self.ne16=Ne16PerfModel_generalized(
                    name='conv',
                    ks=(self.loop_sizes["FY"],self.loop_sizes["FX"]),
                    depthwise=is_dw,
                    WEIGHTS_BITWIDTH=8,
                    layer=layer_params)
            self.ne16.set_subtile(self.size_per_mem_level["O"]["OY"][0],
                                self.size_per_mem_level["O"]["OX"][0],
                                self.size_per_mem_level["O"]["K"][0] + (16-(self.size_per_mem_level["O"]["K"][0]%16)) if is_dw else self.size_per_mem_level["O"]["K"][0],
                                self.size_per_mem_level["I"]["C"][0] + (16-(self.size_per_mem_level["I"]["C"][0]%16)) if not is_dw else self.size_per_mem_level["O"]["K"][0] + (16-(self.size_per_mem_level["O"]["K"][0]%16)))
            return {
                "O":int(self.ne16.out_store_latency),
                "W":int(self.ne16.weight_load_latency),
                "I":int(self.ne16.input_load_latency)
            }
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

                    # 1D transfer
                    if IN_WIDTH_L1 == IN_WIDTH_L2 and IN_CHANNELS_L1 == IN_CHANNELS_L2:
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
                        TRANS_CYCLES = (((OUT_WIDTH_L1 * OUT_CHANNELS_L1 / BYTES_PER_CYCLE) + OUT_WIDTH_L1 * OVERHEAD_2D_TRANSFER) * OUT_HEIGHT_L1) * OUT_HEIGHT_L1 * OVERHEAD_3D_TRANSFER
                # add API and MATCH overhead
                return MATCH_SETUP_OVERHEAD + API_OVERHEAD + TRANS_CYCLES

            return {operand:calc_transfer_costs(operand) for operand in self.operands}
    
    def def_innermost_loops_cost(self):
        is_dw = self.pattern_name=='depthwise_conv2d'
        if IS_PADDING_COUNTED:
            # HOW MUCH PADDING IS REQUIRED
            self.padding_of_k = 0 if self.loop_sizes["K"]%16==0 else 16-(self.loop_sizes["K"]%16)
            self.padding_of_k_l1 = 0 if self.size_per_mem_level["O"]["K"][0]%16==0 else 16-(self.size_per_mem_level["O"]["K"][0]%16)
            self.padding_of_c = 0 if self.loop_sizes["C"]%16==0 else 16-(self.loop_sizes["C"]%16)
        else:
            # TRYING CASE WHERE PADDING IS NOT REQURIED
            self.padding_of_k = 0
            self.padding_of_k_l1 = 0
            self.padding_of_c = 0

        self.ne16_layer_params = (
            self.loop_sizes["OY"],
            self.loop_sizes["OX"],
            self.loop_sizes["K"] if not is_dw else self.loop_sizes["K"] + self.padding_of_k,
            self.loop_sizes["C"] + self.padding_of_c if not is_dw else self.loop_sizes["K"] + self.padding_of_k
        )
        self.ne16=Ne16PerfModel_generalized(
                name='conv',
                ks=(self.loop_sizes["FY"],self.loop_sizes["FX"]),
                depthwise=is_dw,
                WEIGHTS_BITWIDTH=8,
                layer=self.ne16_layer_params)
        #account for strides as well
        STRIDES_OVERHEAD = 1

        #STRIDES_OVERHEAD = (sum(self.layer_data.strides)/2)
        self.tiled_layer_latency = int(self.ne16.tiled_layer_latency(
            layer_shape_in=(self.partial_relevant_loop_sizes["IY"],self.partial_relevant_loop_sizes["IX"],self.loop_sizes["K"] + self.padding_of_k if is_dw else self.loop_sizes["C"] + self.padding_of_c),
            layer_shape_out=(self.loop_sizes["OY"],self.loop_sizes["OX"],self.loop_sizes["K"] + self.padding_of_k if is_dw else self.loop_sizes["K"]),
            tile_shape_out=(self.size_per_mem_level["O"]["OY"][0],self.size_per_mem_level["O"]["OX"][0],self.size_per_mem_level["O"]["K"][0]+self.padding_of_k_l1 if is_dw else self.size_per_mem_level["O"]["K"][0])))
        
        self.ne16.set_subtile(self.size_per_mem_level["O"]["OY"][0],
                              self.size_per_mem_level["O"]["OX"][0],
                              self.size_per_mem_level["O"]["K"][0] + self.padding_of_k_l1 if is_dw else self.size_per_mem_level["O"]["K"][0],
                              self.loop_sizes["C"] if not is_dw else self.size_per_mem_level["O"]["K"][0] + self.padding_of_k_l1)
        self.tiled_layer_latency_old = int(self.ne16.computation_iteration_latency+self.ne16.update_indexes_latency+self.ne16.iter_normquant_latency)
        return STRIDES_OVERHEAD * self.tiled_layer_latency / self.computational_iters
    
    def def_overall_execution(self):
        is_dw = self.pattern_name=='depthwise_conv2d'

        self.overall_latency_sync()   
        
        SOFTWARE_DIM_TO_PAD = "K" if is_dw else "C"
        self.SOFTWARE_PAD_COST = 0
        self.SOFTWARE_SLICING_COST = 0
        
        if self.loop_sizes[SOFTWARE_DIM_TO_PAD]%16!=0 and IS_PADDING_COUNTED:
            #add software padding cost
            self.SOFTWARE_PAD_COST = self.partial_relevant_loop_sizes["IY"] * self.partial_relevant_loop_sizes["IX"] * (self.loop_sizes[SOFTWARE_DIM_TO_PAD]+(16-self.loop_sizes[SOFTWARE_DIM_TO_PAD]%16))*1.5
            if is_dw:
                self.SOFTWARE_SLICING_COST = self.loop_sizes["OY"] * self.loop_sizes["OX"] * (self.loop_sizes[SOFTWARE_DIM_TO_PAD]+(16-self.loop_sizes[SOFTWARE_DIM_TO_PAD]%16))*1.5
        # add software costs
        self.match_overall_latency += self.SOFTWARE_PAD_COST + self.SOFTWARE_SLICING_COST