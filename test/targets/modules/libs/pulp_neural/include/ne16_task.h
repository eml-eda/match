/*
 * Luka Macan <luka.macan@unibo.it>
 *
 * Copyright 2023 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __NE16_TASK_H__
#define __NE16_TASK_H__

#include <pulp_neural/ne16_task_defs.h>
#include <stdint.h>

typedef enum ne16_task_flag_e {
  ne16TaskFlagFalse = 0,
  ne16TaskFlagTrue = 1
} ne16_task_flag_e;

typedef enum ne16_weight_offset_mode_e {
  weightOffsetModeSymmetric = NE16_FLAG_WEIGHT_OFFSET_SYMMETRIC,
  weightOffsetModeLayerWise = NE16_FLAG_WEIGHT_OFFSET_LAYER_WISE
} ne16_weight_offset_mode_e;

typedef enum {
  normMode8Bit = NE16_NORM_MODE_8BIT,
  normMode16Bit = NE16_NORM_MODE_16BIT,
  normMode32Bit = NE16_NORM_MODE_32BIT
} ne16_norm_mode_e;

typedef struct ne16_norm_t {
  ne16_norm_mode_e mode;
  ne16_task_flag_e flag_bias;
  ne16_task_flag_e flag_shift;
} ne16_norm_t;

typedef enum ne16_quant_mode_e {
  quantMode8Bit = NE16_QUANT_MODE_8BIT,
  quantMode16Bit = NE16_QUANT_MODE_16BIT,
  quantMode32Bit = NE16_QUANT_MODE_32BIT
} ne16_quant_mode_e;

typedef enum ne16_quant_function_e {
  quantFunctionIdentity = NE16_FLAG_QUANT_FUNCTION_IDENTITY,
  quantFunctionRelu = NE16_FLAG_QUANT_FUNCTION_RELU
} ne16_quant_function_e;

typedef struct ne16_quant_t {
  // Shift amount must be in range 0x00-0x1F
  uint8_t shift_amount;
  ne16_quant_function_e function;
  ne16_task_flag_e flag_rounding;
} ne16_quant_t;

typedef struct ne16_stride_t {
  uint32_t d0;
  uint32_t d1;
  uint32_t d2;
} ne16_stride_t;

typedef struct ne16_subtile_remainder_t {
  uint32_t KoKi;
  uint32_t HoWo;
  uint32_t HiWi;
} ne16_subtile_remainder_t;

typedef struct ne16_subtile_number_t {
  uint32_t KoKi;
  uint32_t HoWo;
} ne16_subtile_number_t;

typedef struct ne16_subtile_t {
  ne16_subtile_remainder_t remainder;
  ne16_subtile_number_t number;
} ne16_subtile_t;

typedef struct ne16_cfg_t {
  ne16_stride_t input_stride;
  ne16_stride_t output_stride;
  ne16_stride_t weights_stride;
  ne16_subtile_t subtile;
  uint32_t padding;
  uint32_t weight_offset_factor;
  uint32_t filter_mask;
  uint32_t conf0;
} ne16_cfg_t;

typedef struct ne16_task_data_t {
  uint32_t weights_addr;
  uint32_t infeat_addr;
  uint32_t outfeat_addr;
  uint32_t scale_addr;
  uint32_t scale_shift_addr;
  uint32_t scale_bias_addr;
  ne16_cfg_t cfg;
} ne16_task_data_t;

typedef struct ne16_task_t {
  ne16_task_data_t data;
  uint8_t weight_d0_stride;
  uint8_t qw;
  uint8_t subtile_output_channel;
  uint8_t kernel_shape;
  uint8_t depthwise;
  uint8_t id;
} ne16_task_t;

void ne16_task_init(ne16_task_t *task);
void ne16_task_set_op_to_conv(ne16_task_t *task, const uint8_t kernel_shape,
                              const uint8_t depthwise, const uint8_t stride);
void ne16_task_set_bits(ne16_task_t *task, const uint8_t input_bits,
                        const uint8_t output_bits, const uint8_t weight_bits);
void ne16_task_set_norm_quant(ne16_task_t *task, ne16_quant_t quant,
                              ne16_norm_t norm);
void ne16_task_set_weight_offset(ne16_task_t *task,
                                 ne16_weight_offset_mode_e weight_offset_mode,
                                 const int32_t weight_offset);
uint32_t ne16_get_tile_padding(uint32_t padding, uint32_t i_height,
                               uint32_t i_width, uint32_t n_height,
                               uint32_t n_width);
uint32_t ne16_pad_addr(uint32_t ptr, const uint32_t width,
                       const uint32_t width_stride, const uint8_t padding_top,
                       const uint8_t padding_left);
void ne16_task_set_addr_conv(ne16_task_t *task, uint32_t input_addr,
                             uint32_t w_in, uint32_t w_in_stride,
                             uint8_t padding_top, uint8_t padding_left,
                             uint32_t output_addr, uint32_t weights_addr);
void ne16_task_set_addr_norm_quant(ne16_task_t *task, uint32_t scale_addr,
                                   uint32_t shift_addr, uint32_t bias_addr);
/** ne16_task_set_strides
 *
 * All the strides variables are strides between elements alongside that
 * dimension and expressed in bytes. There is no stride variable for the channel
 * dimension because the NE16 requires the channels to be contiguous.
 */
void ne16_task_set_strides(ne16_task_t *task, const uint32_t k_in,
                           const uint32_t h_in_stride,
                           const uint32_t w_in_stride,
                           const uint32_t h_out_stride,
                           const uint32_t w_out_stride);
void ne16_task_set_counters(ne16_task_t *task, const uint32_t k_in,
                            const uint32_t h_out, const uint32_t w_out,
                            const uint32_t k_out, const uint8_t padding_bottom,
                            const uint8_t padding_right);
void ne16_task_set_padding(ne16_task_t *task, const uint8_t top,
                           const uint8_t bottom, const uint8_t left,
                           const uint8_t right, const uint8_t value);
void ne16_task_set_mask_filter(ne16_task_t *task, const uint8_t top,
                               const uint8_t bottom, const uint8_t left,
                               const uint8_t right);
/** ne16_task_set_dims
 *
 * All the strides variables are strides between elements alongside that
 * dimension and expressed in bytes. There is no stride variable for the channel
 * dimension because the NE16 requires the channels to be contiguous.
 */
void ne16_task_set_dims(ne16_task_t *task, const uint32_t w_in,
                        const uint32_t k_in, const uint32_t h_in_stride,
                        const uint32_t w_in_stride, const uint32_t h_out,
                        const uint32_t w_out, const uint32_t k_out,
                        const uint32_t h_out_stride,
                        const uint32_t w_out_stride, const uint8_t padding_top,
                        const uint8_t padding_bottom,
                        const uint8_t padding_left,
                        const uint8_t padding_right);
/** ne16_task_set_dims_stride2x2
 *
 * All the strides variables are strides between elements alongside that
 * dimension and expressed in bytes. There is no stride variable for the channel
 * dimension because the NE16 requires the channels to be contiguous.
 */
void ne16_task_set_dims_stride2x2(
    ne16_task_t *task, const uint32_t h_in, const uint32_t w_in,
    const uint32_t k_in, const uint32_t h_in_stride, const uint32_t w_in_stride,
    const uint32_t h_out, const uint32_t w_out, const uint32_t k_out,
    const uint32_t h_out_stride, const uint32_t w_out_stride,
    const uint8_t h_ker, const uint8_t w_ker, const uint8_t padding_top,
    const uint8_t padding_bottom, const uint8_t padding_left,
    const uint8_t padding_right);

#endif // !__NE16_TASK_H__
