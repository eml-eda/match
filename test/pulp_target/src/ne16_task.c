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

#include <pulp_target/ne16_task.h>
#include <pulp_target/ne16_task_defs.h>
#include <pulp_target/pulp_nnx_util.h>

uint32_t ne16_get_tile_padding(uint32_t padding, uint32_t i_height,
                               uint32_t i_width, uint32_t n_height,
                               uint32_t n_width) {
  uint32_t tile_padding = padding;
  if (i_height > 0) {
    tile_padding &= ~(0xf << 28);
  }
  if (i_width < n_width - 1) {
    tile_padding &= ~(0xf << 24);
  }
  if (i_height < n_height - 1) {
    tile_padding &= ~(0xf << 20);
  }
  if (i_width > 0) {
    tile_padding &= ~(0xf << 16);
  }
  return tile_padding;
}

void ne16_task_init(ne16_task_t *task) { *task = (ne16_task_t){.data = {0}}; }

void ne16_task_set_op_to_conv(ne16_task_t *task, const uint8_t kernel_shape,
                              const uint8_t depthwise, const uint8_t stride) {
  task->depthwise = depthwise;
  task->kernel_shape = kernel_shape;
  task->subtile_output_channel =
      depthwise ? NE16_SUBTILE_INPUT_CHANNEL : NE16_SUBTILE_OUTPUT_CHANNEL;
  const int flag_mode = kernel_shape == 1 ? NE16_FLAG_MODE_1x1
                        : depthwise == 1  ? NE16_FLAG_MODE_3x3_DW
                                          : NE16_FLAG_MODE_3x3;

  const int flag_stride2x2 = stride == 2 ? NE16_FLAG_STRIDE_2x2 : 0;

  task->data.cfg.conf0 &= ~(NE16_MASK_FLAG_MODE | NE16_MASK_FLAG_STRIDE_2x2);
  task->data.cfg.conf0 |= flag_mode | flag_stride2x2;
}

void ne16_task_set_bits(ne16_task_t *task, const uint8_t input_bits,
                        const uint8_t output_bits, const uint8_t weight_bits) {
  const uint32_t flag_mode16 =
      input_bits == 16 ? NE16_FLAG_MODE16 : NE16_FLAG_MODE_BASIC;

  ne16_quant_mode_e quantMode;
  if (output_bits == 16) {
    quantMode = quantMode16Bit;
  } else if (output_bits == 8) {
    quantMode = quantMode8Bit;
  } else {
    quantMode = quantMode32Bit;
  }

  task->weight_d0_stride =
      flag_mode16 ? NE16_WEIGHT_D0_STRIDE_MODE16 : NE16_WEIGHT_D0_STRIDE_MODE8;
  task->qw = weight_bits;
  task->data.cfg.conf0 &= ~(NE16_MASK_QUANT_MODE | NE16_MASK_FLAG_MODE16 |
                            NE16_MASK_FLAG_WEIGHT_BITS);
  task->data.cfg.conf0 |= quantMode | flag_mode16 | (weight_bits - 1);
}

void ne16_task_set_norm_quant(ne16_task_t *task, ne16_quant_t quant,
                              ne16_norm_t norm) {
  task->data.cfg.conf0 &=
      ~(NE16_MASK_QUANT_FUNCTION | NE16_MASK_SHIFT_AMOUNT |
        NE16_MASK_FLAG_ROUNDING | NE16_MASK_NORM_MODE |
        NE16_MASK_FLAG_NORM_BIAS | NE16_MASK_FLAG_NORM_SHIFT);
  task->data.cfg.conf0 |=
      NE16_FLAG_NORM_QUANT | quant.function | (quant.shift_amount << 16) |
      quant.flag_rounding << NE16_SHIFT_FLAG_ROUNDING | norm.mode |
      norm.flag_bias << NE16_SHIFT_FLAG_NORM_BIAS |
      norm.flag_shift << NE16_SHIFT_FLAG_NORM_SHIFT;
}

void ne16_task_set_weight_offset(ne16_task_t *task,
                                 ne16_weight_offset_mode_e weight_offset_mode,
                                 const int32_t weight_offset) {
  task->data.cfg.conf0 &= ~NE16_MASK_WEIGHT_OFFSET_MODE;
  task->data.cfg.conf0 |= weight_offset_mode;
  task->data.cfg.weight_offset_factor = weight_offset;
}

/** ne16_pad_addr
 *
 * Calculate the pointer to the start of the ptr as if
 * it was the start to the padded data.
 * Necessary for input pointer when it's padded.
 */
uint32_t ne16_pad_addr(uint32_t ptr, const uint32_t width,
                       uint32_t width_stride, const uint8_t padding_top,
                       const uint8_t padding_left) {
  return ptr - (padding_top * width + padding_left) * width_stride;
}

void ne16_task_set_addr_conv(ne16_task_t *task, uint32_t input_addr,
                             uint32_t w_in, uint32_t w_in_stride,
                             uint8_t padding_top, uint8_t padding_left,
                             uint32_t output_addr, uint32_t weights_addr) {
  task->data.infeat_addr =
      ne16_pad_addr(input_addr, w_in, w_in_stride, padding_top, padding_left);
  task->data.outfeat_addr = output_addr;
  task->data.weights_addr = weights_addr;
}

void ne16_task_set_addr_norm_quant(ne16_task_t *task, uint32_t scale_addr,
                                   uint32_t shift_addr, uint32_t bias_addr) {
  task->data.scale_addr = scale_addr;
  task->data.scale_shift_addr = shift_addr;
  task->data.scale_bias_addr = bias_addr;
}

void ne16_task_set_strides(ne16_task_t *task, const uint32_t k_in,
                           const uint32_t h_in_stride,
                           const uint32_t w_in_stride,
                           const uint32_t h_out_stride,
                           const uint32_t w_out_stride) {
  const uint32_t num_k_in =
      nnx_calculate_number_of_tiles(k_in, NE16_SUBTILE_INPUT_CHANNEL);

  const ne16_stride_t input_stride = {
      .d0 = w_in_stride, .d1 = h_in_stride, .d2 = 0};
  task->data.cfg.input_stride = input_stride;

  const ne16_stride_t output_stride = {.d0 = NE16_OUTPUT_BANDWIDTH_BYTES,
                                       .d1 = w_out_stride,
                                       .d2 = h_out_stride};
  task->data.cfg.output_stride = output_stride;

  if (task->kernel_shape == 1) {
    task->data.cfg.weights_stride.d0 = task->weight_d0_stride * task->qw;
    task->data.cfg.weights_stride.d1 =
        task->weight_d0_stride * task->qw * num_k_in;
  } else if (!task->depthwise) {
    task->data.cfg.weights_stride.d0 =
        NE16_FILTER_SIZE * NE16_FILTER_SIZE * task->weight_d0_stride;
    task->data.cfg.weights_stride.d1 = NE16_FILTER_SIZE * NE16_FILTER_SIZE *
                                       task->weight_d0_stride * task->qw *
                                       num_k_in;
  } else {
    task->data.cfg.weights_stride.d0 =
        NE16_FILTER_SIZE * NE16_FILTER_SIZE * task->weight_d0_stride;
    task->data.cfg.weights_stride.d1 = 0;
  }
  task->data.cfg.weights_stride.d2 = 0;
}

void ne16_task_set_counters(ne16_task_t *task, const uint32_t k_in,
                            const uint32_t h_out, const uint32_t w_out,
                            const uint32_t k_out, const uint8_t padding_bottom,
                            const uint8_t padding_right) {
  const uint16_t num_Ko =
      nnx_calculate_number_of_tiles(k_out, task->subtile_output_channel);
  const uint16_t num_Ki =
      nnx_calculate_number_of_tiles(k_in, NE16_SUBTILE_INPUT_CHANNEL);
  const uint16_t num_Ho =
      nnx_calculate_number_of_tiles(h_out, NE16_SUBTILE_OUTPUT_HEIGHT);
  const uint16_t num_Wo =
      nnx_calculate_number_of_tiles(w_out, NE16_SUBTILE_OUTPUT_WIDTH);

  const uint16_t rem_Ko =
      nnx_calculate_last_tile_size(k_out, task->subtile_output_channel);
  const uint16_t rem_Ki =
      nnx_calculate_last_tile_size(k_in, NE16_SUBTILE_INPUT_CHANNEL);
  const uint16_t rem_Ho =
      nnx_calculate_last_tile_size(h_out, NE16_SUBTILE_OUTPUT_HEIGHT);
  const uint16_t rem_Wo =
      nnx_calculate_last_tile_size(w_out, NE16_SUBTILE_OUTPUT_WIDTH);
  const uint16_t rem_Hi =
      (task->kernel_shape == 1 ? rem_Ho : rem_Ho + 2) - padding_bottom;
  const uint16_t rem_Wi =
      (task->kernel_shape == 1 ? rem_Wo : rem_Wo + 2) - padding_right;

  const ne16_subtile_t subtile = {
      .number = {.KoKi = nnx_concat_half(num_Ko, num_Ki),
                 .HoWo = nnx_concat_half(num_Ho, num_Wo)},
      .remainder = {.KoKi = nnx_concat_half(rem_Ko, rem_Ki),
                    .HoWo = nnx_concat_half(rem_Ho, rem_Wo),
                    .HiWi = nnx_concat_half(rem_Hi, rem_Wi)}};
  task->data.cfg.subtile = subtile;
}

void ne16_task_set_padding(ne16_task_t *task, const uint8_t top,
                           const uint8_t bottom, const uint8_t left,
                           const uint8_t right, const uint8_t value) {
  task->data.cfg.padding = ((top & 0xf) << 28) | ((right & 0xf) << 24) |
                           ((bottom & 0xf) << 20) | ((left & 0xf) << 16) |
                           (value & 0xff);
}

void ne16_task_set_mask_filter(ne16_task_t *task, const uint8_t top,
                               const uint8_t bottom, const uint8_t left,
                               const uint8_t right) {
  task->data.cfg.filter_mask = ((top & 0xff) << 24) | ((right & 0xff) << 16) |
                               ((bottom & 0xff) << 8) | ((left & 0xff) << 0);
}

void ne16_task_set_dims(ne16_task_t *task, const uint32_t w_in,
                        const uint32_t k_in, const uint32_t h_in_stride,
                        const uint32_t w_in_stride, const uint32_t h_out,
                        const uint32_t w_out, const uint32_t k_out,
                        const uint32_t h_out_stride,
                        const uint32_t w_out_stride, const uint8_t padding_top,
                        const uint8_t padding_bottom,
                        const uint8_t padding_left,
                        const uint8_t padding_right) {
  ne16_task_set_strides(task, k_in, h_in_stride, w_in_stride, h_out_stride,
                        w_out_stride);
  ne16_task_set_counters(task, k_in, h_out, w_out, k_out, padding_bottom,
                         padding_right);
  ne16_task_set_padding(task, padding_top, padding_bottom, padding_left,
                        padding_right, 0);
}

void ne16_task_set_dims_stride2x2(
    ne16_task_t *task, const uint32_t h_in, const uint32_t w_in,
    const uint32_t k_in, const uint32_t h_in_stride, const uint32_t w_in_stride,
    const uint32_t h_out, const uint32_t w_out, const uint32_t k_out,
    const uint32_t h_out_stride, const uint32_t w_out_stride,
    const uint8_t h_ker, const uint8_t w_ker, const uint8_t padding_top,
    const uint8_t padding_bottom, const uint8_t padding_left,
    const uint8_t padding_right) {
  const uint8_t stride = 2;

  // WARNING: works only for even output channel stride (divisible by 2)
  ne16_task_set_strides(task, k_in, h_in_stride, w_in_stride, h_out_stride >> 1,
                        w_out_stride >> 1);
  ne16_task_set_counters(task, k_in, h_out > 1 ? 3 : 1, w_out > 1 ? 3 : 1,
                         k_out, h_in + padding_top >= 5 ? 0 : padding_bottom,
                         0);

  const uint8_t padding_bottom_new =
      (h_in + padding_top - h_ker) % stride == 0 ? 0 : padding_bottom;
  const uint8_t padding_right_new =
      (w_in + padding_left - w_ker) % stride == 0 ? 0 : padding_right;

  ne16_task_set_padding(task, padding_top, padding_bottom_new, padding_left,
                        padding_right_new, 0);
}
