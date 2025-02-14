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

#ifndef __NE16_DEFS_H__
#define __NE16_DEFS_H__

/* ARHITECTURE */

#define NE16_FILTER_SIZE (3)
#define NE16_FILTER_BUFFER_SIZE (5)
#define NE16_SUBTILE_INPUT_HEIGHT (5)
#define NE16_SUBTILE_INPUT_WIDTH (5)
#define NE16_SUBTILE_INPUT_CHANNEL (16)
#define NE16_SUBTILE_OUTPUT_HEIGHT (3)
#define NE16_SUBTILE_OUTPUT_WIDTH (3)
#define NE16_SUBTILE_OUTPUT_CHANNEL (32)
#define NE16_OUTPUT_BANDWIDTH_BYTES (32)

#define NE16_WEIGHT_D0_STRIDE_MODE8 (2)
#define NE16_WEIGHT_D0_STRIDE_MODE16 (1)

/* TASK REGISTERS */

// job configuration
#define NE16_REG_WEIGHTS_PTR 0
#define NE16_REG_INFEAT_PTR 1
#define NE16_REG_OUTFEAT_PTR 2
#define NE16_REG_SCALE_PTR 3
#define NE16_REG_SCALE_SHIFT_PTR 4
#define NE16_REG_SCALE_BIAS_PTR 5
#define NE16_REG_INFEAT_D0_STRIDE 6
#define NE16_REG_INFEAT_D1_STRIDE 7
#define NE16_REG_INFEAT_D2_STRIDE 8
#define NE16_REG_OUTFEAT_D0_STRIDE 9
#define NE16_REG_OUTFEAT_D1_STRIDE 10
#define NE16_REG_OUTFEAT_D2_STRIDE 11
#define NE16_REG_WEIGHTS_D0_STRIDE 12
#define NE16_REG_WEIGHTS_D1_STRIDE 13
#define NE16_REG_WEIGHTS_D2_STRIDE 14
#define NE16_REG_SUBTILE_REMAINDER_0 15
#define NE16_REG_SUBTILE_REMAINDER_1 16
#define NE16_REG_SUBTILE_REMAINDER_2 17
#define NE16_REG_SUBTILE_NUMBER_0 18
#define NE16_REG_SUBTILE_NUMBER_1 19
#define NE16_REG_PADDING 20
#define NE16_REG_WEIGHT_OFFSET_FACTOR 21
#define NE16_REG_FILTER_MASKING 22
#define NE16_REG_CONF0 23

/*  CONF0 FLAGS */

#define NE16_FLAG_NORM_BIAS (1 << 25)
#define NE16_FLAG_NORM_SHIFT (1 << 24)
#define NE16_FLAG_QUANT_FUNCTION_IDENTITY (1 << 23)
#define NE16_FLAG_QUANT_FUNCTION_RELU (0 << 23)
#define NE16_QUANT_MODE_8BIT (0 << 21)
#define NE16_QUANT_MODE_16BIT (1 << 21)
#define NE16_QUANT_MODE_32BIT (2 << 21)
// conf0[20:16] - quantization shift amount
#define NE16_FLAG_WEIGHT_OFFSET_SYMMETRIC (0 << 15)
#define NE16_FLAG_WEIGHT_OFFSET_LAYER_WISE (1 << 15)
#define NE16_FLAG_STREAMIN (1 << 14)
#define NE16_NORM_MODE_8BIT (0 << 12)
#define NE16_NORM_MODE_16BIT (1 << 12)
#define NE16_NORM_MODE_32BIT (2 << 12)
#define NE16_FLAG_ROUNDING (1 << 11)
#define NE16_FLAG_STRIDE_2x2 (1 << 8)
#define NE16_FLAG_LINEAR_MODE (1 << 7)
#define NE16_FLAG_MODE_3x3 (0 << 5)
#define NE16_FLAG_MODE_3x3_DW (1 << 5)
#define NE16_FLAG_MODE_1x1 (2 << 5)
#define NE16_FLAG_NORM_QUANT (1 << 4)
#define NE16_FLAG_MODE_BASIC (0 << 3)
#define NE16_FLAG_MODE16 (1 << 3)

/*  SHIFT  */

#define NE16_SHIFT_FLAG_NORM_BIAS (25)
#define NE16_SHIFT_FLAG_NORM_SHIFT (24)
#define NE16_SHIFT_FLAG_ROUNDING (11)

/* Masks */

#define NE16_MASK_FLAG_NORM_BIAS (0x1 << 25)
#define NE16_MASK_FLAG_NORM_SHIFT (0x1 << 24)
#define NE16_MASK_QUANT_FUNCTION (0x1 << 23)
#define NE16_MASK_QUANT_MODE (0x3 << 21)
#define NE16_MASK_SHIFT_AMOUNT (0x1f << 16)
#define NE16_MASK_WEIGHT_OFFSET_MODE (0x1 << 15)
#define NE16_MASK_NORM_MODE (0x3 << 12)
#define NE16_MASK_FLAG_ROUNDING (0x1 << 11)
#define NE16_MASK_FLAG_STRIDE_2x2 (0x1 << 8)
#define NE16_MASK_FLAG_MODE (0x3 << 5)
#define NE16_MASK_FLAG_MODE16 (0x1 << 3)
#define NE16_MASK_FLAG_WEIGHT_BITS (0x7 << 0)

/* PADDING */

#define NE16_DONT_PAD (0)
#define NE16_MAX_PAD (2)

/* NORM */
#define NE16_NORM_MAX_LEN (32)

#endif // __NE16_DEFS_H__
