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

#ifndef __NNX_UTIL_H__
#define __NNX_UTIL_H__

#include <stdint.h>

/**
 * nnx_calculate_number_of_iterations
 *
 * Calculates the number of iterations to go through a dimension.
 * It does it by dividing the dimension with the tile size and doing a ceiling
 * the result.
 */
int nnx_calculate_number_of_tiles(const int dim_size, const int tile_size);

/**
 * nnx_calculate_last_tile_size
 *
 * Calculates the size of the last executed tile by calculating the remainder of
 * the dim_size and the tile_size. In case the remainder is 0, it returns the
 * full tile_size.
 */
int nnx_calculate_last_tile_size(const int dim_size, const int tile_size);

/**
 * concat_half
 *
 * Concatenate 2 16-bit numbers into a 32-bit number.
 */
uint32_t nnx_concat_half(const uint16_t high, const uint16_t low);

#endif // __NNX_UTIL_H__
