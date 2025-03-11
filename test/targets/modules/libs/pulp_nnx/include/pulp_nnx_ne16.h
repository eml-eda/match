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

#include <pulp_neural/ne16.h>
#include <pulp_neural/ne16_pulp_bsp.h>
#include <pulp_neural/ne16_task.h>
#include <stdint.h>

/* PULP-NNX interface */

void ne16_nnx_init(const ne16_dev_t *dev, ne16_pulp_conf_t *conf);
void ne16_nnx_term(const ne16_dev_t *dev);

/** ne16_nnx_dispatch_check
 *
 * Check whether you can dispatch to the accelerator.
 */
int ne16_nnx_dispatch_check(const ne16_dev_t *dev);

/** ne16_nnx_dispatch_wait
 *
 * Block until you can dispatch to the accelerator.
 */
void ne16_nnx_dispatch_wait(const ne16_dev_t *dev);

/** ne16_nnx_dispatch
 *
 * Dispatch a task to the accelerator.
 * Fails with return code 1 if the task cannot be dispatched. Otherwise returns
 * 0.
 */
int ne16_nnx_dispatch(const ne16_dev_t *dev, ne16_task_t *task);

/** ne16_nnx_resolve_check
 *
 * Check whether the task has been resolved.
 */
int ne16_nnx_resolve_check(const ne16_dev_t *dev, ne16_task_t *task);

/** ne16_nnx_resolve_wait
 *
 * Block until you can resolve the task.
 */
void ne16_nnx_resolve_wait(const ne16_dev_t *dev, ne16_task_t *task);

/* Additional helper functions */

/** ne16_nnx_dispatch_stride2x2
 *
 * It uses NE16's 2x2 strided mode which reduces the number of writes NE16 does.
 * This mode doesn't stride the NE16's subtile input pointer, so we have to
 * tile the tile to the subtile's spatial dimensions (in this case 3x3 output).
 * Works only if the k_out is divisible by 2.
 */
void ne16_nnx_dispatch_stride2x2(const ne16_dev_t *dev, ne16_task_t *task,
                                 const uint32_t w_in, const uint32_t k_in,
                                 const uint32_t h_out, const uint32_t w_out,
                                 const uint32_t k_out, const uint8_t h_ker,
                                 const uint8_t w_ker);
