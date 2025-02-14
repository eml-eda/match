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

#ifndef __HWPE_H__
#define __HWPE_H__

#include <stdint.h>

/* HWPE device */
typedef struct hwpe_dev_t {
  volatile uint32_t *base_addr;
} hwpe_dev_t;

void hwpe_reg_write(hwpe_dev_t *dev, int reg, uint32_t value);
uint32_t hwpe_reg_read(hwpe_dev_t *dev, int reg);
void hwpe_task_reg_write(hwpe_dev_t *dev, int reg, uint32_t value);
uint32_t hwpe_task_reg_read(hwpe_dev_t *dev, int reg);
void hwpe_soft_clear(hwpe_dev_t *dev);
uint32_t hwpe_task_queue_status(hwpe_dev_t *dev);
int hwpe_task_queue_acquire_task(hwpe_dev_t *dev, uint8_t *id);
void hwpe_task_queue_write_task(hwpe_dev_t *dev, uint32_t *data, int len);
void hwpe_task_queue_release_and_run(hwpe_dev_t *dev);
void hwpe_task_queue_release(hwpe_dev_t *dev);
uint8_t hwpe_last_task_id(hwpe_dev_t *dev);

#endif // !__HWPE_H__
