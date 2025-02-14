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

#ifndef __NE16_H__
#define __NE16_H__

#include <pulp_target/hwpe.h>
#include <stdint.h>

#define NE16_TASK_QUEUE_SIZE (2)

typedef struct ne16_dev_t {
  hwpe_dev_t hwpe_dev; /* Implements the HWPE device interface */
} ne16_dev_t;

int ne16_task_queue_tasks_in_flight(const ne16_dev_t *dev);
int ne16_task_queue_empty(const ne16_dev_t *dev);
int ne16_task_queue_full(const ne16_dev_t *dev);

#endif // __NE16_H__
