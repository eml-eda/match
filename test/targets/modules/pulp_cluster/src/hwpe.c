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

#include <pulp_cluster/hwpe.h>
#include <stdint.h>

#define HWPE_TRIGGER 0
#define HWPE_ACQUIRE 1
#define HWPE_FINISHED 2
#define HWPE_STATUS 3
#define HWPE_RUNNING_JOB 4
#define HWPE_SOFT_CLEAR 5
#define HWPE_SWSYNC 6
#define HWPE_TASK_REG_OFFSET 8

inline void hwpe_reg_write(hwpe_dev_t *dev, int reg, uint32_t value) {
  dev->base_addr[reg] = value;
}

inline uint32_t hwpe_reg_read(hwpe_dev_t *dev, int reg) {
  return dev->base_addr[reg];
}

inline void hwpe_task_reg_write(hwpe_dev_t *dev, int reg, uint32_t value) {
  hwpe_reg_write(dev, HWPE_TASK_REG_OFFSET + reg, value);
}

inline uint32_t hwpe_task_reg_read(hwpe_dev_t *dev, int reg) {
  return hwpe_reg_read(dev, HWPE_TASK_REG_OFFSET + reg);
}

void hwpe_soft_clear(hwpe_dev_t *dev) {
  hwpe_reg_write(dev, HWPE_SOFT_CLEAR, 0);
  for (volatile int i = 0; i < 10; i++)
    ;
}

uint32_t hwpe_task_queue_status(hwpe_dev_t *dev) {
  return hwpe_reg_read(dev, HWPE_STATUS);
}

int hwpe_task_queue_acquire_task(hwpe_dev_t *dev, uint8_t *id) {
  uint32_t read_value = (int32_t)hwpe_reg_read(dev, HWPE_ACQUIRE);
  if (read_value >= 256) {
    return 1;
  } else {
    *id = (uint8_t)read_value;
    return 0;
  }
}

void hwpe_task_queue_write_task(hwpe_dev_t *dev, uint32_t *data, int len) {
  for (int i = 0; i < len; i++) {
    hwpe_task_reg_write(dev, i, data[i]);
  }
}

void hwpe_task_queue_release_and_run(hwpe_dev_t *dev) {
  hwpe_reg_write(dev, HWPE_TRIGGER, 0);
}

void hwpe_task_queue_release(hwpe_dev_t *dev) {
  hwpe_reg_write(dev, HWPE_TRIGGER, 1);
}

uint8_t hwpe_last_task_id(hwpe_dev_t *dev) {
  return (uint8_t)hwpe_reg_read(dev, HWPE_RUNNING_JOB);
}
