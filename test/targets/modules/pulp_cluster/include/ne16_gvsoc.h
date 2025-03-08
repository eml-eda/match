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

#ifndef __NE16_GVSOC_H__
#define __NE16_GVSOC_H__

#include <pulp_cluster/ne16.h>
#include <pulp_cluster/ne16_task.h>

#define NE16_REG_GVSOC_LOG_LEVEL 24
#define NE16_REG_GVSOC_LOG_FORMAT 25

typedef enum ne16_gvsoc_log_format_e {
  NE16_GVSOC_LOG_FORMAT_DECIMAL = 0,
  NE16_GVSOC_LOG_FORMAT_HEXADECIMAL = 3
} ne16_gvsoc_log_format_e;

typedef enum ne16_gvsoc_log_level_e {
  NE16_GVSOC_LOG_LEVEL_CONFIG = 0,
  NE16_GVSOC_LOG_LEVEL_ACTIV_INOUT = 1,
  NE16_GVSOC_LOG_LEVEL_DEBUG = 2,
  NE16_GVSOC_LOG_LEVEL_ALL = 3
} ne16_gvsoc_log_level_e;

static void ne16_gvsoc_log_activate(const ne16_dev_t *dev,
                                    ne16_gvsoc_log_level_e log_level,
                                    ne16_gvsoc_log_format_e format) {
  hwpe_task_reg_write(&dev->hwpe_dev, NE16_REG_GVSOC_LOG_LEVEL, log_level);
  hwpe_task_reg_write(&dev->hwpe_dev, NE16_REG_GVSOC_LOG_FORMAT, format);
}

static void ne16_gvsoc_log_deactivate(const ne16_dev_t *dev) {
  hwpe_task_reg_write(&dev->hwpe_dev, NE16_REG_GVSOC_LOG_LEVEL,
                      NE16_GVSOC_LOG_LEVEL_CONFIG);
}

#endif // __NE16_GVSOC_H__
