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

#ifndef __NE16_PULP_BSP_H__
#define __NE16_PULP_BSP_H__

#include <pulp_cluster/ne16.h>
#include <stdint.h>

/**
 * ne16_pulp_cg_enable
 *
 * Enable clock gating of the NE16.
 */
void ne16_pulp_cg_enable();

/**
 * ne16_pulp_cg_enable
 *
 * Disable clock gating of the NE16.
 */
void ne16_pulp_cg_disable();

/**
 * ne16_pulp_setpriority_ne16
 *
 * Set HCI interconnect bus priority to prioritize NE16.
 */
void ne16_pulp_hci_setpriority_ne16();

/**
 * ne16_pulp_setpriority_core
 *
 * Set HCI bus priority to prioritize cores.
 */
void ne16_pulp_hci_setpriority_core();

/**
 * ne16_pulp_hci_reset_maxstall
 *
 * Reset the HCI bus maxstall parameter.
 * TODO: Check if it disables it also or just resets?
 */
void ne16_pulp_hci_reset_max_stall();

/**
 * ne16_pulp_hci_set_maxstall
 *
 * Set the HCI bus maxstall. Maxstall defines how many cycles
 * will the HCI bus stall the lower priority master, i.e. ne16 or core,
 * before letting it do a transaction.
 */
void ne16_pulp_hci_set_max_stall(uint32_t max_stall);

typedef struct ne16_pulp_conf_t {
  int max_stall;
} ne16_pulp_conf_t;

void ne16_pulp_open(ne16_pulp_conf_t *conf);
void ne16_pulp_close();
void ne16_pulp_event_wait_and_clear();
const ne16_dev_t *ne16_pulp_get_dev();

#endif // !__NE16_PULP_BSP_H__
