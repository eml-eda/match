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

#include <pulp_neural/ne16_pulp_bsp.h>
#include <pmsis.h>

#define NE16_PULP_CLUSTER_CTRL_ADDR_BASE (0x00200000)
#define NE16_PULP_CLUSTER_CTRL_HWPE_OFFS 0x18
#define NE16_PULP_CLUSTER_CTRL_HWPE_ADDR                                       \
  (NE16_PULP_CLUSTER_CTRL_ADDR_BASE + NE16_PULP_CLUSTER_CTRL_HWPE_OFFS)
#define NE16_PULP_CLUSTER_CTRL_HWPE_MASK_CG_EN 0x800
#define NE16_PULP_CLUSTER_CTRL_HWPE_MASK_HCI_PRIO 0x100
#define NE16_PULP_CLUSTER_CTRL_HWPE_MASK_HCI_MAXSTALL 0xff
#define NE16_PULP_MAX_STALL (8)
#define NE16_PULP_EVENT (1 << 12)
#define NE16_PULP_BASE_ADDR (0x00201000)

void ne16_pulp_cg_enable() {
  *(volatile uint32_t *)NE16_PULP_CLUSTER_CTRL_HWPE_ADDR |=
      NE16_PULP_CLUSTER_CTRL_HWPE_MASK_CG_EN;
}

void ne16_pulp_cg_disable() {
  *(volatile uint32_t *)NE16_PULP_CLUSTER_CTRL_HWPE_ADDR &=
      ~NE16_PULP_CLUSTER_CTRL_HWPE_MASK_CG_EN;
}

void ne16_pulp_hci_setpriority_ne16() {
  *(volatile uint32_t *)NE16_PULP_CLUSTER_CTRL_HWPE_ADDR |=
      NE16_PULP_CLUSTER_CTRL_HWPE_MASK_HCI_PRIO;
}

void ne16_pulp_hci_setpriority_core() {
  *(volatile uint32_t *)NE16_PULP_CLUSTER_CTRL_HWPE_ADDR &=
      ~NE16_PULP_CLUSTER_CTRL_HWPE_MASK_HCI_PRIO;
}

void ne16_pulp_hci_reset_max_stall() {
  *(volatile uint32_t *)NE16_PULP_CLUSTER_CTRL_HWPE_ADDR &=
      ~NE16_PULP_CLUSTER_CTRL_HWPE_MASK_HCI_MAXSTALL;
}

void ne16_pulp_hci_set_max_stall(uint32_t max_stall) {
  *(volatile uint32_t *)NE16_PULP_CLUSTER_CTRL_HWPE_ADDR |=
      max_stall & NE16_PULP_CLUSTER_CTRL_HWPE_MASK_HCI_MAXSTALL;
}

void ne16_pulp_open(ne16_pulp_conf_t *conf) {
  ne16_pulp_cg_enable();
  ne16_pulp_hci_setpriority_ne16();
  ne16_pulp_hci_set_max_stall(conf->max_stall);
}

void ne16_pulp_close() {
  ne16_pulp_hci_reset_max_stall();
  ne16_pulp_hci_setpriority_core();
  ne16_pulp_cg_disable();
}

void ne16_pulp_event_wait_and_clear() {
  eu_evt_maskWaitAndClr(NE16_PULP_EVENT);
}

static const ne16_dev_t ne16_pulp_dev = {
    .hwpe_dev = (struct hwpe_dev_t){
        .base_addr = (volatile uint32_t *)NE16_PULP_BASE_ADDR}};

const ne16_dev_t *ne16_pulp_get_dev() { return &ne16_pulp_dev; }
