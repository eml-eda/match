/*
 * Copyright (C) 2022-2023 ETH Zurich and University of Bologna
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
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Author: Yvan Tortorella  <yvan.tortorella@unibo.it>
 *
 * RedMulE Hardware Abstraction Layer (HAL)
 */
#ifdef __pulp_cluster__
#ifndef __HAL_REDMULE_H__
#define __HAL_REDMULE_H__

#include <pulp.h>

/*
 * 
 * For control, generic configuration register layout,
 * and job-dependent register map, look at redmule_archi.h
 *
 */

// For all the following functions we use __builtin_pulp_OffsetedWrite and __builtin_pulp_OffsetedRead
// instead of classic load/store because otherwise the compiler is not able to correctly factorize
// the HWPE base in case several accesses are done, ending up with twice more code

#define HWPE_WRITE(value, offset) *(volatile int *)(ARCHI_CLUST_HWPE_BASE + offset) = value
#define HWPE_READ(offset) *(volatile int *)(ARCHI_CLUST_HWPE_BASE + offset)

static inline void redmule_x_add_set (unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_MARITH0);
}

static inline void redmule_w_add_set (unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_MARITH1);
}

static inline void redmule_z_add_set (unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_MARITH2);
}

static inline void redmule_mcnfig_set (uint32_t mcnfig0, uint32_t mcnfig1, uint32_t mcnfig2) {
  HWPE_WRITE(mcnfig0, REDMULE_REG_OFFS + REDMULE_MCNFIG0);
  HWPE_WRITE(mcnfig1, REDMULE_REG_OFFS + REDMULE_MCNFIG1);
  HWPE_WRITE(mcnfig2, REDMULE_REG_OFFS + REDMULE_MCNFIG2);
}

static inline unsigned int redmule_mopcnt_get(void) {
  return HWPE_READ(REDMULE_REG_OFFS + REDMULE_MOPCNT);
}

static inline void hwpe_trigger_job() {
  HWPE_WRITE(0, REDMULE_TRIGGER);
}

static inline int hwpe_acquire_job() {
  return HWPE_READ(REDMULE_ACQUIRE);
}

static inline unsigned int hwpe_get_status() {
  return HWPE_READ(REDMULE_STATUS);
}

static inline unsigned int hwpe_get_running_job() {
  return HWPE_READ(REDMULE_RUNNING_JOB);
}

static inline void hwpe_soft_clear() {
  HWPE_WRITE(0, REDMULE_SOFT_CLEAR);
}

static inline void hwpe_cg_enable() {
  *(volatile int*) (ARCHI_CLUST_CTRL_BASE + CLUST_CTRL_HWPE_EN) |= CLUST_CTRL_HWPE_EN_MASK;
}

static inline void hwpe_cg_disable() {
  *(volatile int*) (ARCHI_CLUST_CTRL_BASE + CLUST_CTRL_HWPE_EN) &= ~CLUST_CTRL_HWPE_EN_MASK;
}

static inline void redmule_evt_wait() {
  do {
    eu_evt_maskWaitAndClr (1 << ARCHI_CL_HWPE_EVT0);
  } while((*(int volatile *)(ARCHI_CLUST_HWPE_BASE + REDMULE_STATUS)) != 0);
}

static inline int hwpe_wait_acquire() {
  int job_id = hwpe_acquire_job();
  while(job_id < 0) {
    eu_evt_maskWaitAndClr (1 << ARCHI_CL_HWPE_EVT0);
    job_id = hwpe_acquire_job();
  }
  return job_id;
}

static void redmule_init() {
    hwpe_cg_enable();
    asm volatile("" : : : "memory");

    hwpe_soft_clear();
    asm volatile("" : : : "memory");

    int offload_id_tmp;
    do {
        offload_id_tmp = hwpe_acquire_job();
    } while(offload_id_tmp < 0);
    asm volatile("" : : : "memory");
}



static void redmule_start() {
    hwpe_trigger_job();
    asm volatile("fence r,rw": : :"memory");
}


static void redmule_wait() {
    redmule_evt_wait();
    asm volatile("" : : : "memory");
    hwpe_cg_disable();
    asm volatile("" : : : "memory");
}


static void redmule_cfg(unsigned int x, unsigned int w, unsigned int z, uint16_t m_size, uint16_t n_size,
  uint16_t k_size, uint8_t gemm_op, uint8_t gemm_fmt) {

  uint32_t mcnfig0 = ((uint32_t)k_size << 16) | (uint32_t)m_size;
  uint32_t mcnfig1 = ((uint32_t)(gemm_fmt & 0x3) << REDMULE_MCNFIG1_OUTPUT_FMT_SHIFT)
                   | ((uint32_t)(gemm_fmt & 0x3) << REDMULE_MCNFIG1_INPUT_FMT_SHIFT)
                   | ((uint32_t)(gemm_op  & 0x7) << REDMULE_MCNFIG1_GEMM_OPS_SHIFT)
                   | (uint32_t)(n_size & 0xFFFF);

  redmule_x_add_set((unsigned int)x);
  redmule_w_add_set((unsigned int)w);
  redmule_z_add_set((unsigned int)z);
  redmule_mcnfig_set(mcnfig0, mcnfig1, 0);
}

#endif
#endif  // __pulp_cluster__