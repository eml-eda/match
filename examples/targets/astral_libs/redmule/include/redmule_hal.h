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
#ifndef __REDMULE_HAL_H__
#define __REDMULE_HAL_H__

#include <stdint.h>

#include "pulp.h"

#include "redmule/redmule_arch.h"


#if defined(__riscv__) && !defined(RV_ISA_RV32) && !defined(__LLVM__)

#define HWPE_WRITE(value, offset) \
    __builtin_pulp_OffsetedWrite((value), (int*)(REDMULE_HWPE_BASE), (offset))
#define HWPE_READ(offset) \
    __builtin_pulp_OffsetedRead((int*)(REDMULE_HWPE_BASE), (offset))

#else

#define HWPE_WRITE(value, offset)  \
    *(volatile int*)(REDMULE_HWPE_BASE + (offset)) = (value)
#define HWPE_READ(offset) \
    *(volatile int*)(REDMULE_HWPE_BASE + (offset))

#endif


static inline void redmule_set_x_addr(unsigned int value) {
    HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_X_PTR);
}

static inline void redmule_set_w_addr(unsigned int value) {
    HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_W_PTR);
}

static inline void redmule_set_z_addr(unsigned int value) {
    HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_Z_PTR);
}

static inline void redmule_set_mcfg(uint32_t mcfg0, uint32_t mcfg1) {
    HWPE_WRITE(mcfg0, REDMULE_REG_OFFS + REDMULE_MCFG0_PTR);
    HWPE_WRITE(mcfg1, REDMULE_REG_OFFS + REDMULE_MCFG1_PTR);
}

static inline void redmule_set_arith(uint32_t arith) {
    HWPE_WRITE(arith, REDMULE_REG_OFFS + REDMULE_ARITH_PTR);
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
    volatile int i;
    HWPE_WRITE(0, REDMULE_SOFT_CLEAR); 
}

static inline void hwpe_cg_enable() {
    *(volatile int*)(ARCHI_CLUST_CTRL_BASE + CLUST_CTRL_HWPE_EN) |= CLUST_CTRL_HWPE_EN_MASK;
}

static inline void hwpe_cg_disable() {
    *(volatile int*)(ARCHI_CLUST_CTRL_BASE + CLUST_CTRL_HWPE_EN) &= ~CLUST_CTRL_HWPE_EN_MASK;
}

static inline void redmule_evt_wait() {
    do {
        eu_evt_maskWaitAndClr(1 << REDMULE_EU_HWPE_EVT0);
    } while (hwpe_get_status() != 0);
}

static inline int hwpe_wait_acquire() {
    int job_id = hwpe_acquire_job();
    while (job_id < 0) {
        eu_evt_maskWaitAndClr(1 << REDMULE_EU_HWPE_EVT0);
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


static void redmule_config(
    uint32_t x_addr,  
    uint32_t w_addr,  
    uint32_t z_addr,
    uint32_t m_size, 
    uint32_t n_size, 
    uint32_t k_size,
    enum redmule_op gemm_op,
    enum redmule_op_fmt gemm_fmt
){
    uint32_t mcfg_reg0 = (k_size << 16) | (m_size << 0);
    
    uint32_t mcfg_reg1 = n_size << 0;
 
    uint32_t arith_reg = (gemm_op << 10) | (gemm_fmt << 7);

    redmule_set_x_addr(x_addr);
    redmule_set_w_addr(w_addr);
    redmule_set_z_addr(z_addr);
    redmule_set_mcfg(mcfg_reg0, mcfg_reg1);
    redmule_set_arith(arith_reg);
    asm volatile("fence rw,rw": : :"memory"); 
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


#endif  // __REDMULE_HAL_H__
#endif  // __pulp_cluster__
