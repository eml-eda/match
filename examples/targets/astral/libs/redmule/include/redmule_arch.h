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
 * High-level architecture of RedMulE
 *
 */

#ifdef __pulp_cluster__
#ifndef __REDMULE_ARCH_H__
#define __REDMULE_ARCH_H__

#include "pulp.h"

/*
 * |========================================================================|
 * ||                                                                      ||
 * ||Control and generic configuration register layout                     ||
 * |========================================================================|
 * || # reg |  offset  |  bits   |   bitmask    ||  content                ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    0  |  0x0000  |  31: 0  |  0xFFFFFFFF  ||  TRIGGER                ||
 * ||    1  |  0x0004  |  31: 0  |  0xFFFFFFFF  ||  ACQUIRE                ||
 * ||    2  |  0x0008  |  31: 0  |  0xFFFFFFFF  ||  EVT_ENABLE             ||
 * ||    3  |  0x000c  |  31: 0  |  0xFFFFFFFF  ||  STATUS                 ||
 * ||    4  |  0x0010  |  31: 0  |  0xFFFFFFFF  ||  RUNNING_JOB            ||
 * ||    5  |  0x0014  |  31: 0  |  0xFFFFFFFF  ||  SOFT_CLEAR             ||
 * |========================================================================|
 * ||                                                                      ||
 * ||Job-dependent registers layout                                        ||
 * |========================================================================|
 * || # reg |  offset  |  bits   |   bitmask    ||  content                ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    0  |  0x0040  |  31: 0  |  0xFFFFFFFF  ||  X_ADDR                 ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    1  |  0x0044  |  31: 0  |  0xFFFFFFFF  ||  W_ADDR                 ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    2  |  0x0048  |  31: 0  |  0xFFFFFFFF  ||  Z_ADDR                 ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    3  |  0x004C  |         |              ||  Matrix Config 0 Reg    ||
 * ||       |          |  31:16  |  0xFFFF0000  ||  K Size (W Columns)     ||
 * ||       |          |  15: 0  |  0x0000FFFF  ||  M Size (X Rows)        ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    4  |  0x0050  |         |              ||  Matrix Config 1 Reg    ||
 * ||       |          |  31: 0  |  0xFFFFFFFF  ||  N Size (X Cols/W Rows) ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    5  |  0x0054  |         |              ||  Matrix Arithmetic Reg  ||
 * ||       |          |  12:10  |  0x00001C00  ||  Operation selection    ||
 * ||       |          |   9: 7  |  0x00000380  ||  Input/Output format    ||
 * |========================================================================|
 *
 */

/* PULP Cluster Archi defines */
#define ARCHI_CLUST_CTRL_BASE   ARCHI_CLUSTER_CTRL_ADDR
#define REDMULE_HWPE_BASE       ARCHI_HWCE_ADDR
#define REDMULE_EU_HWPE_EVT0    12
#define REDMULE_EU_HWPE_EVT1    13
#define CLUST_CTRL_HWPE_EN      0x18
#define CLUST_CTRL_HWPE_EN_MASK 0x800

// RedMulE architecture
#define REDMULE_ADDR_WIDTH      32
#define REDMULE_DATA_WIDTH      256
#define REDMULE_FMT             16
#define REDMULE_ARRAY_HEIGHT    4
#define REDMULE_PIPE_REGS       3
#define REDMULE_ARRAY_WIDTH     12

// Commands
#define REDMULE_TRIGGER         0x00
#define REDMULE_ACQUIRE         0x04
#define REDMULE_FINISHED        0x08
#define REDMULE_STATUS          0x0C
#define REDMULE_RUNNING_JOB     0x10
#define REDMULE_SOFT_CLEAR      0x14

// Registers
#define REDMULE_REG_OFFS        0x40
#define REDMULE_REG_X_PTR       0x00
#define REDMULE_REG_W_PTR       0x04
#define REDMULE_REG_Z_PTR       0x08
#define REDMULE_MCFG0_PTR       0x0C
#define REDMULE_MCFG1_PTR       0x10
#define REDMULE_ARITH_PTR       0x14

// Supported operation types
enum redmule_op {
    REDMULE_OP_MATMUL           = 0x0,
    REDMULE_OP_GEMM             = 0x1,
    REDMULE_OP_ADDMAX           = 0x2,
    REDMULE_OP_ADDMIN           = 0x3,
    REDMULE_OP_MULMAX           = 0x4,
    REDMULE_OP_MULMIN           = 0x5,
    REDMULE_OP_MAXMIN           = 0x6,
    REDMULE_OP_MINMAX           = 0x7
};

// Supported operation formats
enum redmule_op_fmt {
    REDMULE_OP_FMT_FP8          = 0x0,
    REDMULE_OP_FMT_FP16         = 0x1,
    REDMULE_OP_FMT_FP8ALT       = 0x2,
    REDMULE_OP_FMT_FP16ALT      = 0x3
};

#endif // __REDMULE_ARCH_H__
#endif // __pulp_cluster__
