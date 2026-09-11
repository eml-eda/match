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
#ifndef __ARCHI_REDMULE_H__
#define __ARCHI_REDMULE_H__
#include "pulp.h"
/*
 * |========================================================================|
 * ||                                                                      ||
 * ||Control and generic configuration register layout                     ||
 * |========================================================================|
 * || # reg |  offset  |  bits   |   bitmask    ||  content                ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    0  |  0x0000  |   1: 0  |  0x00000003  ||  COMMIT_TRIGGER         ||
 * ||    1  |  0x0004  |  31: 0  |  0xFFFFFFFF  ||  ACQUIRE                ||
 * ||    2  |  0x0008  |  31: 0  |  0xFFFFFFFF  ||  (reserved)             ||
 * ||    3  |  0x000c  |  31: 0  |  0xFFFFFFFF  ||  STATUS                 ||
 * ||    4  |  0x0010  |   7: 0  |  0x000000FF  ||  RUNNING_JOB            ||
 * ||    5  |  0x0014  |   1: 0  |  0x00000003  ||  SOFT_CLEAR             ||
 * |========================================================================|
 * ||                                                                      ||
 * ||Job-dependent registers layout                                        ||
 * |========================================================================|
 * || # reg |  offset  |  bits   |   bitmask    ||  content                ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    0  |  0x0020  |         |              ||  MCNFIG0                ||
 * ||       |          |  31:16  |  0xFFFF0000  ||  K Size (W Columns)     ||
 * ||       |          |  15: 0  |  0x0000FFFF  ||  M Size (X Rows)        ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    1  |  0x0024  |         |              ||  MCNFIG1                ||
 * ||       |          |  26:25  |  0x06000000  ||  Output format          ||
 * ||       |          |  24:23  |  0x01800000  ||  Input format           ||
 * ||       |          |  22:20  |  0x00700000  ||  Operation selection    ||
 * ||       |          |     19  |  0x00080000  ||  send_w                 ||
 * ||       |          |     18  |  0x00040000  ||  receive_w              ||
 * ||       |          |     17  |  0x00020000  ||  send_x                 ||
 * ||       |          |     16  |  0x00010000  ||  receive_x              ||
 * ||       |          |  15: 0  |  0x0000FFFF  ||  N Size (X Cols/W Rows) ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    2  |  0x0028  |  31: 0  |  0xFFFFFFFF  ||  MCNFIG2 (Y offset)    ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    3  |  0x002c  |  31: 0  |  0xFFFFFFFF  ||  MARITH0 (X base addr) ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    4  |  0x0030  |  31: 0  |  0xFFFFFFFF  ||  MARITH1 (W base addr) ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    5  |  0x0034  |  31: 0  |  0xFFFFFFFF  ||  MARITH2 (Z base addr) ||
 * ||-------+----------+---------+--------------++-------------------------||
 * ||    6  |  0x0038  |  31: 0  |  0xFFFFFFFF  ||  MOPCNT (ops complete) ||
 * |========================================================================|
 *
 */

/* PULP Cluster Archi defines */
#define ARCHI_CLUST_CTRL_BASE ARCHI_CLUSTER_CTRL_ADDR
#define ARCHI_CLUST_HWPE_BASE ARCHI_HWCE_ADDR
#define DMA_COMMAND_QUEUE     ARCHI_MCHAN_DEMUX_ADDR
#define DMA_STATUS_REGISTER   (ARCHI_MCHAN_DEMUX_ADDR + 4)
#define ARCHI_CL_HWPE_EVT0 12
#define ARCHI_CL_HWPE_EVT1 13
#define FC_DMA_EVENT 8
#define CL_DMA_EVENT 22
#define CLUST_CTRL_HWPE_EN 0x18
#define CLUST_CTRL_HWPE_EN_MASK 0x800
#define __builtin_bitinsert(a,b,c,d) (a | (((b << (32-c)) >> (32-c)) << d))

// RedMulE architecture
#define ADDR_WIDTH   32
#define DATA_WIDTH   256
#define REDMULE_FMT  16
#define ARRAY_HEIGHT 4
#define PIPE_REGS    3
#define ARRAY_WIDTH  12 /* Superior limit is ARRAY_HEIGHT*PIPE_REGS */

// Commands
#define REDMULE_TRIGGER     0x00
#define REDMULE_ACQUIRE     0x04
#define REDMULE_FINISHED    0x08
#define REDMULE_STATUS      0x0C
#define REDMULE_RUNNING_JOB 0x10
#define REDMULE_SOFT_CLEAR  0x14

// Job-dependent registers (base offset)
#define REDMULE_REG_OFFS 0x20

// Register offsets relative to REDMULE_REG_OFFS
#define REDMULE_MCNFIG0  0x00
#define REDMULE_MCNFIG1  0x04
#define REDMULE_MCNFIG2  0x08
#define REDMULE_MARITH0  0x0c
#define REDMULE_MARITH1  0x10
#define REDMULE_MARITH2  0x14
#define REDMULE_MOPCNT   0x18

// MCNFIG1 bit-field shifts
#define REDMULE_MCNFIG1_OUTPUT_FMT_SHIFT  25
#define REDMULE_MCNFIG1_INPUT_FMT_SHIFT   23
#define REDMULE_MCNFIG1_GEMM_OPS_SHIFT    20
#define REDMULE_MCNFIG1_SEND_W_SHIFT      19
#define REDMULE_MCNFIG1_RECEIVE_W_SHIFT   18
#define REDMULE_MCNFIG1_SEND_X_SHIFT      17
#define REDMULE_MCNFIG1_RECEIVE_X_SHIFT   16

// OPs definition
#define MATMUL 0x0
#define GEMM   0x1
#define ADDMAX 0x2
#define ADDMIN 0x3
#define MULMAX 0x4
#define MULMIN 0x5
#define MAXMIN 0x6
#define MINMAX 0x7

// GEMM formats
#define Float8     0x0
#define Float16    0x1
#define Float8Alt  0x2
#define Float16Alt 0x3

#define RNE       0x0
#define RTZ       0x1
#define OP_FMADD  0x0
#define OP_ADD    0x2
#define OP_MUL    0x3
#define OP_MINMAX 0x7

// FP Formats encoding
#define FP16    0x2
#define FP8     0x3
#define FP16ALT 0x4
#define FP8ALT  0x5

/* DMA Archi */
#define DMA_TX  0
#define DMA_RX  1
#define DMA_INC 1

#define PLP_DMA_TYPE_BIT    0x00000011
#define PLP_DMA_INCR_BIT    0x00000012
#define PLP_DMA_2D_BIT      0x00000013
#define PLP_DMA_ELE_BIT     0x00000014
#define PLP_DMA_ILE_BIT     0x00000015
#define PLP_DMA_BLE_BIT     0x00000016
#define PLP_DMA_2D_TCDM_BIT 0x0000017

#endif
#endif // __pulp_cluster__