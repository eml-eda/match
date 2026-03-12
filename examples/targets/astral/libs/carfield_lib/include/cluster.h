#ifdef __pulp_cluster__
#ifndef CAR_LIB_CLUSTER_H
#define CAR_LIB_CLUSTER_H

#include <stdint.h>

#include "match/ctx.h"
#include "carfield.h"

#include "pulp_cluster.h"

#include "carfield_lib/dma.h"

// could run the model directly in L1 so it can happen that no L2 is used, but the library still needs to compile
#ifndef __MEM_L2__
#define __MEM_L2__
#define MEM_L2 0
#endif

#define MEM_L1_SIZE 128 * 1024 // 128kB

extern const uint8_t __l2_host_mex_addr__[];

#define pulp_cluster_args ((volatile uint32_t*)__l2_host_mex_addr__)

extern volatile dma_transfer_id_t dma_transfer_;
extern volatile void* im2col_pt_;
extern volatile void* pwt_pt_;
extern volatile void* l1_scratchpad_pt_;

// Cluster-Host Synchronization

void cluster_wait_for_task_poll(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id);
void cluster_end_of_task_poll(uint32_t task_id);

void cluster_wait_for_task_mbox(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id);
void cluster_end_of_task_mbox(uint32_t task_id);

// SMP

int cluster_check_should_run();
int cluster_check_main_core(MatchCtx* ctx);
void cluster_sync_cores(MatchCtx* ctx);

void wait_pulp_nn_computation(MatchCtx* ctx);

// Profiling

void cluster_timer_start();
uint32_t cluster_timer_stop();

// Cluster Init

void cluster_lib_init(MatchCtx* ctx);

void* init_MEM_L1_PULPD_memory(MatchCtx* ctx);

void free_MEM_L1_PULPD_memory(MatchCtx* ctx, void* l1_memory_pt);

void cluster_alloc_buffer(const char* name, int tensor_l1_pt, int size, int mem, int buffer_idx);

// Pulp Cluster DMA

int handle_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem 
);

void wait_l1_dma_transfers(MatchCtx* ctx);


// Utils

void smp_printf(const char* fmt, ...);

// Kernel Wrappers

void kernel_wrapper(MatchCtx* ctx);

// Pulp NN Wrappers (int8)

void pulp_nn_dense_wrapper(MatchCtx* ctx);
void pulp_nn_dense_out_int_wrapper(MatchCtx* ctx);
void pulp_nn_dw_conv2d_less_4_wrapper(MatchCtx* ctx);
void pulp_nn_dw_conv2d_wrapper(MatchCtx* ctx);
void pulp_nn_pw_conv2d_wrapper(MatchCtx* ctx);
void pulp_nn_hoparallel_conv2d_wrapper(MatchCtx* ctx);
void pulp_nn_add_wrapper(MatchCtx* ctx);

// Pulp NN Wrappers (fp16)

void pulp_nn_conv3d_wrapper(void* args);

void pulp_nn_wrapper(MatchCtx* ctx);
void pulp_nn_fp16_conv2d_wrapper(MatchCtx* ctx);
void pulp_nn_fp16_conv2d_grouped_wrapper(MatchCtx* ctx);
void pulp_nn_fp16_dense_wrapper(MatchCtx* ctx);
void pulp_nn_fp16_batch_matmul_wrapper(MatchCtx* ctx);

// Redmule Wrappers (fp16)

void redmule_fp16_dense_wrapper(MatchCtx* ctx);
void redmule_fp16_gemm_wrapper(MatchCtx* ctx);
void redmule_fp16_matmul_wrapper(MatchCtx* ctx);

// Pulp Trainlib Wrappers (fp16)

// TODO


// Debug Flags

#define DEBUG_CLUSTER_LIB           0
#define DEBUG_CALLOC_L1_SCRATCHPAD  0
#define DEBUG_BLOCKING_DMA          0
#define DEBUG_COUNT_CORE_SYNCS      0

#define ALLOC_L1_ONCE               1

// Float16 typedef for cluster

typedef float16 fp16;
typedef fp16 v2f16 __attribute__((vector_size (4)));

#endif // CAR_LIB_CLUSTER_H
#endif // __pulp_cluster__