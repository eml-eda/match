#ifndef CAR_LIB_CLUSTER_H
#define CAR_LIB_CLUSTER_H

#include <stdint.h>

#include "match/ctx.h"
#include "carfield.h"

#include "pulp_cluster.h"

#include "carfield_lib/dma.h"

#ifdef __pulp_cluster__
#include "pulp.h"
#include "bench/bench.h"
#include "pulp_nn/pulp_nn_kernels.h"
#endif


#define L1_SCRATCHPAD_SIZE 32768 * 4

// #define CLUSTER_LIB_DEBUG

extern const uint8_t __l2_common_start[];
extern const uint8_t __l2_common_end[];

#define offload_args ((volatile uint32_t*)__l2_common_start)

extern volatile dma_transfer_id_t dma_transfer_;
extern volatile void* im2col_pt_;
extern volatile void* pwt_pt_;

void cluster_wait_for_task_poll(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id);
void cluster_end_of_task_poll(uint32_t task_id);

void cluster_wait_for_task_mbox(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id);
void cluster_end_of_task_mbox(uint32_t task_id);

int cluster_check_should_run();
int cluster_check_main_core(MatchCtx* ctx);
void cluster_sync_cores(MatchCtx* ctx);

void cluster_timer_start();
uint32_t cluster_timer_stop();

void cluster_alloc_buffer(const char* name, int tensor_l1_pt, int size, int mem, int buffer_idx);

void cluster_lib_init(MatchCtx* ctx);

void* init_l1_scratchpad_memory(MatchCtx* ctx);

void handle_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int match_tensor_type,
    int ext_mem, int int_mem 
);

void wait_l1_dma_transfers(MatchCtx* ctx);

void free_l1_scrachpad_memory(MatchCtx* ctx, void* l1_memory_pt);

void wait_pulp_nn_computation(MatchCtx* ctx);

void pulp_nn_dense_wrapper(MatchCtx* ctx);

void pulp_nn_dense_out_int_wrapper(MatchCtx* ctx);

void pulp_nn_dw_conv2d_less_4_wrapper(MatchCtx* ctx);

void pulp_nn_dw_conv2d_wrapper(MatchCtx* ctx);

void pulp_nn_pw_conv2d_wrapper(MatchCtx* ctx);

void pulp_nn_hoparallel_conv2d_wrapper(MatchCtx* ctx);

void pulp_nn_add_wrapper(MatchCtx* ctx);

void pulp_nn_wrapper(MatchCtx* ctx);

#endif // CAR_LIB_CLUSTER_H