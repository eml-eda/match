#ifdef __spatz__
#ifndef CAR_LIB_SPATZ_H
#define CAR_LIB_SPATZ_H

#include <stdint.h>

#include "match/ctx.h"
#include "carfield.h"

#include "spatz.h"

#include "carfield_lib/dma.h"

#define MEM_L1_SIZE 32768 * 2 // 128kB

extern const uint8_t __l2_common_start[];
extern const uint8_t __l2_common_end[];

#define offload_args ((volatile uint32_t*)__l2_common_start)

extern volatile dma_transfer_id_t dma_transfer_;
extern volatile void* im2col_pt_;
extern volatile void* pwt_pt_;
extern volatile void* l1_scratchpad_pt_;

// Cluster-Host Synchronization

void spatz_wait_for_task_poll(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id);
void spatz_end_of_task_poll(uint32_t task_id);

void spatz_wait_for_task_mbox(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id);
void spatz_end_of_task_mbox(uint32_t task_id);

// SMP

int spatz_check_should_run();
int spatz_check_main_core(MatchCtx* ctx);
void spatz_sync_cores(MatchCtx* ctx);

// Profiling

void spatz_timer_start();
uint32_t spatz_timer_stop();

// Spatz Init

void spatz_startup();
void spatz_init(MatchCtx* ctx);
void spatz_free(MatchCtx* ctx);

void* spatz_l1_init(MatchCtx* ctx);

void spatz_l1_free(MatchCtx* ctx, void* l1_memory_pt);

void spatz_alloc_buffer(const char* name, int tensor_l1_pt, int size, int mem, int buffer_idx);

// Spatz DMA

int handle_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int match_tensor_type,
    int ext_mem, int int_mem 
);

void wait_l1_dma_transfers(MatchCtx* ctx);

// Utils

void smp_printf(const char* fmt, ...);

// Kernel Wrappers

void kernel_wrapper(MatchCtx* ctx);

// Spatz Kernel Wrappers

void spatz_fp16_conv2d_wrapper(MatchCtx* ctx);
void spatz_fp16_dense_wrapper(MatchCtx* ctx);

// Debug Flags

#define DEBUG_SPATZ_LIB             0
#define DEBUG_CALLOC_L1_SCRATCHPAD  0
#define DEBUG_BLOCKING_DMA          0
#define DEBUG_COUNT_CORE_SYNCS      0

#define ALLOC_L1_ONCE               1


typedef _Float16 fp16;


#endif // CAR_LIB_SPATZ_H
#endif // __spatz__