#ifdef __spatz__

#include <stdint.h>
#include <stdarg.h>

#include "snrt.h"

#include "match/ctx.h"
#include "carfield.h"

#include "carfield_lib/spatz.h"
#include "carfield_lib/printf.h"
#include "carfield_lib/mbox.h"
#include "carfield_lib/utils.h"


volatile dma_transfer_id_t dma_transfer_ = 0;
volatile void* im2col_pt_ = NULL;
volatile void* pwt_pt_ = NULL;
volatile void* l1_scratchpad_pt_ = NULL;

#if DEBUG_COUNT_CORE_SYNCS
volatile int num_syncs[16] = {0};
#endif


int spatz_check_should_run() 
{
    return snrt_cluster_core_idx() < snrt_cluster_core_num();
}

int spatz_check_main_core(MatchCtx* ctx) 
{
    return snrt_cluster_core_idx() == 0;
}

void spatz_sync_cores(MatchCtx* ctx) 
{
    #if DEBUG_COUNT_CORE_SYNCS
    num_syncs[snrt_cluster_core_idx()]++;
    #endif

    asm volatile("fence rw,rw":::"memory");
    snrt_cluster_hw_barrier();

    #if DEBUG_COUNT_CORE_SYNCS
        if (snrt_cluster_core_idx() == 0) {
            mini_printf("[SPATZ][SYN] Per-core Barrier Count: ");
            for (int i = 0; i < snrt_cluster_core_num(); i++) {
                mini_printf("%d ", num_syncs[i]);
            }
            mini_printf("\r\n");
        } else {
            for (int i = 0; i < 300 + snrt_cluster_core_idx(); i++)
                asm volatile("fence rw,rw":::"memory");
        }
        snrt_cluster_hw_barrier();
        for (int i = 0; i < 300 + snrt_cluster_core_idx(); i++)
            asm volatile("fence rw,rw":::"memory");
    #endif
}

void spatz_startup() {
    mailbox_clear(HOST_TO_SPATZ_C0_MBOX + snrt_cluster_core_idx());

    #if DEBUG_SPATZ_LIB
    if (snrt_cluster_core_idx() == 0) {
        mini_printf("[SPATZ] Spatz startup...\r\n");
    }
    #endif
}

void spatz_init(MatchCtx* ctx)
{
    #if DEBUG_SPATZ_LIB
    for (int i = 0; i < 20000; i++) {
        asm volatile("fence rw,rw":::"memory");
    }
    #endif
    dma_transfer_ = dma_transfer_create();
    #if DEBUG_SPATZ_LIB
    mini_printf("[SPATZ] Yo! Spatz is alive! DMA counter is %d\r\n", dma_transfer_);
    #endif
}


void* spatz_l1_init(MatchCtx* ctx) {
    // TODO - check if snrt_l1alloc return 8 byte aligned but I think so
    
    #if DEBUG_SPATZ_LIB
    mini_printf("[SPATZ] Inizialing L1 Scratchpad...\r\n");
    #endif

    void *l1_memory_pt;

    #if ALLOC_L1_ONCE
    if (l1_scratchpad_pt_ == NULL) {
        l1_scratchpad_pt_ = snrt_l1alloc(MEM_L1_SIZE);   
    }
    l1_memory_pt = l1_scratchpad_pt_;
    #else
    l1_memory_pt = snrt_l1alloc(MEM_L1_SIZE);
    #endif

    #if DEBUG_CALLOC_MEM_L1
    for (int i = 0; i < MEM_L1_SIZE; i++)
        ((volatile char*)l1_memory_pt)[i] = 0;
    #endif

    #if DEBUG_SPATZ_LIB
    mini_printf("[SPATZ] Success.\r\n");
    #endif

    return l1_memory_pt;
}

void spatz_l1_free(MatchCtx* ctx, void* l1_memory_pt) {
    #if !ALLOC_L1_ONCE
    mini_printf("[SPATZ] snrt.h does not support l1 free.\r\n");
    #endif
}


void spatz_free(MatchCtx* ctx) 
{
    dma_transfer_free(dma_transfer_);
}


void spatz_alloc_buffer(const char* name, int tensor_l1_pt, int size, int mem, int buffer_idx)
{
    im2col_pt_ = (void*)tensor_l1_pt;
}

static void wait_l1_dma_transfers_impl(MatchCtx* ctx) {
    asm volatile("fence rw,rw":::"memory");
    dma_transfer_wait(dma_transfer_);
    asm volatile("fence rw,rw":::"memory");
    dma_transfer_ = dma_transfer_create();
    asm volatile("fence rw,rw":::"memory");
}


int handle_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int match_tensor_type,
    int ext_mem, int int_mem 
){
    asm volatile("fence rw,rw":::"memory");
    // shouldnt happen, we currently support only L2 and L1
    if(ext_mem!=MEM_L2 || int_mem!=MEM_L1_SPATZ)
        exit(1);
    // we should handle only 4-dims tensors
    if(tensor->num_dims>5)
        exit(1);

    if(!tensor->num_dims) return 0;

    int transferred_bytes = 0;

    #if DEBUG_SPATZ_LIB
    mini_printf("[SPATZ][DMA] DMA Transfer: %s(%p) %s %s(%p) - Tensor type: %s\r\n",
        ext_mem == MEM_L2 ? "L2" : "L1", tensor_l2_pt,
        match_transfer_type==MATCH_SW_LOAD_TENSOR ? "►" : "◄",
        int_mem == MEM_L1_SPATZ ? "L1" : "L2", tensor_l1_pt,
        match_tensor_type == MATCH_VAR_TENSOR ? "VAR" : (match_tensor_type == MATCH_CONST_TENSOR ? "CONST" : "OUT"));
    mini_printf("            Tile dim. sizes:");
    for(int idx=0; idx<tensor->num_dims; idx++) 
        mini_printf(" [L2: %d L1: %d]", tensor->tiles[MEM_L2*tensor->num_dims+idx].size, tensor->tiles[MEM_L1_SPATZ*tensor->num_dims+idx].size);
    mini_printf("\r\n");
    #endif

    switch(tensor->num_dims){
        case 1: {
            int bytes = tensor->tiles[MEM_L1_SPATZ*1+0].size * tensor->bits/8;
            #if DEBUG_SPATZ_LIB
            mini_printf("            1D transfer | Elem. Bytes: %d\r\n", tensor->bits/8);
            #endif
            dma_transfer_1d_async((dma_transfer_cfg_t) {
                .ext = tensor_l2_pt,
                .loc = tensor_l1_pt,
                .length_1d_copy = bytes,
                .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
            });
            transferred_bytes = bytes;
            break;
        }
        case 2: {
            int is_1d = tensor->tiles[MEM_L2*2+1].size==tensor->tiles[MEM_L1_SPATZ*2+1].size;
            int bytes = 0;
            #if DEBUG_SPATZ_LIB
            mini_printf("            2D transfer | Can 1D: %d | Elem. Bytes: %d\r\n", is_1d, tensor->bits/8);
            #endif
            if(is_1d){
                bytes = tensor->tiles[MEM_L1_SPATZ*2+0].size * tensor->tiles[MEM_L1_SPATZ*2+1].size * tensor->bits/8;
                dma_transfer_1d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .length_1d_copy = bytes,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else {
                bytes = tensor->tiles[MEM_L1_SPATZ*2+0].size * tensor->tiles[MEM_L1_SPATZ*2+1].size * tensor->bits/8;
                dma_transfer_2d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_1d_copies = tensor->tiles[MEM_L1_SPATZ*2+0].size,
                    .length_1d_copy = tensor->tiles[MEM_L1_SPATZ*2+1].size*tensor->bits/8,
                    .stride_1d = tensor->tiles[MEM_L2*2+1].size*tensor->bits/8,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            }
            transferred_bytes = bytes;
            break;
        }
        case 3: {
            int is_1d = tensor->tiles[MEM_L2*3+1].size==tensor->tiles[MEM_L1_SPATZ*3+1].size
                        && tensor->tiles[MEM_L2*3+2].size==tensor->tiles[MEM_L1_SPATZ*3+2].size;
            int is_2d = tensor->tiles[MEM_L2*3+2].size==tensor->tiles[MEM_L1_SPATZ*3+2].size;
            int bytes = 0;
            #if DEBUG_SPATZ_LIB
            mini_printf("            3D transfer | Can 1D: %d | Can 2D: %d | Elem. Bytes: %d\r\n", is_1d, is_2d, tensor->bits/8);
            #endif
            if(is_1d){
                bytes = tensor->tiles[MEM_L1_SPATZ*3+0].size*
                        tensor->tiles[MEM_L1_SPATZ*3+1].size*
                        tensor->tiles[MEM_L1_SPATZ*3+2].size*
                        tensor->bits/8;
                dma_transfer_1d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .length_1d_copy = bytes,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else if(is_2d){
                bytes = tensor->tiles[MEM_L1_SPATZ*3+0].size*
                        tensor->tiles[MEM_L1_SPATZ*3+1].size*
                        tensor->tiles[MEM_L1_SPATZ*3+2].size*
                        tensor->bits/8;
                dma_transfer_2d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_1d_copies = tensor->tiles[MEM_L1_SPATZ*3+0].size,
                    .length_1d_copy = tensor->tiles[MEM_L1_SPATZ*3+1].size*tensor->tiles[MEM_L1_SPATZ*3+2].size*tensor->bits/8,
                    .stride_1d = tensor->tiles[MEM_L2*3+1].size*tensor->tiles[MEM_L2*3+2].size*tensor->bits/8,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else {
                bytes = tensor->tiles[MEM_L1_SPATZ*3+0].size*
                        tensor->tiles[MEM_L1_SPATZ*3+1].size*
                        tensor->tiles[MEM_L1_SPATZ*3+2].size*
                        tensor->bits/8;
                dma_transfer_3d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_2d_copies = tensor->tiles[MEM_L1_SPATZ*3+0].size,
                    .number_of_1d_copies = tensor->tiles[MEM_L1_SPATZ*3+1].size,
                    .length_1d_copy = tensor->tiles[MEM_L1_SPATZ*3+2].size*tensor->bits/8,
                    .stride_1d = tensor->tiles[MEM_L2*3+2].size*tensor->bits/8,
                    .stride_2d = tensor->tiles[MEM_L2*3+1].size*tensor->tiles[MEM_L2*3+2].size*tensor->bits/8,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            }
            transferred_bytes = bytes;
            break;
        }
        case 4: {
            int is_hwc_to_chw = (/* ctx->pattern_name==depthwise_conv2d && */ 0 && match_tensor_type==MATCH_VAR_TENSOR && ctx->exec_module==PULP_CLUSTER);
            int is_1d = tensor->tiles[MEM_L2*4+1].size==tensor->tiles[MEM_L1_SPATZ*4+1].size
                        && tensor->tiles[MEM_L2*4+2].size==tensor->tiles[MEM_L1_SPATZ*4+2].size
                        && tensor->tiles[MEM_L2*4+3].size==tensor->tiles[MEM_L1_SPATZ*4+3].size;
            int is_2d = tensor->tiles[MEM_L2*4+2].size==tensor->tiles[MEM_L1_SPATZ*4+2].size
                        && tensor->tiles[MEM_L2*4+3].size==tensor->tiles[MEM_L1_SPATZ*4+3].size;
            int bytes = 0;
            #if DEBUG_SPATZ_LIB
            mini_printf("            4D transfer | HWC-to-CHW: %d | Can 1D: %d | Can 2D: %d | Elem. Bytes: %d\r\n",
                is_hwc_to_chw, is_1d, is_2d, tensor->bits/8);
            #endif
            if(is_hwc_to_chw){
                bytes = tensor->tiles[MEM_L1_SPATZ*4+1].size*
                        tensor->tiles[MEM_L1_SPATZ*4+2].size*
                        tensor->tiles[MEM_L1_SPATZ*4+3].size;
                dma_transfer_hwc_to_chw((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_2d_copies = tensor->tiles[MEM_L1_SPATZ*4+1].size,
                    .number_of_1d_copies = tensor->tiles[MEM_L1_SPATZ*4+2].size,
                    .length_1d_copy = tensor->tiles[MEM_L1_SPATZ*4+3].size,
                    .stride_2d = tensor->tiles[MEM_L2*4+3].size*tensor->tiles[MEM_L2*4+2].size,
                    .stride_1d = tensor->tiles[MEM_L2*4+3].size,
                    .dir = 1
                });
                bytes *= tensor->bits/8;
            } else if(is_1d){
                bytes = tensor->tiles[MEM_L1_SPATZ*4+0].size*
                        tensor->tiles[MEM_L1_SPATZ*4+1].size*
                        tensor->tiles[MEM_L1_SPATZ*4+2].size*
                        tensor->tiles[MEM_L1_SPATZ*4+3].size*
                        tensor->bits/8;
                dma_transfer_1d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .length_1d_copy = bytes,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else if(is_2d){
                bytes = tensor->tiles[MEM_L1_SPATZ*4+0].size*
                        tensor->tiles[MEM_L1_SPATZ*4+1].size*
                        tensor->tiles[MEM_L1_SPATZ*4+2].size*
                        tensor->tiles[MEM_L1_SPATZ*4+3].size*
                        tensor->bits/8;
                dma_transfer_2d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_1d_copies = tensor->tiles[MEM_L1_SPATZ*4+0].size,
                    .length_1d_copy = tensor->tiles[MEM_L1_SPATZ*4+1].size*
                                        tensor->tiles[MEM_L1_SPATZ*4+2].size*
                                        tensor->tiles[MEM_L1_SPATZ*4+3].size*
                                        tensor->bits/8,
                    .stride_1d = tensor->tiles[MEM_L2*4+1].size*
                                    tensor->tiles[MEM_L2*4+2].size*
                                    tensor->tiles[MEM_L2*4+3].size*
                                    tensor->bits/8,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else {
                bytes = tensor->tiles[MEM_L1_SPATZ*4+1].size*
                        tensor->tiles[MEM_L1_SPATZ*4+2].size*
                        tensor->tiles[MEM_L1_SPATZ*4+3].size*
                        tensor->bits/8;
                // this is per 3d transfer, but we are doing N of them
                for(int idx=0; idx<tensor->tiles[MEM_L1_SPATZ*4+0].size; idx++)
                    dma_transfer_3d_async((dma_transfer_cfg_t) {
                        .ext = tensor_l2_pt + idx*tensor->tiles[MEM_L2*4+1].size*
                                    tensor->tiles[MEM_L2*4+2].size*
                                    tensor->tiles[MEM_L2*4+3].size*
                                    tensor->bits/8,
                        .loc = tensor_l1_pt + idx*tensor->tiles[MEM_L1_SPATZ*4+1].size*
                                    tensor->tiles[MEM_L1_SPATZ*4+2].size*
                                    tensor->tiles[MEM_L1_SPATZ*4+3].size*
                                    tensor->bits/8,
                        .number_of_2d_copies = tensor->tiles[MEM_L1_SPATZ*4+1].size,
                        .number_of_1d_copies = tensor->tiles[MEM_L1_SPATZ*4+2].size,
                        .length_1d_copy = tensor->tiles[MEM_L1_SPATZ*4+3].size*
                                            tensor->bits/8,
                        .stride_1d = tensor->tiles[MEM_L2*4+3].size*
                                        tensor->bits/8,
                        .stride_2d = tensor->tiles[MEM_L2*4+2].size*
                                        tensor->tiles[MEM_L2*4+3].size*
                                        tensor->bits/8,
                        .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                    });
                bytes *= tensor->tiles[MEM_L1_SPATZ*4+0].size;
            }
            transferred_bytes = bytes;
            break;
        }
        case 5: {
            int bytes = 0;
            if(tensor->tiles[MEM_L2*5+1].dim==tensor->tiles[MEM_L1_SPATZ*5+4].dim){
                int is_1d = tensor->tiles[MEM_L2*5+1].size==tensor->tiles[MEM_L1_SPATZ*5+1].size
                        && tensor->tiles[MEM_L2*5+2].size==tensor->tiles[MEM_L1_SPATZ*5+2].size
                        && tensor->tiles[MEM_L2*5+3].size==tensor->tiles[MEM_L1_SPATZ*5+3].size;
                int is_2d = tensor->tiles[MEM_L2*5+2].size==tensor->tiles[MEM_L1_SPATZ*5+2].size
                        && tensor->tiles[MEM_L2*5+3].size==tensor->tiles[MEM_L1_SPATZ*5+3].size;
                #if DEBUG_SPATZ_LIB
                mini_printf("            5D transfer | Can 1D: %d | Can 2D: %d | Elem. Bytes: %d\r\n", is_1d, is_2d, tensor->bits/8);
                #endif
                if(is_1d){
                    bytes = tensor->tiles[MEM_L1_SPATZ*5+0].size*
                            tensor->tiles[MEM_L1_SPATZ*5+1].size*
                            tensor->tiles[MEM_L1_SPATZ*5+2].size*
                            tensor->tiles[MEM_L1_SPATZ*5+3].size*
                            tensor->bits/8;
                    dma_transfer_1d_async((dma_transfer_cfg_t) {
                        .ext = tensor_l2_pt,
                        .loc = tensor_l1_pt,
                        .length_1d_copy = bytes,
                        .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                    });
                } else if(is_2d){
                    bytes = tensor->tiles[MEM_L1_SPATZ*5+0].size*
                            tensor->tiles[MEM_L1_SPATZ*5+1].size*
                            tensor->tiles[MEM_L1_SPATZ*5+2].size*
                            tensor->tiles[MEM_L1_SPATZ*5+3].size*
                            tensor->bits/8;
                    dma_transfer_2d_async((dma_transfer_cfg_t) {
                        .ext = tensor_l2_pt,
                        .loc = tensor_l1_pt,
                        .number_of_1d_copies = tensor->tiles[MEM_L1_SPATZ*5+0].size,
                        .length_1d_copy = tensor->tiles[MEM_L1_SPATZ*5+1].size*
                                            tensor->tiles[MEM_L1_SPATZ*5+2].size*
                                            tensor->tiles[MEM_L1_SPATZ*5+3].size*
                                            tensor->bits/8,
                        .stride_1d = tensor->tiles[MEM_L2*5+1].size*
                                        tensor->tiles[MEM_L2*5+2].size*
                                        tensor->tiles[MEM_L2*5+3].size*
                                        tensor->bits/8,
                        .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                    });
                } else {
                    bytes = tensor->tiles[MEM_L1_SPATZ*5+1].size*
                            tensor->tiles[MEM_L1_SPATZ*5+2].size*
                            tensor->tiles[MEM_L1_SPATZ*5+3].size*
                            tensor->bits/8;
                    for(int idx=0; idx<tensor->tiles[MEM_L1_SPATZ*5+0].size; idx++)
                        dma_transfer_3d_async((dma_transfer_cfg_t) {
                            .ext = tensor_l2_pt + idx*tensor->tiles[MEM_L2*5+1].size*
                                        tensor->tiles[MEM_L2*5+2].size*
                                        tensor->tiles[MEM_L2*5+3].size*
                                        tensor->bits/8,
                            .loc = tensor_l1_pt + idx*tensor->tiles[MEM_L1_SPATZ*5+1].size*
                                        tensor->tiles[MEM_L1_SPATZ*5+2].size*
                                        tensor->tiles[MEM_L1_SPATZ*5+3].size*
                                        tensor->bits/8,
                            .number_of_2d_copies = tensor->tiles[MEM_L1_SPATZ*5+1].size,
                            .number_of_1d_copies = tensor->tiles[MEM_L1_SPATZ*5+2].size,
                            .length_1d_copy = tensor->tiles[MEM_L1_SPATZ*5+3].size*
                                                tensor->bits/8,
                            .stride_1d = tensor->tiles[MEM_L2*5+3].size*
                                            tensor->bits/8,
                            .stride_2d = tensor->tiles[MEM_L2*5+2].size*
                                            tensor->tiles[MEM_L2*5+3].size*
                                            tensor->bits/8,
                            .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                        });
                    bytes *= tensor->tiles[MEM_L1_SPATZ*5+0].size;
                }
                transferred_bytes = bytes;
            } else
                exit(1);
            break;
        }
    }


    #if DEBUG_BLOCKING_DMA
        wait_l1_dma_transfers_impl(ctx);
        #if DEBUG_SPATZ_LIB
            unsigned l2_crc = crc32(tensor_l2_pt, transferred_bytes);
            unsigned l1_crc = crc32(tensor_l1_pt, transferred_bytes);
            mini_printf("            Transferred %d bytes. CRC32 checksums: SRC = %p - DST = %p\r\n", 
                transferred_bytes,
                match_transfer_type == MATCH_SW_LOAD_TENSOR ? l2_crc : l1_crc, 
                match_transfer_type == MATCH_SW_LOAD_TENSOR ? l1_crc : l2_crc);
        #endif
    #else
        #if DEBUG_SPATZ_LIB
            unsigned l2_crc = crc32(tensor_l2_pt, transferred_bytes);
            unsigned l1_crc = crc32(tensor_l1_pt, transferred_bytes);
            mini_printf("            Transferred %d bytes. CRC32 checksums: SRC = %p\r\n", 
                transferred_bytes,
                match_transfer_type == MATCH_SW_LOAD_TENSOR ? l2_crc : l1_crc);
        #endif
    #endif

    return transferred_bytes;
}


void wait_l1_dma_transfers(MatchCtx* ctx) {
    #if DEBUG_BLOCKING_DMA
    ;
    #else
    wait_l1_dma_transfers_impl(ctx);
    #endif
}


void spatz_wait_for_task_poll(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id) {
    // Polling for the start signal
    while (offload_args[0] == 0xFFFFFFF0) {
        asm volatile("fence r,rw" ::: "memory");
    }
    *tensor_ptrs = offload_args+1;
    *task_id = offload_args[0];
    asm volatile("fence r,rw" ::: "memory");
}


void spatz_end_of_task_poll(uint32_t task_id) {
    // Set end signal
    asm volatile("fence rw,rw":::"memory");
    offload_args[0] = 0xFFFFFFF0;
    asm volatile("fence rw,rw":::"memory");
}


void spatz_wait_for_task_mbox(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id) {
    asm volatile("wfi" ::: "memory");
    mailbox_read(HOST_TO_SPATZ_C0_MBOX, tensor_ptrs, task_id);
    mailbox_clear(HOST_TO_SPATZ_C0_MBOX);
    asm volatile("fence rw,rw" ::: "memory");

    #if DEBUG_SPATZ_LIB
        mini_printf("[SPATZ] Received task %d with %p tensors\r\n", *task_id, *tensor_ptrs);
    #endif
}


void spatz_end_of_task_mbox(uint32_t task_id) {
    asm volatile("fence rw,rw" ::: "memory");
    mailbox_send(SPATZ_TO_HOST_MBOX, task_id, 0);
    asm volatile("fence rw,rw" ::: "memory");
}


static volatile uint32_t timer_cycles = 0;

void spatz_timer_start() {
    if (snrt_cluster_core_idx() == 0) {
        asm volatile("fence rw,rw" ::: "memory");
        timer_cycles = read_csr(mcycle);
        asm volatile("fence rw,rw" ::: "memory");
    }
}

uint32_t spatz_timer_stop() {
    if (snrt_cluster_core_idx() == 0) {
        asm volatile("fence rw,rw" ::: "memory");
        return read_csr(mcycle) - timer_cycles;
        asm volatile("fence rw,rw" ::: "memory");
    }
    return 0;
}


void smp_printf(const char* fmt, ...) {
    if (snrt_cluster_core_idx() == 0) {
        va_list args;
        va_start(args, fmt);
        mini_vprintf(fmt, args);
        va_end(args);
    }
}

/*
double __attribute__((weak)) __extendhfdf2(float16 val)
{
  float res;
  __asm__ __volatile__ ("fcvt.s.h %0, %1": "=f"(res): "f"(val) :);
  return (double) res;
}
*/


#endif