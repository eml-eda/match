#ifndef __MATCH_SAURIA_H__
#define __MATCH_SAURIA_H__

#include "match/ctx.h"
#include "sauria_regs.h"
#include "sauria.h"
#include "dif/dma.h"
#include "astral.h"
#include "approx_output_tensor_0.h"  //Change with proper header file name
#include <stdint.h>
#include <stddef.h>

#define SAURIA_PERIPH_START_ADDRESS 0x21004000
#define SAURIA_START_ADDRESS 0x52000000

#define L2_SZ(nd, i) (tensor->tiles[0 * (nd) + (i)].size)
#define L1_SZ(nd, i) (tensor->tiles[1 * (nd) + (i)].size)

// #define __DEBUG_DMA__
// #define __DEBUG_COMPUTE__

void __init_sauria__(MatchCtx* ctx, void (*__inner_node_fn__)(void *), unsigned int **args);

void sauria_1d_dma_transfer(void*src, void *dst, size_t size);


void* __sauria_scratchpad_get_weights_bank__(MatchCtx* ctx);
void* __sauria_scratchpad_get_ifmap_bank__(MatchCtx* ctx);
void* __sauria_scratchpad_get_output_bank__(MatchCtx* ctx);

void __sauria_wait_dma_transfer__(MatchCtx* ctx);

int sauria_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem 
);

void sauria_conv2d_wrapper(MatchCtx* ctx);
void sauria_conv3d_wrapper(MatchCtx* ctx);
void sauria_conv3d_fused_wrapper(MatchCtx* ctx);

void execute_sauria_conv2d(
    uint16_t inp_c, uint16_t inp_h, uint16_t inp_w,
    uint16_t fil_h, uint16_t fil_w,
    uint16_t out_c, uint16_t out_h, uint16_t out_w,
    uint16_t stride_x, uint16_t dil_x, bool preload
);

void __sauria_compute__(MatchCtx* ctx);

#endif  /*__MATCH_SAURIA_H__*/