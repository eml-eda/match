#ifndef __CARUS_HELPER_H__
#define __CARUS_HELPER_H__

#include <match/ctx.h>
#include <stdlib.h>
#include <carus_l1/l1_hal.h>
#include <carus_l1/l1_bare.h>
#include <carus_l1/l1_loader.h>
#include <carus_l1/l1_kernels.h>

#define m0 0
#define m1 1
#define m2 2
#define mNONE 255
void* carus_l1_mem_init(MatchCtx* ctx);

void carus_mem_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int match_tensor_type,
    int ext_mem, int int_mem
);

void carus_compute_wrapper(MatchCtx* ctx);

#endif 