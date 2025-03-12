#include <carus_helper/carus_helper.h>

void* carus_l1_mem_init(MatchCtx* ctx){
    err = l1_hal_init(0, l1_loader, sizeof(l1_loader), L1_TRANSFER_MODE_SAFE);
    if (err) exit(L1_ALLOC_FAIL)
    return EMEM_START_ADDRESS;
}

void carus_mem_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int match_tensor_type,
    int ext_mem, int int_mem
){
    int arcane_tensor_code = m0;
    if (match_tensor_type==MATCH_CONST_TENSOR) arcane_tensor_code = m1;
    if (match_tensor_type==MATCH_OUTPUT_TENSOR) arcane_tensor_code = m2;
    xmr(arcane_tensor_code, tensor->pts[ARCANE_L2_MEM], 1, tensor->tiles[ARCANE_L1_MEM*2+0].size, tensor->tiles[ARCANE_L1_MEM*2+1].size);
}

void carus_compute_wrapper(MatchCtx* ctx){
    carus_mmul_tiling(m2, m0, m1, mNONE, 0, 0);
}