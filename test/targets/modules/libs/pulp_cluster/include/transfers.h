#ifndef __PULP_CLUSTER_TENSOR_TRANSFER_H__
#define __PULP_CLUSTER_TENSOR_TRANSFER_H__

#include <pulp_cluster/match_dev.h>
#include <pulp_mem/dma.h>

void handle_dma_transfer_1d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
);
void handle_dma_transfer_2d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
);
void handle_dma_transfer_3d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
);
void handle_dma_transfer_4d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
);
void handle_dma_transfer_5d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
);

#endif // __PULP_CLUSTER_TENSOR_TRANSFER_H__