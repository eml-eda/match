#include <pulp_cluster/transfers.h>

void handle_dma_transfer_1d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
){
    #ifdef CLUSTER_LIB_DEBUG
    printf("1D transfer prec %d bytes\n", tensor->bits/8);
    #endif
    dma_transfer_1d_async((DmaTransferConf) {
        .ext = tensor_l2_pt,
        .loc = tensor_l1_pt,
        .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*1+0].size*tensor->bits/8,
        .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
    });
}

void handle_dma_transfer_2d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
){
    #ifdef CLUSTER_LIB_DEBUG
    printf("2D transfer 1D %d prec %d bytes\n", 
        tensor->tiles[L2_SHARED_MEM*2+1].size==tensor->tiles[L1_SCRATCHPAD*2+1].size,
        tensor->bits/8
    );
    #endif
    // check if we can do a 1D transfer
    if(tensor->tiles[L2_SHARED_MEM*2+1].size==tensor->tiles[L1_SCRATCHPAD*2+1].size)
        dma_transfer_1d_async((DmaTransferConf) {
            .ext = tensor_l2_pt,
            .loc = tensor_l1_pt,
            .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*2+0].size*tensor->tiles[L1_SCRATCHPAD*2+1].size*tensor->bits/8,
            .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
        });
    // resort to 2D transfers 
    else
        dma_transfer_2d_async((DmaTransferConf) {
            .ext = tensor_l2_pt,
            .loc = tensor_l1_pt,
            .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*2+0].size,
            .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*2+1].size*tensor->bits/8,
            .stride_1d = tensor->tiles[L2_SHARED_MEM*2+1].size*tensor->bits/8,
            .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
        });
}

void handle_dma_transfer_3d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
){
    #ifdef CLUSTER_LIB_DEBUG
    printf("3D transfer 1D %d 2D %d prec %d bytes\n", 
        tensor->tiles[L2_SHARED_MEM*3+1].size==tensor->tiles[L1_SCRATCHPAD*3+1].size
        && tensor->tiles[L2_SHARED_MEM*3+2].size==tensor->tiles[L1_SCRATCHPAD*3+2].size,
        tensor->tiles[L2_SHARED_MEM*3+2].size==tensor->tiles[L1_SCRATCHPAD*3+2].size,
        tensor->bits/8
    );
    #endif
    // check if we can do a 1D transfer
    if(tensor->tiles[L2_SHARED_MEM*3+1].size==tensor->tiles[L1_SCRATCHPAD*3+1].size
        && tensor->tiles[L2_SHARED_MEM*3+2].size==tensor->tiles[L1_SCRATCHPAD*3+2].size)
        dma_transfer_1d_async((DmaTransferConf) {
            .ext = tensor_l2_pt,
            .loc = tensor_l1_pt,
            .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*3+0].size*
                                tensor->tiles[L1_SCRATCHPAD*3+1].size*
                                tensor->tiles[L1_SCRATCHPAD*3+2].size*
                                tensor->bits/8,
            .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
        });
    // fallback to 2D if possible
    else if(tensor->tiles[L2_SHARED_MEM*3+2].size==tensor->tiles[L1_SCRATCHPAD*3+2].size)
        dma_transfer_2d_async((DmaTransferConf) {
            .ext = tensor_l2_pt,
            .loc = tensor_l1_pt,
            .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*3+0].size,
            .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*3+1].size*tensor->tiles[L1_SCRATCHPAD*3+2].size*tensor->bits/8,
            .stride_1d = tensor->tiles[L2_SHARED_MEM*3+1].size*tensor->tiles[L2_SHARED_MEM*3+2].size*tensor->bits/8,
            .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
        });
    // fallback to 3D
    else
        dma_transfer_3d_async((DmaTransferConf) {
            .ext = tensor_l2_pt,
            .loc = tensor_l1_pt,
            .number_of_2d_copies = tensor->tiles[L1_SCRATCHPAD*3+0].size,
            .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*3+1].size,
            .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*3+2].size*tensor->bits/8,
            .stride_1d = tensor->tiles[L2_SHARED_MEM*3+2].size*tensor->bits/8,
            .stride_2d = tensor->tiles[L2_SHARED_MEM*3+1].size*tensor->tiles[L2_SHARED_MEM*3+2].size*tensor->bits/8,
            .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
        });
}

void handle_dma_transfer_4d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
){
    #ifdef CLUSTER_LIB_DEBUG
    printf("4D transfer HWC_TO_CHW %d 1D %d 2D %d prec %d bytes\n", 
        ctx->pattern_name==depthwise_conv2d && tensor->tensor_type==MATCH_VAR_TENSOR
        && ctx->exec_module==PULP_CLUSTER,
        tensor->tiles[L2_SHARED_MEM*4+1].size==tensor->tiles[L1_SCRATCHPAD*4+1].size
        && tensor->tiles[L2_SHARED_MEM*4+2].size==tensor->tiles[L1_SCRATCHPAD*4+2].size
        && tensor->tiles[L2_SHARED_MEM*4+3].size==tensor->tiles[L1_SCRATCHPAD*4+3].size,
        tensor->tiles[L2_SHARED_MEM*4+2].size==tensor->tiles[L1_SCRATCHPAD*4+2].size
        && tensor->tiles[L2_SHARED_MEM*4+3].size==tensor->tiles[L1_SCRATCHPAD*4+3].size,
        tensor->bits/8
    );
    #endif
    // check if depthwise conv2d and activations
    if(ctx->pattern_name==depthwise_conv2d && tensor->tensor_type==MATCH_VAR_TENSOR
        && ctx->exec_module==PULP_CLUSTER)
        dma_transfer_hwc_to_chw((DmaTransferConf) {
            .ext = tensor_l2_pt,
            .loc = tensor_l1_pt,
            .number_of_2d_copies = tensor->tiles[L1_SCRATCHPAD*4+1].size,
            .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*4+2].size,
            .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*4+3].size,
            .stride_2d = tensor->tiles[L2_SHARED_MEM*4+3].size*tensor->tiles[L2_SHARED_MEM*4+2].size,
            .stride_1d = tensor->tiles[L2_SHARED_MEM*4+3].size,
            .dir = 1
        });
    // check if we can do a 1D transfer
    else if(tensor->tiles[L2_SHARED_MEM*4+1].size==tensor->tiles[L1_SCRATCHPAD*4+1].size
        && tensor->tiles[L2_SHARED_MEM*4+2].size==tensor->tiles[L1_SCRATCHPAD*4+2].size
        && tensor->tiles[L2_SHARED_MEM*4+3].size==tensor->tiles[L1_SCRATCHPAD*4+3].size)
        dma_transfer_1d_async((DmaTransferConf) {
            .ext = tensor_l2_pt,
            .loc = tensor_l1_pt,
            .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*4+0].size*
                                tensor->tiles[L1_SCRATCHPAD*4+1].size*
                                tensor->tiles[L1_SCRATCHPAD*4+2].size*
                                tensor->tiles[L1_SCRATCHPAD*4+3].size*
                                tensor->bits/8,
            .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
        });
    // fallback to 2D if possible
    else if( tensor->tiles[L2_SHARED_MEM*4+2].size==tensor->tiles[L1_SCRATCHPAD*4+2].size
        && tensor->tiles[L2_SHARED_MEM*4+3].size==tensor->tiles[L1_SCRATCHPAD*4+3].size)
        dma_transfer_2d_async((DmaTransferConf) {
            .ext = tensor_l2_pt,
            .loc = tensor_l1_pt,
            .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*4+0].size,
            .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*4+1].size*
                                tensor->tiles[L1_SCRATCHPAD*4+2].size*
                                tensor->tiles[L1_SCRATCHPAD*4+3].size*
                                tensor->bits/8,
            .stride_1d = tensor->tiles[L2_SHARED_MEM*4+1].size*
                            tensor->tiles[L2_SHARED_MEM*4+2].size*
                            tensor->tiles[L2_SHARED_MEM*4+3].size*
                            tensor->bits/8,
            .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
        });
    // fallback to 3D
    else
        for(int idx=0; idx<tensor->tiles[L1_SCRATCHPAD*4+0].size; idx++)
            dma_transfer_3d_async((DmaTransferConf) {
                .ext = tensor_l2_pt + idx*tensor->tiles[L2_SHARED_MEM*4+1].size*
                            tensor->tiles[L2_SHARED_MEM*4+2].size*
                            tensor->tiles[L2_SHARED_MEM*4+3].size*
                            tensor->bits/8,
                .loc = tensor_l1_pt + idx*tensor->tiles[L1_SCRATCHPAD*4+1].size*
                            tensor->tiles[L1_SCRATCHPAD*4+2].size*
                            tensor->tiles[L1_SCRATCHPAD*4+3].size*
                            tensor->bits/8,
                .number_of_2d_copies = tensor->tiles[L1_SCRATCHPAD*4+1].size,
                .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*4+2].size,
                .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*4+3].size*
                                    tensor->bits/8,
                .stride_1d = tensor->tiles[L2_SHARED_MEM*4+3].size*
                                tensor->bits/8,
                .stride_2d = tensor->tiles[L2_SHARED_MEM*4+2].size*
                                tensor->tiles[L2_SHARED_MEM*4+3].size*
                                tensor->bits/8,
                .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
            });
}

void handle_dma_transfer_5d(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int ext_mem, int int_mem
){
    // checking manually if the layout is NCHWc16 on NE16
    if(tensor->tiles[L2_SHARED_MEM*5+1].dim==tensor->tiles[L1_SCRATCHPAD*5+4].dim){
        // check if we can do a 1D transfer
        if(tensor->tiles[L2_SHARED_MEM*5+1].size==tensor->tiles[L1_SCRATCHPAD*5+1].size
            && tensor->tiles[L2_SHARED_MEM*5+2].size==tensor->tiles[L1_SCRATCHPAD*5+2].size
            && tensor->tiles[L2_SHARED_MEM*5+3].size==tensor->tiles[L1_SCRATCHPAD*5+3].size)
            dma_transfer_1d_async((DmaTransferConf) {
                .ext = tensor_l2_pt,
                .loc = tensor_l1_pt,
                .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*5+0].size*
                                    tensor->tiles[L1_SCRATCHPAD*5+1].size*
                                    tensor->tiles[L1_SCRATCHPAD*5+2].size*
                                    tensor->tiles[L1_SCRATCHPAD*5+3].size*
                                    tensor->bits/8,
                .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
            });
        // fallback to 2D if possible
        else if( tensor->tiles[L2_SHARED_MEM*5+2].size==tensor->tiles[L1_SCRATCHPAD*5+2].size
            && tensor->tiles[L2_SHARED_MEM*5+3].size==tensor->tiles[L1_SCRATCHPAD*5+3].size)
            dma_transfer_2d_async((DmaTransferConf) {
                .ext = tensor_l2_pt,
                .loc = tensor_l1_pt,
                .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*5+0].size,
                .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*5+1].size*
                                    tensor->tiles[L1_SCRATCHPAD*5+2].size*
                                    tensor->tiles[L1_SCRATCHPAD*5+3].size*
                                    tensor->bits/8,
                .stride_1d = tensor->tiles[L2_SHARED_MEM*5+1].size*
                                tensor->tiles[L2_SHARED_MEM*5+2].size*
                                tensor->tiles[L2_SHARED_MEM*5+3].size*
                                tensor->bits/8,
                .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
            });
        // fallback to 3D
        else
            for(int idx=0; idx<tensor->tiles[L1_SCRATCHPAD*5+0].size; idx++)
                dma_transfer_3d_async((DmaTransferConf) {
                    .ext = tensor_l2_pt + idx*tensor->tiles[L2_SHARED_MEM*5+1].size*
                                tensor->tiles[L2_SHARED_MEM*5+2].size*
                                tensor->tiles[L2_SHARED_MEM*5+3].size*
                                tensor->bits/8,
                    .loc = tensor_l1_pt + idx*tensor->tiles[L1_SCRATCHPAD*5+1].size*
                                tensor->tiles[L1_SCRATCHPAD*5+2].size*
                                tensor->tiles[L1_SCRATCHPAD*5+3].size*
                                tensor->bits/8,
                    .number_of_2d_copies = tensor->tiles[L1_SCRATCHPAD*5+1].size,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*5+2].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*5+3].size*
                                        tensor->bits/8,
                    .stride_1d = tensor->tiles[L2_SHARED_MEM*5+3].size*
                                    tensor->bits/8,
                    .stride_2d = tensor->tiles[L2_SHARED_MEM*5+2].size*
                                    tensor->tiles[L2_SHARED_MEM*5+3].size*
                                    tensor->bits/8,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
    }
    else{
        // if not NCHWc16, we should do a 4D transfer
        #ifdef CLUSTER_LIB_DEBUG
        printf("5D transfer is now naive doing a set of 3d transfers ALWAYS\n");
        #endif
        for(int first_dim_idx=0; first_dim_idx<tensor->tiles[L1_SCRATCHPAD*5+0].size; first_dim_idx++)
            for(int second_dim_idx=0; second_dim_idx<tensor->tiles[L1_SCRATCHPAD*5+1].size; second_dim_idx++)
                dma_transfer_3d_async((DmaTransferConf) {
                    .ext = tensor_l2_pt + first_dim_idx*tensor->tiles[L2_SHARED_MEM*5+1].size*
                                tensor->tiles[L2_SHARED_MEM*5+2].size*
                                tensor->tiles[L2_SHARED_MEM*5+3].size*
                                tensor->tiles[L2_SHARED_MEM*5+4].size*
                                tensor->bits/8 + second_dim_idx*tensor->tiles[L2_SHARED_MEM*5+2].size*
                                tensor->tiles[L2_SHARED_MEM*5+3].size*
                                tensor->tiles[L2_SHARED_MEM*5+4].size*
                                tensor->bits/8,
                    .loc = tensor_l1_pt + first_dim_idx*tensor->tiles[L1_SCRATCHPAD*5+1].size*
                                tensor->tiles[L1_SCRATCHPAD*5+2].size*
                                tensor->tiles[L1_SCRATCHPAD*5+3].size*
                                tensor->tiles[L1_SCRATCHPAD*5+4].size*
                                tensor->bits/8 + second_dim_idx*tensor->tiles[L1_SCRATCHPAD*5+2].size*
                                tensor->tiles[L1_SCRATCHPAD*5+3].size*
                                tensor->tiles[L1_SCRATCHPAD*5+4].size*
                                tensor->bits/8,
                    .number_of_2d_copies = tensor->tiles[L1_SCRATCHPAD*5+2].size,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*5+3].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*5+4].size*
                                        tensor->bits/8,
                    .stride_1d = tensor->tiles[L2_SHARED_MEM*5+4].size*
                                    tensor->bits/8,
                    .stride_2d = tensor->tiles[L2_SHARED_MEM*5+3].size*
                                    tensor->tiles[L2_SHARED_MEM*5+4].size*
                                    tensor->bits/8,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
    }
}