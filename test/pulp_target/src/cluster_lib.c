#include <pulp_target/cluster_lib.h>

void cluster_lib_init(MatchCtx* ctx){
    // alloc L1 memory
    l1_memory_pt_ = pi_cl_l1_malloc(NULL, 90*1024);
    im2col_size_ = 0;
    pwt_buffer_size_ = 0;
    pi_team_config_offload(NUM_CORES);
    dma_transfer_ = dma_transfer_create();
}

void* init_l1_scratchpad_memory(MatchCtx* ctx){
    return l1_memory_pt_ + im2col_size_ + pwt_buffer_size_;
}

void handle_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int match_tensor_type,
    int ext_mem, int int_mem 
){
    // shouldnt happen, we currently support only L2 and L1
    if(ext_mem!=L2_SHARED_MEM || int_mem!=L1_SCRATCHPAD)
        exit(1);
    // we should handle only 4-dims tensors
    if(tensor->num_dims>4)
        exit(1);
    
    switch(tensor->num_dims){
        case 1:
             dma_transfer_1d_async((DmaTransferConf) {
                .ext = tensor_l2_pt,
                .loc = tensor_l1_pt,
                .length_1d_copy = tensor->tiles[L1_SCRATCHPAD][0].size,
                .dir = match_transfer_type==MATCH_SW_STORE_TENSOR
            });
            break
        case 2:
            // check if we can do a 1D transfer
            if(tensor->dims[1].size==tensor->tiles[L1_SCRATCHPAD][1].size)
                dma_transfer_1d_async((DmaTransferConf) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD][0].size*tensor->tiles[L1_SCRATCHPAD][1].size,
                    .dir = match_transfer_type==MATCH_SW_STORE_TENSOR
                });
            // resort to 2D transfers 
            else
                dma_transfer_2d_async((DmaTransferConf) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD][0].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD][1].size,
                    .stride_1d = tensor->tiles[L2_SHARED_MEM][0].size,
                    .dir = match_transfer_type==MATCH_SW_STORE_TENSOR
                });
            break
        case 3:
            // check if we can do a 1D transfer
            if(tensor->dims[1].size==tensor->tiles[L1_SCRATCHPAD][1].size
                && tensor->dims[2].size==tensor->tiles[L1_SCRATCHPAD][2].size)
                dma_transfer_1d_async((DmaTransferConf) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD][0].size*
                                        tensor->tiles[L1_SCRATCHPAD][1].size*
                                        tensor->tiles[L1_SCRATCHPAD][2].size,
                    .dir = match_transfer_type==MATCH_SW_STORE_TENSOR
                });
            // fallback to 2D if possible
            else if(tensor->dims[2].size==tensor->tiles[L1_SCRATCHPAD][2].size)
                dma_transfer_2d_async((DmaTransferConf) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD][0].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD][1].size,
                    .stride_1d = tensor->tiles[L2_SHARED_MEM][0].size,
                    .dir = match_transfer_type==MATCH_SW_STORE_TENSOR
                });
            // fallback to 3D
            else
                dma_transfer_2d_async((DmaTransferConf) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD][0].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD][1].size,
                    .stride_1d = tensor->tiles[L2_SHARED_MEM][0].size,
                    .dir = match_transfer_type==MATCH_SW_STORE_TENSOR
                });
            break
        default:
            // check if we can do a 1D transfer
            // fallback to 2D if possible
            // fallback to 3D if possible
            // otherwise do a bunch of 3D transfers
            break
    }
}

void wait_l1_dma_transfers(MatchCtx* ctx){
    dma_transfer_wait(dma_transfer_);
    dma_transfer_ = dma_transfer_create();
}

void free_l1_scrachpad_memory(MatchCtx* ctx, void* l1_memory_pt){
    pi_cl_l1_free(NULL, l1_memory_pt_, 90*1024);
}

void wait_pulp_nn_computation(MatchCtx* ctx){
    pi_team_offload_wait();
}

void pulp_nn_dense_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    pulp_nn_linear(
        // activations pt  
        tensors[0]->pts[L1_SCRATCHPAD], // acts pt
        // bias pt
        tensors[4]->pts[L1_SCRATCHPAD], // bias pt
        // output pt
        tensors[3]->pts[L1_SCRATCHPAD], // output pt
        // weights pt
        tensors[1]->pts[L1_SCRATCHPAD], // weights pt
        tensors[2]->pts[L1_SCRATCHPAD], // bnorm mul pt
        tensors[3]->pts[L1_SCRATCHPAD], // bnorm add pt
        1, // requant mult factor
        *tensors[4]->data, // requant shift factor
        tensors[1]->tiles[L1_SCRATCHPAD][1].size, // input channels
        tensors[1]->tiles[L1_SCRATCHPAD][0].size, // output channels
        1, // activation is on
        1 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_dense_out_int_wrapper(void* args){}

void pulp_nn_dw_conv2d_less_4_wrapper(void* args){}

void pulp_nn_dw_conv2d_wrapper(void* args){}

void pulp_nn_pw_conv2d_wrapper(void* args){}

void pulp_nn_hoparallel_conv2d_wrapper(void* args){}

void pulp_nn_add_wrapper(void* args){}

void pulp_nn_wrapper(MatchCtx* ctx){
    switch(ctx->pattern_name){
        case pulp_nn_dense_pattern:
            pi_team_offload_preset(pulp_nn_dense_wrapper, ctx);
            break
        case pulp_nn_dense_out_int_pattern:
            pi_team_offload_preset(pulp_nn_dense_out_int_wrapper, ctx);
            break
        case pulp_nn_dw_conv2d_less_4_pattern:
            pi_team_offload_preset(pulp_nn_dw_conv2d_less_4_wrapper, ctx);
            break
        case pulp_nn_dw_conv2d_pattern:
            pi_team_offload_preset(pulp_nn_dw_conv2d_wrapper, ctx);
            break
        case pulp_nn_pw_conv2d_pattern:
            pi_team_offload_preset(pulp_nn_pw_conv2d_wrapper, ctx);
            break
        case pulp_nn_hoparallel_conv2d_pattern:
            pi_team_offload_preset(pulp_nn_hoparallel_conv2d_wrapper, ctx);
            break
        case pulp_nn_add_pattern:
            pi_team_offload_preset(pulp_nn_add_wrapper, ctx);
            break
        default:
            break
    }
}