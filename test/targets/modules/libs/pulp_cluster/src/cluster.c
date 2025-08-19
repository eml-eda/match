#include <pulp_cluster/cluster.h>

static DmaTransfer dma_transfer_;

void offload_to_pulp_cluster(MatchCtx* ctx, void (inner_function)(unsigned int* args_inner_function),
                                unsigned int* args){
    #ifndef GAP_SDK
    set_pulp_open_l1_pt(pmsis_l1_malloc(L1_SCRATCHPAD_SIZE));
    #endif
    pi_cluster_task(&cluster_task,inner_function,args);
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    #ifndef GAP_SDK
    pmsis_l1_malloc_free( get_pulp_open_l1_pt(), L1_SCRATCHPAD_SIZE);
    #endif
}

void cluster_lib_cleanup_dma_transfers(){
    dma_transfer_free(dma_transfer_);
}

void cluster_lib_init_dma_transfers(){
    dma_transfer_ = dma_transfer_create();
}

void cluster_lib_init(MatchCtx* ctx){
//    #ifdef GAP_SDK
//    pi_team_config_offload(NUM_CORES);
//    #endif
    cluster_lib_init_dma_transfers();
}

void* init_l1_scratchpad_memory(MatchCtx* ctx){
    #ifdef GAP_SDK
    return pi_cl_l1_malloc(NULL, L1_SCRATCHPAD_SIZE);
    #else
    return get_pulp_open_l1_pt();
    #endif
}

void cluster_lib_cleanup(MatchCtx* ctx){
    cluster_lib_cleanup_dma_transfers();
}

void free_l1_scratchpad_memory(MatchCtx* ctx, void* l1_memory_pt){
    #ifdef GAP_SDK
    pi_cl_l1_free(NULL, l1_memory_pt, L1_SCRATCHPAD_SIZE);
    #endif
    free_pulp_device_buffers();
}

void* cluster_alloc_buffer(const char* name, int tensor_l1_pt, int size, int mem, int buffer_idx){
    // more general approach, but useless since we know its always just im2col
    if(match_strcmp("im2col", name) == 0)
        set_im2col_pt(tensor_l1_pt);
    else if(match_strcmp("pwt", name) == 0)
        set_pwt_pt(tensor_l1_pt);
    else if(match_strcmp("bt_buffer", name) == 0)
        set_bt_buffer_pt(tensor_l1_pt);
    // set_im2col_pt(tensor_l1_pt);
}

void handle_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem 
){
    // shouldnt happen, we currently support only L2 and L1
    if(ext_mem!=L2_SHARED_MEM || int_mem!=L1_SCRATCHPAD)
        exit(1);
    // we should handle only 4-dims tensors
    if(tensor->num_dims>5)
        exit(1);
    
    if(!tensor->num_dims) return;

    #ifdef CLUSTER_LIB_DEBUG
    printf("Handle transfer params tensor l2 pt 0x%x tensor l1 pt 0x%x transfer type %d tensor type %d ext mem %d int mem %d\n",
        tensor_l2_pt, tensor_l1_pt, match_transfer_type, tensor->tensor_type, ext_mem, int_mem);
    for(int idx=0; idx<tensor->num_dims; idx++) printf(" [L2: %d L1: %d]", tensor->tiles[L2_SHARED_MEM*tensor->num_dims+idx].size, tensor->tiles[L1_SCRATCHPAD*tensor->num_dims+idx].size);
    printf("\n");
    #endif
    
    switch(tensor->num_dims){
        case 1:
            handle_dma_transfer_1d(
                ctx, tensor,
                tensor_l2_pt, tensor_l1_pt,
                match_transfer_type,
                ext_mem, int_mem
            );
            break;
        case 2:
            handle_dma_transfer_2d(
                ctx, tensor,
                tensor_l2_pt, tensor_l1_pt,
                match_transfer_type,
                ext_mem, int_mem
            );
            break;
        case 3:
            handle_dma_transfer_3d(
                ctx, tensor,
                tensor_l2_pt, tensor_l1_pt,
                match_transfer_type,
                ext_mem, int_mem
            );
            break;
        case 4:
            handle_dma_transfer_4d(
                ctx, tensor,
                tensor_l2_pt, tensor_l1_pt,
                match_transfer_type,
                ext_mem, int_mem
            );
            break;
        case 5:
            handle_dma_transfer_5d(
                ctx, tensor,
                tensor_l2_pt, tensor_l1_pt,
                match_transfer_type,
                ext_mem, int_mem
            );
            break;
        default:
            printf("Unsupported tensor num dims %d\n", tensor->num_dims);
            exit(1);
    }
    #ifdef CLUSTER_LIB_DEBUG
    printf("\n");
    #endif
}

void wait_l1_dma_transfers(MatchCtx* ctx){
    dma_transfer_wait(dma_transfer_);
    cluster_lib_init_dma_transfers();
}

void wait_pulp_nn_computation(MatchCtx* ctx){
//    #ifdef GAP_SDK
//    pi_team_offload_wait();
//    #endif
}
/*
    Main Wrapper Function
*/
void pulp_nn_wrapper(MatchCtx* ctx){
    // int end_cycles = stop_match_perf_counter();
    // int start_cycles = start_match_perf_counter();
    switch(ctx->pattern_name){
        case dense:
            #ifdef GAP_SDK
            pi_team_offload_preset(
            #else
            pi_cl_team_fork(NUM_CORES,
            #endif
                pulp_nn_dense_wrapper, ctx);
            break;
        case conv3d:
            #ifdef GAP_SDK
            pi_team_offload_preset(
            #else
            pi_cl_team_fork(NUM_CORES,
            #endif
                pulp_nn_conv3d_wrapper, ctx);
            break;
        case conv2d:
            #ifdef GAP_SDK
            pi_team_offload_preset(
            #else
            pi_cl_team_fork(NUM_CORES,
            #endif
                pulp_nn_hoparallel_conv2d_wrapper, ctx);
            break;
        case dense_out:
            #ifdef GAP_SDK
            pi_team_offload_preset(
            #else
            pi_cl_team_fork(NUM_CORES,
            #endif
                pulp_nn_dense_out_int_wrapper, ctx);
            break;
        // case pulp_nn_dw_conv2d_less_4_pattern:
        //     pi_team_offload_preset(pulp_nn_dw_conv2d_less_4_wrapper, ctx);
        //     break;
        case depthwise_conv2d:
            #ifdef GAP_SDK
            pi_team_offload_preset(
            #else
            pi_cl_team_fork(NUM_CORES,
            #endif
                pulp_nn_dw_conv2d_wrapper, ctx);
            break;
        case pointwise_conv2d:
            #ifdef GAP_SDK
            pi_team_offload_preset(
            #else
            pi_cl_team_fork(NUM_CORES,
            #endif
                pulp_nn_pw_conv2d_wrapper, ctx);
            break;
        case add_requant:
            #ifdef GAP_SDK
            pi_team_offload_preset(
            #else
            pi_cl_team_fork(NUM_CORES,
            #endif
                pulp_nn_add_wrapper, ctx);
            break;

        case conv2d_train:
            pulp_train_conv2d_fp32_wrapper(ctx);    
            break;

        case conv2ddw_train:
            pulp_train_conv2ddw_fp32_wrapper(ctx);    
            break;

        case conv2d_transpose:
            // Intelligent kernel selection based on kernel size and stride
            {
                MatchConv2DTransposeAttrs* conv_attrs = (MatchConv2DTransposeAttrs*)ctx->ops->ops[0].attrs;
                int kernel_h = conv_attrs->kernel_size[0];
                int kernel_w = conv_attrs->kernel_size[1];
                int* strides = conv_attrs->strides;
                int* padding = conv_attrs->padding;
                int is_dw = conv_attrs->depthwise;
                void* im2col_pt = get_im2col_pt();
                if (kernel_h == 1 && kernel_w == 1 && strides[0] == 1 && strides[1] == 1 && !is_dw
                    && padding[0] == 0 && padding[1] == 0 && padding[2] == 0 && padding[3] == 0)
                    pi_cl_team_fork(NUM_CORES, odl_naive_parallel_conv2d_transpose_pw_stride_1_fp32, ctx);
                else if (strides[0] == 2 && strides[1] == 2)
                    pi_cl_team_fork(NUM_CORES, odl_naive_parallel_conv2d_transpose_stride_2_fp32, ctx);
                else
                    pi_cl_team_fork(NUM_CORES, odl_naive_parallel_conv2d_transpose_fp32, ctx);
            }
            break;
        
        case conv2d_grad_params:
            {
                MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
                int is_dw = conv_attrs->depthwise;
                int is_pw = conv_attrs->kernel_size[0] == 1 && conv_attrs->kernel_size[1] == 1;
                void* im2col_pt = get_im2col_pt();
                if(is_dw)
                    pi_cl_team_fork(NUM_CORES, odl_optimized_parallel_conv2d_bw_dw_fp32, ctx);
                else if(!is_pw && im2col_pt)
                    pi_cl_team_fork(NUM_CORES, odl_fast_conv2d_bw_fp32_im2col, ctx);
                else if(is_pw || im2col_pt)
                    pulp_train_conv2d_bw_fp32_wrapper(ctx);
                else
                    pi_cl_team_fork(NUM_CORES, odl_fast_parallel_conv2d_bw_fp32, ctx);
            }
            break;
        
        case bw_instance_norm_tail:
            pi_cl_team_fork(NUM_CORES, odl_bw_instance_norm_tail_fp32, ctx);
            break;

        case fw_instance_norm_tail:
            pi_cl_team_fork(NUM_CORES, odl_fw_instance_norm_tail_fp32, ctx);
            break;

        default:
            break;
    }
    // end_cycles = stop_match_perf_counter();
    // printf("PULP NN Wrapper executed in %d cycles\n", end_cycles - start_cycles);
    // start_cycles = start_match_perf_counter();
}