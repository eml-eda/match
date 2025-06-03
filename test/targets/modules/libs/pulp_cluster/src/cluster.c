#include <pulp_cluster/cluster.h>

static void* im2col_pt_ = NULL;
static void* pwt_pt_ = NULL;
static DmaTransfer dma_transfer_;
#ifndef GAP_SDK
static void* pulp_open_l1_pt_ = NULL;
#endif

void offload_to_pulp_cluster(MatchCtx* ctx, void (inner_function)(unsigned int* args_inner_function),
                                unsigned int* args){
    #ifndef GAP_SDK
    pulp_open_l1_pt_ = pmsis_l1_malloc(L1_SCRATCHPAD_SIZE);
    #endif
    pi_cluster_task(&cluster_task,inner_function,args);
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    #ifndef GAP_SDK
    pmsis_l1_malloc_free( pulp_open_l1_pt_, L1_SCRATCHPAD_SIZE);
    #endif
}

void cluster_lib_cleanup_dma_transfers(){
    dma_transfer_free(dma_transfer_);
}

void cluster_lib_init_dma_transfers(){
    dma_transfer_ = dma_transfer_create();
}

void cluster_lib_init(MatchCtx* ctx){
    #ifdef GAP_SDK
    pi_team_config_offload(NUM_CORES);
    #endif
    cluster_lib_init_dma_transfers();
}

void* init_l1_scratchpad_memory(MatchCtx* ctx){
    #ifdef GAP_SDK
    return pi_cl_l1_malloc(NULL, L1_SCRATCHPAD_SIZE);
    #else
    return pulp_open_l1_pt_;
    #endif
}

void cluster_lib_cleanup(MatchCtx* ctx){
    cluster_lib_cleanup_dma_transfers();
}

void free_l1_scrachpad_memory(MatchCtx* ctx, void* l1_memory_pt){
    #ifdef GAP_SDK
    pi_cl_l1_free(NULL, l1_memory_pt, L1_SCRATCHPAD_SIZE);
    #endif
}

void* cluster_alloc_buffer(const char* name, int tensor_l1_pt, int size, int mem, int buffer_idx){
    /* more general approach, but useless since we know its always just im2col
    if(match_strcmp("im2col", name) == 0)
        im2col_pt_ = tensor_l1_pt;
    else if(match_strcmp("pwt", name) == 0)
        pwt_pt_ = tensor_l1_pt;
    */
    im2col_pt_ = tensor_l1_pt;
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
    if(tensor->num_dims>5)
        exit(1);
    
    if(!tensor->num_dims) return;

    #ifdef CLUSTER_LIB_DEBUG
    printf("Handle transfer params tensor l2 pt %d tensor l1 pt %d transfer type %d tensor type %d ext mem %d int mem %d\n",
        tensor_l2_pt, tensor_l1_pt, match_transfer_type, match_tensor_type, ext_mem, int_mem);
    for(int idx=0; idx<tensor->num_dims; idx++) printf(" [L2: %d L1: %d]", tensor->tiles[L2_SHARED_MEM*tensor->num_dims+idx].size, tensor->tiles[L1_SCRATCHPAD*tensor->num_dims+idx].size);
    printf("\n");
    #endif
    
    switch(tensor->num_dims){
        case 1:
            #ifdef CLUSTER_LIB_DEBUG
            printf("1D transfer prec %d bytes\n", tensor->bits/8);
            #endif
            dma_transfer_1d_async((DmaTransferConf) {
                .ext = tensor_l2_pt,
                .loc = tensor_l1_pt,
                .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*1+0].size*tensor->bits/8,
                .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
            });
            break;
        case 2:
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
            break;
        case 3:
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
            break;
        case 4:
            #ifdef CLUSTER_LIB_DEBUG
            printf("4D transfer HWC_TO_CHW %d 1D %d 2D %d prec %d bytes\n", 
                ctx->pattern_name==depthwise_conv2d && match_tensor_type==MATCH_VAR_TENSOR
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
            if(ctx->pattern_name==depthwise_conv2d && match_tensor_type==MATCH_VAR_TENSOR
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
            break;
        case 5:
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
            break;
                
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
    #ifdef GAP_SDK
    pi_team_offload_wait();
    #endif
}

void pulp_nn_dense_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    pulp_nn_linear(
        // activations pt  
        tensors[0].pt, // acts pt
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        tensors[0].tiles[L1_SCRATCHPAD*2+1].size, // input channels
        tensors[num_tensors-1].tiles[L1_SCRATCHPAD*2+1].size, // output channels
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_dense_out_int_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    pulp_nn_linear_out_32(
        // activations pt  
        tensors[0].pt, // acts pt
        // bias pt
        tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        tensors[0].tiles[L1_SCRATCHPAD*2+1].size, // input channels
        tensors[num_tensors-1].tiles[L1_SCRATCHPAD*2+1].size // output channels
    );
}

void pulp_nn_dw_conv2d_less_4_wrapper(void* args){
    // TODO: implement, currently not used
    return;
}

void pulp_nn_dw_conv2d_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    #ifdef CLUSTER_LIB_DEBUG
    if(pi_core_id()==0)
        printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_depthwise_generic(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        pwt_pt_, // pwt buffer pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_width, // input width
        inp_height, // input height
        inp_ch, // input channels
        out_width, // out width
        out_height, // out height
        out_ch, // out ch
        conv_attrs->kernel_size[1], // filter width
        conv_attrs->kernel_size[0], // filter height
        pad_top, // pad top
        pad_bottom, // pad bottom
        pad_left, // pad left
        pad_right, // pad right
        conv_attrs->strides[1], // stride width
        conv_attrs->strides[0], // stride height
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_pw_conv2d_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    #ifdef CLUSTER_LIB_DEBUG
    if(pi_core_id()==0)
        printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_pointwise_HoWo_parallel(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_width, // input width
        inp_height, // input height
        inp_ch, // input channels
        out_width, // out width
        out_height, // out height
        out_ch, // out ch
        conv_attrs->kernel_size[1], // filter width
        conv_attrs->kernel_size[0], // filter height
        pad_top, // pad top
        pad_bottom, // pad bottom
        pad_left, // pad left
        pad_right, // pad right
        conv_attrs->strides[1], // stride width
        conv_attrs->strides[0], // stride height
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_hoparallel_conv2d_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    #ifdef CLUSTER_LIB_DEBUG
    if(pi_core_id()==0)
        printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_conv_Ho_parallel(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_width, // input width
        inp_height, // input height
        inp_ch, // input channels
        out_width, // out width
        out_height, // out height
        out_ch, // out ch
        conv_attrs->kernel_size[1], // filter width
        conv_attrs->kernel_size[0], // filter height
        pad_top, // pad top
        pad_bottom, // pad bottom
        pad_left, // pad left
        pad_right, // pad right
        conv_attrs->strides[1], // stride width
        conv_attrs->strides[0], // stride height
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_conv3d_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv3DAttrs* conv_attrs = (MatchConv3DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*5+3].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*5+2].size; // out height
    int out_depth = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*5+1].size; // out depth
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*5+4].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*5+3].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*5+2].size; // out height
    int inp_depth = tensors[0].tiles[L1_SCRATCHPAD*5+1].size; // out depth
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*5+4].size; // out ch
    // pad
    int pad_front = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+1]));
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+2]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+3]));
    int pad_back = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+1]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+2]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+3]));
    #ifdef CLUSTER_LIB_DEBUG
    if(pi_core_id()==0)
        printf("Out tile [%d %d %d %d] Inp tile [%d %d %d %d] pad \\ %d / %d ^ %d v %d < %d > %d\n",
            out_ch, out_depth, out_height, out_width,
            inp_ch, inp_depth, inp_height, inp_width,
            pad_front, pad_back,
            pad_top, pad_bottom, pad_left, pad_right
        );
    #endif
    #ifdef CLUSTER_LIB_USE_NAIVE_CONV3D
    pulp_nn_conv3d_naive
    #else
    pulp_nn_conv3d_Co_parallel
    #endif
    (
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_width, // input width
        inp_height, // input height
        inp_depth, // input depth
        inp_ch, // input channels
        out_width, // out width
        out_height, // out height
        out_depth, // out depth
        out_ch, // out ch
        conv_attrs->kernel_size[2], // filter width
        conv_attrs->kernel_size[1], // filter height
        conv_attrs->kernel_size[0], // filter depth
        pad_top, // pad top
        pad_bottom, // pad bottom
        pad_left, // pad left
        pad_right, // pad right
        pad_front, // pad front
        pad_back, // pad back
        conv_attrs->strides[2], // stride width
        conv_attrs->strides[1], // stride height
        conv_attrs->strides[0], // stride depth
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_add_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    pulp_nn_add(
        // activations pt  
        tensors[0].pt, // acts 1 pt
        tensors[1].pt, // acts 2 pt
        tensors[num_tensors-1].pt, // out
        1, // out mult 1
        1, // out mult 2
        right_shift,
        // sizes of tile
        out_width, out_height, out_ch
    );
}

void pulp_nn_wrapper(MatchCtx* ctx){
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
        default:
            break;
    }
}