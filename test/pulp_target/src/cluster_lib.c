#include <pulp_cluster/cluster_lib.h>

static void* l1_memory_pt_ = NULL;
static int im2col_size_ = 0;
static int pwt_buffer_size_ = 0;
static void* im2col_pt_ = NULL;
static void* pwt_pt_ = NULL;
static DmaTransfer dma_transfer_;

void* pulp_init_ram(int size){
    mem_init();
    return ram_malloc(size);
}

void pulp_load_file(const char* filename, void* ext_pt, int size){
    load_file_to_ram(ext_pt, filename);
}

void pulp_memcpy_from_ram(void* l2_pt, void* ext_pt, int size){
    ram_read(l2_pt, ext_pt, size);
}

void pulp_memcpy_to_ram(void* l2_pt, void* ext_pt, int size){
    ram_read(ext_pt, l2_pt, size);
}

void pulp_shutdown_ram(void* ext_pt, int size){
    ram_free(ext_pt, size);
}

void offload_to_pulp_cluster(void (inner_function)(unsigned int* args_inner_function),unsigned int* args){
    pi_cluster_task(&cluster_task,inner_function,args);
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
}

void cluster_lib_init(MatchCtx* ctx){
    // alloc L1 memory
    l1_memory_pt_ = pi_cl_l1_malloc(NULL, 92700);
    im2col_size_ = 0;
    pwt_buffer_size_ = 0;
    pi_team_config_offload(NUM_CORES);
    dma_transfer_ = dma_transfer_create();
}

void* init_l1_scratchpad_memory(MatchCtx* ctx){
    return l1_memory_pt_;
}

void* cluster_alloc_buffer(const char* name, int tensor_l1_pt, int size, int mem, int buffer_idx){
    if(match_strcmp("im2col", name) == 0)
        im2col_pt_ = tensor_l1_pt;
    else if(match_strcmp("pwt", name) == 0)
        pwt_pt_ = tensor_l1_pt;
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
    
    if(!tensor->num_dims) return;

    #ifdef CLUSTER_LIB_DEBUG
    printf("Handle transfer params ctx %d tensor %d tensor l2 pt %d tensor l1 pt %d transfer type %d tensor type %d ext mem %d int mem %d\n",
        ctx, tensor, tensor_l2_pt, tensor_l1_pt, match_transfer_type, match_tensor_type, ext_mem, int_mem);
    printf("Transferring tensor from %d to %d, num dims %d tiles addr %d sizes:\n(", tensor_l2_pt, tensor_l1_pt, tensor->num_dims, tensor->tiles);
    for(int idx=0; idx<tensor->num_dims; idx++) printf(" [L2: %d L1: %d]", tensor->tiles[L2_SHARED_MEM*2+idx].size, tensor->tiles[L1_SCRATCHPAD*2+idx].size);
    printf("\n");
    #endif
    
    switch(tensor->num_dims){
        case 1:
            #ifdef CLUSTER_LIB_DEBUG
            printf("1D transfer --> moving %d bytes from %d to %d\n", tensor->tiles[L1_SCRATCHPAD*1+0].size*tensor->bits/8, tensor_l2_pt, tensor_l1_pt);
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
            printf("2D transfer 1D %d --> moving %d bytes from %d to %d\n", 
                tensor->tiles[L2_SHARED_MEM*2+1].size==tensor->tiles[L1_SCRATCHPAD*2+1].size, 
                tensor->tiles[L1_SCRATCHPAD*2+0].size*tensor->tiles[L1_SCRATCHPAD*2+1].size*tensor->bits/8,
                tensor_l2_pt, tensor_l1_pt);
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
                    .stride_1d = tensor->tiles[L2_SHARED_MEM*3+1].size*tensor->tiles[L1_SCRATCHPAD*3+2].size*tensor->bits/8,
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
            // check if depthwise conv2d and activations
            if(ctx->pattern_name==depthwise_conv2d && match_tensor_type==MATCH_VAR_TENSOR)
                dma_transfer_hwc_to_chw((DmaTransferConf) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_2d_copies = tensor->tiles[L1_SCRATCHPAD*4+1].size,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*4+2].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*4+3].size,
                    .stride_2d = tensor->tiles[L2_SHARED_MEM*4+3].size*tensor->tiles[L1_SCRATCHPAD*4+2].size,
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
                        .ext = tensor_l2_pt + tensor->tiles[L2_SHARED_MEM*4+1].size*
                                    tensor->tiles[L2_SHARED_MEM*4+2].size*
                                    tensor->tiles[L2_SHARED_MEM*4+3].size*
                                    tensor->bits/8,
                        .loc = tensor_l1_pt + tensor->tiles[L1_SCRATCHPAD*4+1].size*
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
    }
}

void wait_l1_dma_transfers(MatchCtx* ctx){
    dma_transfer_wait(dma_transfer_);
    dma_transfer_ = dma_transfer_create();
}

void free_l1_scrachpad_memory(MatchCtx* ctx, void* l1_memory_pt){
    dma_transfer_free(dma_transfer_);
    pi_cl_l1_free(NULL, l1_memory_pt_, 92700);
}

void wait_pulp_nn_computation(MatchCtx* ctx){
    pi_team_offload_wait();
}

void pulp_nn_dense_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[2].attrs)->right_shift;
    pulp_nn_linear(
        // activations pt  
        tensors[0].pts[L1_SCRATCHPAD], // acts pt
        // bias pt
        tensors[2].pts[L1_SCRATCHPAD], // bias pt
        // output pt
        tensors[num_tensors-1].pts[L1_SCRATCHPAD], // output pt
        // weights pt
        tensors[1].pts[L1_SCRATCHPAD], // weights pt
        num_tensors>4? tensors[2].pts[L1_SCRATCHPAD]:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pts[L1_SCRATCHPAD]:NULL, // bnorm add pt
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
        tensors[0].pts[L1_SCRATCHPAD], // acts pt
        // bias pt
        tensors[2].pts[L1_SCRATCHPAD], // bias pt
        // output pt
        tensors[num_tensors-1].pts[L1_SCRATCHPAD], // output pt
        // weights pt
        tensors[1].pts[L1_SCRATCHPAD], // weights pt
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
    MatchRightShiftAttrs* r_shift_attrs = (MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs;
    int right_shift = r_shift_attrs->right_shift;
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
    int pad_top = conv_attrs->padding[0]-tensors[0].tiles[L1_SCRATCHPAD*4+1].start_idx;
    pad_top = pad_top>0?pad_top:0;
    int pad_bottom = tensors[0].tiles[L1_SCRATCHPAD*4+1].dim->size-
        (tensors[0].tiles[L1_SCRATCHPAD*4+1].size+tensors[0].tiles[L1_SCRATCHPAD*4+1].start_idx);
    pad_bottom = pad_bottom<0?-pad_bottom:0;
    int pad_left = conv_attrs->padding[1]-tensors[0].tiles[L1_SCRATCHPAD*4+2].start_idx;
    pad_left = pad_left>0?pad_left:0;
    int pad_right = tensors[0].tiles[L1_SCRATCHPAD*4+2].dim->size-
        (tensors[0].tiles[L1_SCRATCHPAD*4+2].size+tensors[0].tiles[L1_SCRATCHPAD*4+2].start_idx);
    pad_right = pad_right<0?-pad_right:0;
    pulp_nn_depthwise_generic(
        // activations pt  
        tensors[0].pts[L1_SCRATCHPAD], // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        tensors[2].pts[L1_SCRATCHPAD], // bias pt
        // output pt
        tensors[num_tensors-1].pts[L1_SCRATCHPAD], // output pt
        // weights pt
        tensors[1].pts[L1_SCRATCHPAD], // weights pt
        pwt_pt_, // pwt buffer pt
        num_tensors>4? tensors[2].pts[L1_SCRATCHPAD]:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pts[L1_SCRATCHPAD]:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_height, // input width
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
    MatchRightShiftAttrs* r_shift_attrs = (MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs;
    int right_shift = r_shift_attrs->right_shift;
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
    int pad_top = conv_attrs->padding[0]-tensors[0].tiles[L1_SCRATCHPAD*4+1].start_idx;
    pad_top = pad_top>0?pad_top:0;
    int pad_bottom = tensors[0].tiles[L1_SCRATCHPAD*4+1].dim->size-
        (tensors[0].tiles[L1_SCRATCHPAD*4+1].size+tensors[0].tiles[L1_SCRATCHPAD*4+1].start_idx);
    pad_bottom = pad_bottom<0?-pad_bottom:0;
    int pad_left = conv_attrs->padding[1]-tensors[0].tiles[L1_SCRATCHPAD*4+2].start_idx;
    pad_left = pad_left>0?pad_left:0;
    int pad_right = tensors[0].tiles[L1_SCRATCHPAD*4+2].dim->size-
        (tensors[0].tiles[L1_SCRATCHPAD*4+2].size+tensors[0].tiles[L1_SCRATCHPAD*4+2].start_idx);
    pad_right = pad_right<0?-pad_right:0;
    pulp_nn_pointwise_HoWo_parallel(
        // activations pt  
        tensors[0].pts[L1_SCRATCHPAD], // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        tensors[2].pts[L1_SCRATCHPAD], // bias pt
        // output pt
        tensors[num_tensors-1].pts[L1_SCRATCHPAD], // output pt
        // weights pt
        tensors[1].pts[L1_SCRATCHPAD], // weights pt
        num_tensors>4? tensors[2].pts[L1_SCRATCHPAD]:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pts[L1_SCRATCHPAD]:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_height, // input width
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
    MatchRightShiftAttrs* r_shift_attrs = (MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs;
    int right_shift = r_shift_attrs->right_shift;
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
    int pad_top = conv_attrs->padding[0]-tensors[0].tiles[L1_SCRATCHPAD*4+1].start_idx;
    pad_top = pad_top>0?pad_top:0;
    int pad_bottom = tensors[0].tiles[L1_SCRATCHPAD*4+1].dim->size-
        (tensors[0].tiles[L1_SCRATCHPAD*4+1].size+tensors[0].tiles[L1_SCRATCHPAD*4+1].start_idx);
    pad_bottom = pad_bottom<0?-pad_bottom:0;
    int pad_left = conv_attrs->padding[1]-tensors[0].tiles[L1_SCRATCHPAD*4+2].start_idx;
    pad_left = pad_left>0?pad_left:0;
    int pad_right = tensors[0].tiles[L1_SCRATCHPAD*4+2].dim->size-
        (tensors[0].tiles[L1_SCRATCHPAD*4+2].size+tensors[0].tiles[L1_SCRATCHPAD*4+2].start_idx);
    pad_right = pad_right<0?-pad_right:0;
    pulp_nn_conv_Ho_parallel(
        // activations pt  
        tensors[0].pts[L1_SCRATCHPAD], // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        tensors[2].pts[L1_SCRATCHPAD], // bias pt
        // output pt
        tensors[num_tensors-1].pts[L1_SCRATCHPAD], // output pt
        // weights pt
        tensors[1].pts[L1_SCRATCHPAD], // weights pt
        num_tensors>4? tensors[2].pts[L1_SCRATCHPAD]:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pts[L1_SCRATCHPAD]:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_height, // input width
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

void pulp_nn_add_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[2].attrs)->right_shift;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    pulp_nn_add(
        // activations pt  
        tensors[0].pts[L1_SCRATCHPAD], // acts 1 pt
        tensors[1].pts[L1_SCRATCHPAD], // acts 2 pt
        tensors[num_tensors-1].pts[L1_SCRATCHPAD], // out
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
            pi_team_offload_preset(pulp_nn_dense_wrapper, ctx);
            break;
        case conv2d:
            pi_team_offload_preset(pulp_nn_hoparallel_conv2d_wrapper, ctx);
            break;
        case dense_out:
            pi_team_offload_preset(pulp_nn_dense_out_int_wrapper, ctx);
            break;
        // case pulp_nn_dw_conv2d_less_4_pattern:
        //     pi_team_offload_preset(pulp_nn_dw_conv2d_less_4_wrapper, ctx);
        //     break;
        case depthwise_conv2d:
            pi_team_offload_preset(pulp_nn_dw_conv2d_wrapper, ctx);
            break;
        case pointwise_conv2d:
            pi_team_offload_preset(pulp_nn_pw_conv2d_wrapper, ctx);
            break;
        case add_requant:
            pi_team_offload_preset(pulp_nn_add_wrapper, ctx);
            break;
        default:
            break;
    }
}