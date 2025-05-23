#ifdef __pulp_cluster__

#include "carfield_lib/cluster.h"
#include "carfield_lib/printf.h"
#include "carfield_lib/mbox.h"

//#define CLUSTER_LIB_DEBUG
#define CALLOC_L1_SCRATCHPAD 0

volatile dma_transfer_id_t dma_transfer_ = 0;
volatile void* im2col_pt_ = NULL;
volatile void* pwt_pt_ = NULL;


int cluster_check_should_run() 
{
    return rt_core_id() < get_core_num();
}

int cluster_check_main_core(MatchCtx* ctx) 
{
    return rt_core_id() == 0;
}

void cluster_sync_cores(MatchCtx* ctx) 
{
    asm volatile("fence rw,rw":::"memory");
    synch_barrier();
}

void cluster_lib_init(MatchCtx* ctx)
{
    #ifdef CLUSTER_LIB_DEBUG
    // Just for not overlapping print :)
    for (int i = 0; i < 10000; i++)
        asm volatile("nop");
    #endif
    dma_transfer_ = dma_transfer_create();
    #ifdef CLUSTER_LIB_DEBUG
    mini_printf("[CLUSTER] Yo! Cluster is alive! DMA counter is %d\r\n", dma_transfer_);
    #endif
}

void* init_l1_scratchpad_memory(MatchCtx* ctx){
    #ifdef CLUSTER_LIB_DEBUG
    mini_printf("[CLUSTER] Inizialing L1 Scratchpad...\r\n");
    #endif
    void* l1_memory_pt = pi_l1_malloc(0, L1_SCRATCHPAD_SIZE);
    #if CALLOC_L1_SCRATCHPAD
    for (int i = 0; i < L1_SCRATCHPAD_SIZE; i++)
        ((volatile char*)l1_memory_pt)[i] = 0;
    #endif
    #ifdef CLUSTER_LIB_DEBUG
    mini_printf("[CLUSTER] Success.\r\n");
    #endif
    return l1_memory_pt;
}

void free_l1_scrachpad_memory(MatchCtx* ctx, void* l1_memory_pt) {
    pi_l1_free(0, l1_memory_pt, L1_SCRATCHPAD_SIZE);
}


void cluster_lib_cleanup(MatchCtx* ctx) 
{
    dma_transfer_free(dma_transfer_);
}


void cluster_alloc_buffer(const char* name, int tensor_l1_pt, int size, int mem, int buffer_idx)
{
    im2col_pt_ = (void*)tensor_l1_pt;
}

int handle_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int match_tensor_type,
    int ext_mem, int int_mem 
){
    asm volatile("fence rw,rw":::"memory");
    // shouldnt happen, we currently support only L2 and L1
    if(ext_mem!=L2_SHARED_MEM || int_mem!=L1_SCRATCHPAD)
        exit(1);
    // we should handle only 4-dims tensors
    if(tensor->num_dims>5)
        exit(1);

    if(!tensor->num_dims) return 0;

    int transferred_bytes = 0;

    #ifdef CLUSTER_LIB_DEBUG
    mini_printf("[CLUSTER] DMA Transfer: %s(%p) %s %s(%p) - Tensor type: %s\r\n",
        ext_mem==L2_SHARED_MEM?"L2":"L1", tensor_l2_pt,
        match_transfer_type==MATCH_SW_LOAD_TENSOR?"→":"<-",
        int_mem==L1_SCRATCHPAD?"L1":"L2", tensor_l1_pt,
        match_tensor_type==MATCH_VAR_TENSOR?"VAR":(match_tensor_type==MATCH_CONST_TENSOR?"CONST":"OUT"));
    for(int idx=0; idx<tensor->num_dims; idx++) 
        mini_printf("  [L2: %d L1: %d]", tensor->tiles[L2_SHARED_MEM*tensor->num_dims+idx].size, tensor->tiles[L1_SCRATCHPAD*tensor->num_dims+idx].size);
    mini_printf("\r\n");
    #endif

    switch(tensor->num_dims){
        case 1: {
            int bytes = tensor->tiles[L1_SCRATCHPAD*1+0].size * tensor->bits/8;
            #ifdef CLUSTER_LIB_DEBUG
            mini_printf("[TRANSFER TYPE] 1D transfer: %d bytes\r\n", bytes);
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
            int is_1d = tensor->tiles[L2_SHARED_MEM*2+1].size==tensor->tiles[L1_SCRATCHPAD*2+1].size;
            int bytes = 0;
            #ifdef CLUSTER_LIB_DEBUG
            mini_printf("[TRANSFER TYPE] 2D transfer: 1D case: %d, bytes per elem: %d\r\n", is_1d, tensor->bits/8);
            #endif
            if(is_1d){
                bytes = tensor->tiles[L1_SCRATCHPAD*2+0].size * tensor->tiles[L1_SCRATCHPAD*2+1].size * tensor->bits/8;
                dma_transfer_1d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .length_1d_copy = bytes,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else {
                bytes = tensor->tiles[L1_SCRATCHPAD*2+0].size * tensor->tiles[L1_SCRATCHPAD*2+1].size * tensor->bits/8;
                dma_transfer_2d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*2+0].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*2+1].size*tensor->bits/8,
                    .stride_1d = tensor->tiles[L2_SHARED_MEM*2+1].size*tensor->bits/8,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            }
            transferred_bytes = bytes;
            break;
        }
        case 3: {
            int is_1d = tensor->tiles[L2_SHARED_MEM*3+1].size==tensor->tiles[L1_SCRATCHPAD*3+1].size
                        && tensor->tiles[L2_SHARED_MEM*3+2].size==tensor->tiles[L1_SCRATCHPAD*3+2].size;
            int is_2d = tensor->tiles[L2_SHARED_MEM*3+2].size==tensor->tiles[L1_SCRATCHPAD*3+2].size;
            int bytes = 0;
            #ifdef CLUSTER_LIB_DEBUG
            mini_printf("[TRANSFER TYPE] 3D transfer: 1D case: %d, 2D case: %d, bytes per elem: %d\r\n", is_1d, is_2d, tensor->bits/8);
            #endif
            if(is_1d){
                bytes = tensor->tiles[L1_SCRATCHPAD*3+0].size*
                        tensor->tiles[L1_SCRATCHPAD*3+1].size*
                        tensor->tiles[L1_SCRATCHPAD*3+2].size*
                        tensor->bits/8;
                dma_transfer_1d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .length_1d_copy = bytes,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else if(is_2d){
                bytes = tensor->tiles[L1_SCRATCHPAD*3+0].size*
                        tensor->tiles[L1_SCRATCHPAD*3+1].size*
                        tensor->tiles[L1_SCRATCHPAD*3+2].size*
                        tensor->bits/8;
                dma_transfer_2d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*3+0].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*3+1].size*tensor->tiles[L1_SCRATCHPAD*3+2].size*tensor->bits/8,
                    .stride_1d = tensor->tiles[L2_SHARED_MEM*3+1].size*tensor->tiles[L2_SHARED_MEM*3+2].size*tensor->bits/8,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else {
                bytes = tensor->tiles[L1_SCRATCHPAD*3+0].size*
                        tensor->tiles[L1_SCRATCHPAD*3+1].size*
                        tensor->tiles[L1_SCRATCHPAD*3+2].size*
                        tensor->bits/8;
                dma_transfer_3d_async((dma_transfer_cfg_t) {
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
            transferred_bytes = bytes;
            break;
        }
        case 4: {
            int is_hwc_to_chw = (ctx->pattern_name==depthwise_conv2d && match_tensor_type==MATCH_VAR_TENSOR && ctx->exec_module==PULP_CLUSTER);
            int is_1d = tensor->tiles[L2_SHARED_MEM*4+1].size==tensor->tiles[L1_SCRATCHPAD*4+1].size
                        && tensor->tiles[L2_SHARED_MEM*4+2].size==tensor->tiles[L1_SCRATCHPAD*4+2].size
                        && tensor->tiles[L2_SHARED_MEM*4+3].size==tensor->tiles[L1_SCRATCHPAD*4+3].size;
            int is_2d = tensor->tiles[L2_SHARED_MEM*4+2].size==tensor->tiles[L1_SCRATCHPAD*4+2].size
                        && tensor->tiles[L2_SHARED_MEM*4+3].size==tensor->tiles[L1_SCRATCHPAD*4+3].size;
            int bytes = 0;
            #ifdef CLUSTER_LIB_DEBUG
            mini_printf("[TRANSFER TYPE] 4D transfer: HWC_TO_CHW: %d, 1D case: %d, 2D case: %d, bytes per elem: %d\r\n",
                is_hwc_to_chw, is_1d, is_2d, tensor->bits/8);
            #endif
            if(is_hwc_to_chw){
                bytes = tensor->tiles[L1_SCRATCHPAD*4+1].size*
                        tensor->tiles[L1_SCRATCHPAD*4+2].size*
                        tensor->tiles[L1_SCRATCHPAD*4+3].size;
                dma_transfer_hwc_to_chw((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .number_of_2d_copies = tensor->tiles[L1_SCRATCHPAD*4+1].size,
                    .number_of_1d_copies = tensor->tiles[L1_SCRATCHPAD*4+2].size,
                    .length_1d_copy = tensor->tiles[L1_SCRATCHPAD*4+3].size,
                    .stride_2d = tensor->tiles[L2_SHARED_MEM*4+3].size*tensor->tiles[L2_SHARED_MEM*4+2].size,
                    .stride_1d = tensor->tiles[L2_SHARED_MEM*4+3].size,
                    .dir = 1
                });
                bytes *= tensor->bits/8;
            } else if(is_1d){
                bytes = tensor->tiles[L1_SCRATCHPAD*4+0].size*
                        tensor->tiles[L1_SCRATCHPAD*4+1].size*
                        tensor->tiles[L1_SCRATCHPAD*4+2].size*
                        tensor->tiles[L1_SCRATCHPAD*4+3].size*
                        tensor->bits/8;
                dma_transfer_1d_async((dma_transfer_cfg_t) {
                    .ext = tensor_l2_pt,
                    .loc = tensor_l1_pt,
                    .length_1d_copy = bytes,
                    .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                });
            } else if(is_2d){
                bytes = tensor->tiles[L1_SCRATCHPAD*4+0].size*
                        tensor->tiles[L1_SCRATCHPAD*4+1].size*
                        tensor->tiles[L1_SCRATCHPAD*4+2].size*
                        tensor->tiles[L1_SCRATCHPAD*4+3].size*
                        tensor->bits/8;
                dma_transfer_2d_async((dma_transfer_cfg_t) {
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
            } else {
                bytes = tensor->tiles[L1_SCRATCHPAD*4+1].size*
                        tensor->tiles[L1_SCRATCHPAD*4+2].size*
                        tensor->tiles[L1_SCRATCHPAD*4+3].size*
                        tensor->bits/8;
                // this is per 3d transfer, but we are doing N of them
                for(int idx=0; idx<tensor->tiles[L1_SCRATCHPAD*4+0].size; idx++)
                    dma_transfer_3d_async((dma_transfer_cfg_t) {
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
                bytes *= tensor->tiles[L1_SCRATCHPAD*4+0].size;
            }
            transferred_bytes = bytes;
            break;
        }
        case 5: {
            int bytes = 0;
            if(tensor->tiles[L2_SHARED_MEM*5+1].dim==tensor->tiles[L1_SCRATCHPAD*5+4].dim){
                int is_1d = tensor->tiles[L2_SHARED_MEM*5+1].size==tensor->tiles[L1_SCRATCHPAD*5+1].size
                        && tensor->tiles[L2_SHARED_MEM*5+2].size==tensor->tiles[L1_SCRATCHPAD*5+2].size
                        && tensor->tiles[L2_SHARED_MEM*5+3].size==tensor->tiles[L1_SCRATCHPAD*5+3].size;
                int is_2d = tensor->tiles[L2_SHARED_MEM*5+2].size==tensor->tiles[L1_SCRATCHPAD*5+2].size
                        && tensor->tiles[L2_SHARED_MEM*5+3].size==tensor->tiles[L1_SCRATCHPAD*5+3].size;
                #ifdef CLUSTER_LIB_DEBUG
                mini_printf("[TRANSFER TYPE] 5D transfer: 1D case: %d, 2D case: %d, bytes per elem: %d\r\n", is_1d, is_2d, tensor->bits/8);
                #endif
                if(is_1d){
                    bytes = tensor->tiles[L1_SCRATCHPAD*5+0].size*
                            tensor->tiles[L1_SCRATCHPAD*5+1].size*
                            tensor->tiles[L1_SCRATCHPAD*5+2].size*
                            tensor->tiles[L1_SCRATCHPAD*5+3].size*
                            tensor->bits/8;
                    dma_transfer_1d_async((dma_transfer_cfg_t) {
                        .ext = tensor_l2_pt,
                        .loc = tensor_l1_pt,
                        .length_1d_copy = bytes,
                        .dir = match_transfer_type==MATCH_SW_LOAD_TENSOR
                    });
                } else if(is_2d){
                    bytes = tensor->tiles[L1_SCRATCHPAD*5+0].size*
                            tensor->tiles[L1_SCRATCHPAD*5+1].size*
                            tensor->tiles[L1_SCRATCHPAD*5+2].size*
                            tensor->tiles[L1_SCRATCHPAD*5+3].size*
                            tensor->bits/8;
                    dma_transfer_2d_async((dma_transfer_cfg_t) {
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
                } else {
                    bytes = tensor->tiles[L1_SCRATCHPAD*5+1].size*
                            tensor->tiles[L1_SCRATCHPAD*5+2].size*
                            tensor->tiles[L1_SCRATCHPAD*5+3].size*
                            tensor->bits/8;
                    for(int idx=0; idx<tensor->tiles[L1_SCRATCHPAD*5+0].size; idx++)
                        dma_transfer_3d_async((dma_transfer_cfg_t) {
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
                    bytes *= tensor->tiles[L1_SCRATCHPAD*5+0].size;
                }
                transferred_bytes = bytes;
            } else
                exit(1);
            break;
        }
    }
    #ifdef CLUSTER_LIB_DEBUG
    mini_printf("[TRANSFER INFO] Transferred %d bytes\r\n", transferred_bytes);
    #endif
    for (int i = 0; i < 300; i++)
        asm volatile("fence rw,rw":::"memory");
    return transferred_bytes;
}



void wait_l1_dma_transfers(MatchCtx* ctx) {
    asm volatile("fence rw,rw":::"memory");
    dma_transfer_wait(dma_transfer_);
    asm volatile("fence rw,rw":::"memory");
    dma_transfer_ = dma_transfer_create();
    asm volatile("fence rw,rw":::"memory");
}

//#define CLUSTER_LIB_DEBUG


void pulp_nn_dense_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    pulp_nn_linear(
        // activations pt  
        tensors[0].pt, // acts pt
        // bias pt
        tensors[2].pt, // bias pt
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

void pulp_nn_dense_out_int_wrapper(MatchCtx* ctx){
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

void pulp_nn_dw_conv2d_less_4_wrapper(MatchCtx* ctx){
    // TODO: implement, currently not used
    return;
}

void pulp_nn_dw_conv2d_wrapper(MatchCtx* ctx){
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
    if(rt_core_id()==0)
        mini_printf("pulp_nn_dw_conv2d_less_4_wrapper: Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\r\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_depthwise_generic(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        tensors[2].pt, // bias pt
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

void pulp_nn_pw_conv2d_wrapper(MatchCtx* ctx){
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
    if(rt_core_id()==0)
        mini_printf("pulp_nn_pointwise_HoWo_parallel: Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\r\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_pointwise_HoWo_parallel(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        tensors[2].pt, // bias pt
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

void pulp_nn_hoparallel_conv2d_wrapper(MatchCtx* ctx){
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
    if(rt_core_id()==0)
        mini_printf("pulp_nn_hoparallel_conv2d_wrapper: Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\r\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_conv_Ho_parallel(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        im2col_pt_,
        // bias pt
        tensors[2].pt, // bias pt
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

void pulp_nn_add_wrapper(MatchCtx* ctx){
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
            pulp_nn_dense_wrapper(ctx);
            break;
        case conv2d:
            pulp_nn_hoparallel_conv2d_wrapper(ctx);
            break;
        case dense_out:
            pulp_nn_dense_out_int_wrapper(ctx);
            break;
        // case pulp_nn_dw_conv2d_less_4_pattern:
        //     pi_team_offload_preset(pulp_nn_dw_conv2d_less_4_wrapper, ctx);
        //     break;
        case depthwise_conv2d:
            pulp_nn_dw_conv2d_wrapper(ctx);
            break;
        case pointwise_conv2d:
            pulp_nn_pw_conv2d_wrapper(ctx);
            break;
        case add_requant:
            pulp_nn_add_wrapper(ctx);
            break;
        default:
            break;
    }
}



void cluster_wait_for_task_poll(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id) {
    // Polling for the start signal
    while (offload_args[0] == 0xFFFFFFF0) {
        asm volatile("fence r,rw" ::: "memory");
    }
    *tensor_ptrs = offload_args+1;
    *task_id = offload_args[0];
    asm volatile("fence r,rw" ::: "memory");
}


void cluster_end_of_task_poll(uint32_t task_id) {
    // Set end signal
    asm volatile("fence rw,rw":::"memory");
    offload_args[0] = 0xFFFFFFF0;
    asm volatile("fence rw,rw":::"memory");
}


void cluster_wait_for_task_mbox(volatile uint32_t** tensor_ptrs, volatile uint32_t* task_id) {
    asm volatile("fence rw,rw" ::: "memory");
    eu_evt_maskWaitAndClr(1 << CLUSTER_MBOX_EVT);
    mailbox_read(HOST_TO_CLUSTER_MBOX, tensor_ptrs, task_id);
    mailbox_clear(HOST_TO_CLUSTER_MBOX);
    eu_evt_clr(1 << CLUSTER_MBOX_EVT);
    asm volatile("fence rw,rw" ::: "memory");
}


void cluster_end_of_task_mbox(uint32_t task_id) {
    asm volatile("fence rw,rw" ::: "memory");
    mailbox_send(CLUSTER_TO_HOST_MBOX, task_id, 0);
    asm volatile("fence rw,rw" ::: "memory");
}


void cluster_timer_start() {
    if (rt_core_id() == 0) {
        reset_timer(0);
        asm volatile("fence rw,rw" ::: "memory");
        start_timer(0);
        asm volatile("fence rw,rw" ::: "memory");
    }
}


uint32_t cluster_timer_stop() {
    if (rt_core_id() == 0) {
        asm volatile("fence rw,rw" ::: "memory");
        stop_timer(0);
        volatile uint32_t time = get_time(0);
        asm volatile("fence rw,rw" ::: "memory");
        return time;
    }
}


#endif