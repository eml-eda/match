#ifdef __pulp_cluster__

#include "carfield_lib/cluster.h"

#include <stdint.h>
#include <stddef.h>

#include "carfield_lib/printf.h"
#include "carfield_lib/utils.h"

#include "match/ctx.h"

#include "pulp.h"

#include "pulp_nn/pulp_nn_kernels.h"
#include "pulp_kernels/pulp_fp16_kernels.h"
#include "redmule/redmule_kernels.h"


static void wtf_wrapper(MatchCtx *ctx) {
    smp_printf("[PULP][KER] No kernel found for pattern ID %d. Crashing... :(\r\n", ctx->pattern_name);
    exit(1);
}


// Feed MatchCtx to library kernel APIs
void kernel_wrapper(MatchCtx* ctx) 
{    
    switch(ctx->pattern_name){

    // Pulp NN int8
    #ifdef pulpd_dense
        case pulpd_dense: pulp_nn_dense_wrapper(ctx); break;
    #endif
    #ifdef pulpd_conv2d
        case pulpd_conv2d: pulp_nn_hoparallel_conv2d_wrapper(ctx); break;
    #endif
    #ifdef pulpd_dense_out
        case pulpd_dense_out: pulp_nn_dense_out_int_wrapper(ctx); break;
    #endif
    #ifdef pulpd_depthwise_conv2d
        case pulpd_depthwise_conv2d: pulp_nn_dw_conv2d_wrapper(ctx); break;
    #endif
    #ifdef pulpd_pointwise_conv2d
        case pulpd_pointwise_conv2d: pulp_nn_pw_conv2d_wrapper(ctx); break;
    #endif
    #ifdef pulpd_add_requant
        case pulpd_add_requant: pulp_nn_add_wrapper(ctx); break;
    #endif

    // Pulp NN fp16

    #ifdef pulpd_conv2d_fp16
        case pulpd_conv2d_fp16: pulp_fp16_conv2d_wrapper(ctx); break;
    #endif
    #ifdef pulpd_conv2d_bias_fp16
        case pulpd_conv2d_bias_fp16: pulp_fp16_conv2d_wrapper(ctx); break;
    #endif
    #ifdef pulpd_conv2d_bnorm_fp16
        case pulpd_conv2d_bnorm_fp16: pulp_fp16_conv2d_wrapper(ctx); break;
    #endif

    #ifdef pulpd_conv2d_grouped_fp16
        case pulpd_conv2d_grouped_fp16: pulp_fp16_conv2d_grouped_wrapper(ctx); break;
    #endif
    #ifdef pulpd_conv2d_grouped_bias_fp16
        case pulpd_conv2d_grouped_bias_fp16: pulp_fp16_conv2d_grouped_wrapper(ctx); break;
    #endif
    #ifdef pulpd_conv2d_grouped_bnorm_fp16
        case pulpd_conv2d_grouped_bnorm_fp16: pulp_fp16_conv2d_grouped_wrapper(ctx); break;
    #endif

    #ifdef pulpd_dense_fp16
        case pulpd_dense_fp16: pulp_fp16_dense_wrapper(ctx); break;
    #endif
    #ifdef pulpd_dense_bias_fp16
        case pulpd_dense_bias_fp16: pulp_fp16_dense_wrapper(ctx); break;
    #endif

    #ifdef pulpd_batch_matmul_fp16
        case pulpd_batch_matmul_fp16: pulp_fp16_batch_matmul_wrapper(ctx); break;
    #endif

    #ifdef pulpd_avgpool2d_fp16
        case pulpd_avgpool2d_fp16: pulp_fp16_avgpool2d_wrapper(ctx); break;
    #endif
    #ifdef pulpd_maxpool2d_fp16
        case pulpd_maxpool2d_fp16: pulp_fp16_maxpool2d_wrapper(ctx); break;
    #endif

    #ifdef pulpd_redmule_conv3d_fp16
        case pulpd_redmule_conv3d_fp16: redmule_fp16_conv3d_wrapper(ctx); break;
    #endif
    
    #ifdef conv2d_transpose
        case conv2d_transpose:
            odl_naive_parallel_conv2d_transpose_fp16(ctx);
            break;
    #endif

    #ifdef pulpd_conv2d_bias_relu_fp16
        case pulpd_conv2d_bias_relu_fp16: pulp_fp16_conv2d_wrapper(ctx); break;
    #endif

        default: wtf_wrapper(ctx); break;
    }
}


/* ======== PULP-NN int8 ======== */


void pulp_nn_dense_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*2+1].size;
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*2+1].size;

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] 'pulp_nn_linear': Out. tile (%d,) | Inp. tile (%d,) | Requant Shift: %d\r\n", out_ch, inp_ch, right_shift);
#endif
    
    pulp_nn_linear(
        tensors[0].pt,                // Activations Ptr
        num_tensors > 4 ? NULL : tensors[2].pt, // Bias Ptr
        tensors[num_tensors-1].pt,    // Output Ptr
        tensors[1].pt,                // Weights Ptr
        num_tensors > 4 ? tensors[2].pt : NULL, // Batch Norm. Mul Tensor Ptr
        num_tensors > 4 ? tensors[3].pt : NULL, // Batch Norm. Add Tensor Ptr
        1,                            // Requant mult factor
        right_shift,                  // Requant shift factor
        inp_ch,                       // Num. Input Channels
        out_ch,                       // Num. Output Channels
        1,                            // Apply ReLU
        num_tensors > 4               // Use Batch Norm or Bias -> using bnorm on this pattern
    );
}


void pulp_nn_dense_out_int_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*2+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*2+1].size;
    
#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] 'pulp_nn_linear_out_32': Out. tile (%d,) | Inp. tile (%d,)\r\n", out_ch, inp_ch);
#endif

    pulp_nn_linear_out_32(
        tensors[0].pt,            // Activations Ptr
        tensors[2].pt,            // Bias Ptr
        tensors[num_tensors-1].pt,// Output Ptr
        tensors[1].pt,            // Weights Ptr
        inp_ch,                   // Num. Input Channels
        out_ch                    // Num. Output Channels
    );
}


void pulp_nn_dw_conv2d_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // Ouput
    int out_width = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+3].size; // out ch
    // Input
    int inp_width = tensors[0].tiles[MEM_L1_PULPD*4+2].size; // out width
    int inp_height = tensors[0].tiles[MEM_L1_PULPD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*4+3].size; // out ch
    // Padding
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] 'pulp_nn_depthwise_generic': ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Inp. tile (%d,%d,%d) | ", inp_ch, inp_height, inp_width);
    smp_printf("Pad ▲ %d ▼ %d ◄ %d ► %d\r\n", pad_top, pad_bottom, pad_left, pad_right);
#endif

    pulp_nn_depthwise_generic(
        tensors[0].pt,               // Activations ptr
        im2col_pt_,                  // im2col buffer ptr
        num_tensors > 4 ? NULL : tensors[2].pt, // Bias ptr
        tensors[num_tensors-1].pt,   // Output ptr
        tensors[1].pt,               // Weights ptr
        pwt_pt_,                     // pwt buffer ptr
        num_tensors > 4 ? tensors[2].pt : NULL, // Batch Norm. Mul Tensor Ptr
        num_tensors > 4 ? tensors[3].pt : NULL, // Batch Norm. Add Tensor Ptr
        1,                           // Requant mult factor
        right_shift,                 // Requant shift factor
        inp_width,                   // Input width
        inp_height,                  // Input height
        inp_ch,                      // Input channels
        out_width,                   // Output width
        out_height,                  // Output height
        out_ch,                      // Output channels
        conv_attrs->kernel_size[1],  // Filter width
        conv_attrs->kernel_size[0],  // Filter height
        pad_top,                     // Padding top
        pad_bottom,                  // Padding bottom
        pad_left,                    // Padding left
        pad_right,                   // Padding right
        conv_attrs->strides[1],      // Stride width
        conv_attrs->strides[0],      // Stride height
        1,                           // Apply ReLU
        num_tensors > 4              // Apply batch norm or bias -> bnorm for this pattern
    );
}

void pulp_nn_pw_conv2d_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // Output
    int out_width = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+3].size; // out ch
    // Input
    int inp_width = tensors[0].tiles[MEM_L1_PULPD*4+2].size; // out width
    int inp_height = tensors[0].tiles[MEM_L1_PULPD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*4+3].size; // out ch
    // Padding
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] pulp_nn_pointwise_HoWo_parallel: ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Inp. tile (%d,%d,%d) | ", inp_ch, inp_height, inp_width);
    smp_printf("Pad ▲ %d ▼ %d ◄ %d ► %d\r\n", pad_top, pad_bottom, pad_left, pad_right);
#endif

    pulp_nn_pointwise_HoWo_parallel(
        tensors[0].pt,               // Activations ptr
        im2col_pt_,                  // im2col buffer ptr
        num_tensors > 4 ? NULL : tensors[2].pt, // Bias ptr if present
        tensors[num_tensors-1].pt,   // Output ptr
        tensors[1].pt,               // Weights ptr
        num_tensors > 4 ? tensors[2].pt : NULL, // Bnorm Mul Tensor ptr if present
        num_tensors > 4 ? tensors[3].pt : NULL, // Bnorm Add Tensor ptr if present
        1,                           // Requant mult factor
        right_shift,                 // Requant shift factor
        inp_width,                   // Input width
        inp_height,                  // Input height
        inp_ch,                      // Num. Input channels
        out_width,                   // Output width
        out_height,                  // Output height
        out_ch,                      // Num. Output channels
        conv_attrs->kernel_size[1],  // Filter width
        conv_attrs->kernel_size[0],  // Filter height
        pad_top,                     // Padding top
        pad_bottom,                  // Padding bottom
        pad_left,                    // Padding left
        pad_right,                   // Padding right
        conv_attrs->strides[1],      // Stride width
        conv_attrs->strides[0],      // Stride height
        1,                           // Apply ReLU activation
        num_tensors > 4              // Using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_hoparallel_conv2d_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+3].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[MEM_L1_PULPD*4+2].size; // out width
    int inp_height = tensors[0].tiles[MEM_L1_PULPD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*4+3].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] pulp_nn_conv_Ho_parallel: ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Inp. tile (%d,%d,%d) | ", inp_ch, inp_height, inp_width);
    smp_printf("Pad ▲ %d ▼ %d ◄ %d ► %d\r\n", pad_top, pad_bottom, pad_left, pad_right);
#endif

    pulp_nn_conv_Ho_parallel(
        tensors[0].pt,               // Activations ptr
        im2col_pt_,                  // im2col buffer ptr
        num_tensors > 4 ? NULL : tensors[2].pt, // Bias ptr
        tensors[num_tensors-1].pt,   // Output ptr
        tensors[1].pt,               // Weights ptr
        num_tensors > 4 ? tensors[2].pt : NULL, // bnorm mul tensor ptr
        num_tensors > 4 ? tensors[3].pt : NULL, // bnorm add tensor ptr
        1,                           // Requant mult factor
        right_shift,                 // Requant shift factor
        inp_width,                   // Input width
        inp_height,                  // Input height
        inp_ch,                      // Input channels
        out_width,                   // Output width
        out_height,                  // Output height
        out_ch,                      // Output channels
        conv_attrs->kernel_size[1],  // Filter width
        conv_attrs->kernel_size[0],  // Filter height
        pad_top,                     // Padding top
        pad_bottom,                  // Padding bottom
        pad_left,                    // Padding left
        pad_right,                   // Padding right
        conv_attrs->strides[1],      // Stride width
        conv_attrs->strides[0],      // Stride height
        1,                           // Apply ReLU activation
        num_tensors > 4              // Using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_add_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    // Output
    int out_width = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+2].size;
    int out_height = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+3].size;

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] pulp_nn_add: ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Requant Shift: %d\r\n", right_shift);
#endif

    pulp_nn_add(
        tensors[0].pt,            // Input 1 Activations Tensor Pointer
        tensors[1].pt,            // Input 2 Activations Tensor Pointer
        tensors[num_tensors-1].pt,// Output Tensor Pointer
        1,                        // Input 1 Multiplier
        1,                        // Input 2 Multiplier
        right_shift,              // Requant Right Shift
        out_width,                // Tile Width
        out_height,               // Tile Height
        out_ch                    // Tile Channels
    );
}



/* ======== pulp-kernels fp16 ======== */


void pulp_fp16_dense_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int batch_size = tensors[0].tiles[MEM_L1_PULPD*tensors[0].num_dims+0].size;
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*tensors[0].num_dims+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*tensors[num_tensors-1].num_dims+1].size;

    // TODO improve this - use RedMulE when supported
    if (inp_ch == 64 && out_ch == 10 && batch_size == 1) {
        #if DEBUG_CLUSTER_LIB
            smp_printf("[PULP][KER] Found supported RedMulE input.\r\n");
        #endif
        redmule_fp16_dense_wrapper(ctx);
        return;
    }

    if (batch_size > 1) {
        #if DEBUG_CLUSTER_LIB
            smp_printf("[PULP][KER] pulp_fp16_gemm: ");
            smp_printf("Out. tile (%d, %d) | ", batch_size, out_ch);
            smp_printf("Inp. tile (%d, %d)\r\n", batch_size, inp_ch);
        #endif
        pulp_fp16_gemm(
            tensors[0].pt,                // Activations pt
            tensors[1].pt,                // Weights pt
            num_tensors > 3 ? tensors[2].pt : NULL, // Bias ptr
            tensors[num_tensors-1].pt,    // Output pt
            batch_size,                   // Batch size
            inp_ch,                       // Input Neurons
            out_ch                        // Output Neurons
        );
    } else {
        #if DEBUG_CLUSTER_LIB
            smp_printf("[PULP][KER] pulp_fp16_linear: ");
            smp_printf("Out. tile (%d, %d) | ", batch_size, out_ch);
            smp_printf("Inp. tile (%d, %d)\r\n", batch_size, inp_ch);
        #endif
        pulp_fp16_linear(
            tensors[0].pt,             // Activations pt
            tensors[1].pt,                                             // Weights pt
            tensors[num_tensors-1].pt,// Output pt
            num_tensors > 3 ? tensors[2].pt : NULL,                    // Bias ptr
            inp_ch,                                                    // Input Neurons
            out_ch                                                     // Output Neurons
        );
    }
}


void pulp_fp16_batch_matmul_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int dim_b = tensors[0].tiles[MEM_L1_PULPD*tensors[0].num_dims+0].size;
    int dim_m = tensors[0].tiles[MEM_L1_PULPD*tensors[0].num_dims+1].size;
    int dim_n = tensors[0].tiles[MEM_L1_PULPD*tensors[0].num_dims+2].size;
    int dim_k = tensors[1].tiles[MEM_L1_PULPD*tensors[1].num_dims+2].size;

    for (int b = 0; b < dim_b; b++) {
        #if DEBUG_CLUSTER_LIB
            smp_printf("[PULP][KER] batch_matmul via pulp_fp16_gemm (batch %d/%d): ", b+1, dim_b);
            smp_printf("Inp. tile (%d, %d) | ", dim_m, dim_n);
            smp_printf("Out. tile (%d, %d)\r\n", dim_n, dim_k);
        #endif
        pulp_fp16_gemm(
            (void*)((uint16_t*)tensors[0].pt + b * dim_m * dim_n),               // input a pt
            (void*)((uint16_t*)tensors[1].pt + b * dim_n * dim_k),               // input b pt
            NULL,                                                                // bias pt
            (void*)((uint16_t*)tensors[num_tensors-1].pt + b * dim_m * dim_k),   // output pt
            dim_m,                                                               // M
            dim_n,                                                               // N
            dim_k                                                                // K
        );
    }
}


void pulp_fp16_conv2d_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;

    void *input = tensors[0].pt;
    void *weight = tensors[1].pt;
    void *bias = num_tensors > 3 ? tensors[2].pt : NULL;
    void *bnorm_mul = num_tensors > 4 ? tensors[3].pt : NULL;
    void *bnorm_add = num_tensors > 4 ? tensors[4].pt : NULL;
    void *output = tensors[num_tensors-1].pt;
    void *im2col = im2col_pt_;

    int out_width = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+2].size; 
    int out_height = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+3].size;

    int inp_width = tensors[0].tiles[MEM_L1_PULPD*4+2].size; 
    int inp_height = tensors[0].tiles[MEM_L1_PULPD*4+1].size;
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*4+3].size;

    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));
    
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    int filter_width = conv_attrs->kernel_size[1];
    int filter_height = conv_attrs->kernel_size[0];
    int stride_x = conv_attrs->strides[1];
    int stride_y = conv_attrs->strides[0];

    int apply_relu = ctx->ops->ops[num_ops-1].op_code == MATCH_OP_RELU;

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] pulp_fp16_conv2d: ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Inp. tile (%d,%d,%d) | ", inp_ch, inp_height, inp_width);
    smp_printf("Pad ▲ %d ▼ %d ◄ %d ► %d | ", pad_top, pad_bottom, pad_left, pad_right);
    smp_printf("Kernel size %dx%d | stride %dx%d | Apply relu %d\r\n", filter_height, filter_width, stride_y, stride_x, apply_relu);
    smp_printf("Inp pt %p | Weight pt %p | Bias pt %p | Bnorm Mul %p | Bnorm add %p | Output %p | Im2col %p\r\n",
        input, weight, bias, bnorm_mul, bnorm_add, output, im2col
    );
#endif

    pulp_fp16_conv2d(
        input, weight, bias, bnorm_mul, bnorm_add, output, im2col,
        inp_width, inp_height, inp_ch,
        out_width,out_height, out_ch, 
        filter_width, filter_height,
        pad_top, pad_bottom, pad_left, pad_right,
        stride_x, stride_y,
        apply_relu
    );
}


void pulp_fp16_conv2d_grouped_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;

    void *input = tensors[0].pt;
    void *weight = tensors[1].pt;
    void *bias = num_tensors > 3 ? tensors[2].pt : NULL;
    void *bnorm_mul = num_tensors > 4 ? tensors[3].pt : NULL;
    void *bnorm_add = num_tensors > 4 ? tensors[4].pt : NULL;
    void *output = tensors[num_tensors-1].pt;
    void *im2col = im2col_pt_;

    int out_width = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+2].size; 
    int out_height = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_PULPD*4+3].size;

    int inp_width = tensors[0].tiles[MEM_L1_PULPD*4+2].size; 
    int inp_height = tensors[0].tiles[MEM_L1_PULPD*4+1].size;
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*4+3].size;

    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+2]));
    
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    int filter_width = conv_attrs->kernel_size[1];
    int filter_height = conv_attrs->kernel_size[0];
    int stride_x = conv_attrs->strides[1];
    int stride_y = conv_attrs->strides[0];
    int groups = conv_attrs->groups;
    int apply_relu = 0;

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] pulp_fp16_conv2d_grouped: ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Inp. tile (%d,%d,%d) | ", inp_ch, inp_height, inp_width);
    smp_printf("Pad ▲ %d ▼ %d ◄ %d ► %d | ", pad_top, pad_bottom, pad_left, pad_right);
    smp_printf("Groups %d\r\n", groups);
#endif

    pulp_fp16_conv2d_grouped(
        input, weight, bias, bnorm_mul, bnorm_add, output, im2col,
        inp_width, inp_height, inp_ch,
        out_width,out_height, out_ch, 
        filter_width, filter_height,
        pad_top, pad_bottom, pad_left, pad_right,
        stride_x, stride_y,
        apply_relu,
        groups
    );
}



void pulp_fp16_avgpool2d_wrapper(MatchCtx* ctx){
    // TODO add support in MATCH
}

void pulp_fp16_maxpool2d_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    MatchMaxPool2DAttrs* pool_attrs = (MatchMaxPool2DAttrs*)ctx->ops->ops[0].attrs;
    MatchTensor* input_tensor = &tensors[0];
    MatchTensor* output_tensor = &tensors[ctx->tensors->num_tensors - 1];

    int input_height = input_tensor->tiles[MEM_L1_PULPD * 4 + 1].size;
    int input_width = input_tensor->tiles[MEM_L1_PULPD * 4 + 2].size;
    int input_channels = input_tensor->tiles[MEM_L1_PULPD * 4 + 3].size;
    int output_height = output_tensor->tiles[MEM_L1_PULPD * 4 + 1].size;
    int output_width = output_tensor->tiles[MEM_L1_PULPD * 4 + 2].size;
    int output_channels = output_tensor->tiles[MEM_L1_PULPD * 4 + 3].size;
    int pad_top = match_get_pad_x_of_tile(&input_tensor->tiles[MEM_L1_PULPD * 4 + 1]);
    int pad_left = match_get_pad_x_of_tile(&input_tensor->tiles[MEM_L1_PULPD * 4 + 2]);
    int pad_bottom = match_get_pad_y_of_tile(&input_tensor->tiles[MEM_L1_PULPD * 4 + 1]);
    int pad_right = match_get_pad_y_of_tile(&input_tensor->tiles[MEM_L1_PULPD * 4 + 2]);

    pulp_fp16_maxpool2d(
        input_tensor->pt, output_tensor->pt,
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        pool_attrs->pool_size[1], pool_attrs->pool_size[0],
        pad_top, pad_bottom, pad_left, pad_right,
        pool_attrs->strides[1], pool_attrs->strides[0]
    );
}




/* ======== RedMulE [WIP] ======== */


void redmule_fp16_dense_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int inp_neurons = tensors[0].tiles[MEM_L1_PULPD*2+1].size;
    int out_neurons = tensors[num_tensors-1].tiles[MEM_L1_PULPD*2+1].size;

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] 'redmule_gemm_fp16': M = 1 | N = %d | K = %d\r\n", inp_neurons,  out_neurons);
#endif

    void *input = tensors[0].pt;
    void *weight = tensors[1].pt;
    void *bias = num_tensors > 3 ? tensors[2].pt : NULL;
    void *output = tensors[num_tensors-1].pt;

    if (num_tensors > 3) {
        // Copy Bias in output because RedMulE use same ptr for Y and Z
        // TODO check if using DMA is possible
        pulp_fp16_copy(bias, output, out_neurons);
    } else {
        // If no bias, fill output with zeros
        for (int i = 0; i < out_neurons; i++) {
            ((fp16*)output)[i] = 0.0f; // 0.0f
        }
    }

    cluster_sync_cores(ctx);

    redmule_fp16_gemm(
        input,          // Activations pt -> X
        weight,         // Weights pt     -> W
        output,         // Output pt      -> YZ
        1,              // M dim
        inp_neurons,    // N dim
        out_neurons     // K dim
    );
}

void redmule_fp16_conv3d_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    MatchOp* ops = ctx->ops->ops;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int d_in = tensors[0].tiles[MEM_L1_PULPD*5+1].size;
    int h_in = tensors[0].tiles[MEM_L1_PULPD*5+2].size;
    int w_in = tensors[0].tiles[MEM_L1_PULPD*5+3].size;
    int c_in = tensors[0].tiles[MEM_L1_PULPD*5+4].size;
    int d_out = tensors[num_tensors-1].tiles[MEM_L1_PULPD*5+1].size;
    int h_out = tensors[num_tensors-1].tiles[MEM_L1_PULPD*5+2].size;
    int w_out = tensors[num_tensors-1].tiles[MEM_L1_PULPD*5+3].size;
    int c_out = tensors[num_tensors-1].tiles[MEM_L1_PULPD*5+4].size;
    MatchConv3DAttrs* conv3d_attrs = (MatchConv3DAttrs*)ctx->ops->ops[0].attrs;
    int k_d = conv3d_attrs->kernel_size[0];
    int k_h = conv3d_attrs->kernel_size[1];
    int k_w = conv3d_attrs->kernel_size[2];
    int s_d = conv3d_attrs->strides[0];
    int s_h = conv3d_attrs->strides[1];
    int s_w = conv3d_attrs->strides[2];
    int pad_d_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*5+1]));
    int pad_h_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*5+2]));
    int pad_w_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*5+3]));
    int pad_d_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*5+1]));
    int pad_h_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*5+2]));
    int pad_w_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*5+3]));
    int apply_relu = ops[num_ops-1].op_code == MATCH_OP_RELU;
    void *input = tensors[0].pt;
    void *weights = tensors[1].pt;
    void *bias = num_tensors > 3 ? tensors[2].pt : NULL;
    void *output_matrix = tensors[num_tensors-1].pt;
    
    #if DEBUG_CLUSTER_LIB
    smp_printf(
        "[PULP][KER] 'redmule_fp16_conv3d': input=%p weights=%p bias=%p output=%p\r\n",
        input, weights, bias, output_matrix
    );
    smp_printf(
        "[PULP][KER] 'redmule_fp16_conv3d': c_in=%d d_in=%d h_in=%d w_in=%d | c_out=%d d_out=%d h_out=%d w_out=%d\r\n",
        c_in, d_in, h_in, w_in, c_out, d_out, h_out, w_out
    );
    smp_printf(
        "[PULP][KER] 'redmule_fp16_conv3d': kernel=(%d,%d,%d) strides=(%d,%d,%d) padding_d=(%d,%d) padding_h=(%d,%d) padding_w=(%d,%d) relu=%d\r\n",
        k_d, k_h, k_w,
        s_d, s_h, s_w,
        pad_d_top, pad_d_bottom,
        pad_h_top, pad_h_bottom,
        pad_w_left, pad_w_right,
        apply_relu
    );
    smp_printf("[PULP][KER] 'redmule_fp16_conv3d': first 4 act values %x %x first 4 weights %x %x\r\n",
        ((uint32_t*)input)[0], ((uint32_t*)input)[1], ((uint32_t*)weights)[0], ((uint32_t*)weights)[1]
    );
    #endif

    redmule_fp16_conv3d_dhwn_rd(
        input,
        weights,
        output_matrix,
        im2col_pt_,
        bias,
        apply_relu,
        c_in, d_in, h_in, w_in,
        c_out, d_out, h_out, w_out,
        k_d, k_h, k_w,
        s_d, s_h, s_w,
        pad_d_top, pad_d_bottom,
        pad_h_top, pad_h_bottom,
        pad_w_left, pad_w_right
    );
}


void odl_naive_parallel_conv2d_transpose_fp16(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1; // output tensor is always the last one
    MatchConv2DTransposeAttrs* conv_attrs = (MatchConv2DTransposeAttrs*)ctx->ops->ops[0].attrs;
    
    _Float16 * __restrict__ activations_pt = tensors[0].pt;
    _Float16 * __restrict__ parameters_pt = tensors[1].pt;
    _Float16 * __restrict__ output_pt = tensors[num_tensors-1].pt;
    _Float16 * __restrict__ bias_pt = NULL;
    if (num_tensors > 3) bias_pt = tensors[2].pt;

    int ch_idx = 3; // channel index
    int height_idx = 1; // height index
    int width_idx = 2; // width index
    int HWC_LAYOUT = 1;

    // out chw
    int out_batches = tensors[output_tensor_idx].tiles[MEM_L1_PULPD*4+0].size; // out batches
    int out_width = tensors[output_tensor_idx].tiles[MEM_L1_PULPD*4+width_idx].size; // out width
    int out_height = tensors[output_tensor_idx].tiles[MEM_L1_PULPD*4+height_idx].size; // out height
    int out_ch = tensors[output_tensor_idx].tiles[MEM_L1_PULPD*4+ch_idx].size; // out ch
    int out_ch_params = tensors[1].tiles[MEM_L1_PULPD*4+ch_idx].size; // out ch params
    // inp chw
    int inp_batches = tensors[0].tiles[MEM_L1_PULPD*4+0].size; // in batches
    int inp_width = tensors[0].tiles[MEM_L1_PULPD*4+width_idx].size; // out width
    int inp_height = tensors[0].tiles[MEM_L1_PULPD*4+height_idx].size; // out height
    int inp_ch = tensors[0].tiles[MEM_L1_PULPD*4+ch_idx].size; // out ch
    // pad
    int idx_remainder_top_int_pad = tensors[0].tiles[MEM_L1_PULPD*4+height_idx].idx_remainder!=0.0f?
        tensors[0].tiles[MEM_L1_PULPD*4+height_idx].idx_remainder>0.0f?
        -1: 1: 0; // int padding remainder for top pad
    int idx_remainder_left_int_pad = tensors[0].tiles[MEM_L1_PULPD*4+width_idx].idx_remainder!=0.0f?
        tensors[0].tiles[MEM_L1_PULPD*4+width_idx].idx_remainder>0.0f?
        -1: 1: 0; // int padding remainder for left pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+height_idx])) + idx_remainder_top_int_pad;
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+width_idx])) + idx_remainder_left_int_pad;
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+height_idx]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_PULPD*4+width_idx]));
    int stride_h = conv_attrs->strides[0];
    int stride_w = conv_attrs->strides[1];
    int groups = conv_attrs->groups;
    int dilation_h = conv_attrs->dilation[0];
    int dilation_w = conv_attrs->dilation[1];
    int kernel_h = conv_attrs->kernel_size[0];
    int kernel_w = conv_attrs->kernel_size[1];
    int is_dw = conv_attrs->depthwise;

    #if DEBUG_CLUSTER_LIB
    smp_printf("[TRANS] Out tile [%d %d %d %d] Inp tile [%d %d %d %d] pad ^ %d v %d < %d > %d Strides < %d %d > Dil < %d %d >",
        out_batches, out_ch, out_height, out_width,
        inp_batches, inp_ch, inp_height, inp_width,
        pad_top, pad_bottom, pad_left, pad_right,
        stride_h, stride_w, dilation_h, dilation_w
    );
    smp_printf(" Num tensors: %d\r\n", num_tensors);
    #endif

    const uint32_t ker_spat_size = kernel_h*kernel_w-1;

    uint32_t start_c_out = 0, stop_c_out = out_ch;
    uint32_t start_h_out = 0, stop_h_out = out_height;
    uint32_t start_w_out = 0, stop_w_out = out_width;
    int NUM_CORES = 8;
    int prefer_blocking_over = out_ch >= out_height && out_ch >= out_width ? 0 :
                                out_height >= out_width ? 1 : 2;
    if(!prefer_blocking_over) {
        int block_size_c_out = (out_ch+NUM_CORES-1) / NUM_CORES;
        start_c_out = rt_core_id() * block_size_c_out;
        stop_c_out = start_c_out + block_size_c_out > out_ch ? out_ch : start_c_out + block_size_c_out;
    } else if(prefer_blocking_over == 1) {
        int block_size_h_out = (out_height+NUM_CORES-1) / NUM_CORES;
        start_h_out = rt_core_id() * block_size_h_out;
        stop_h_out = start_h_out + block_size_h_out > out_height ? out_height : start_h_out + block_size_h_out;
    } else {
        int block_size_w_out = (out_width+NUM_CORES-1) / NUM_CORES;
        start_w_out = rt_core_id() * block_size_w_out;
        stop_w_out = start_w_out + block_size_w_out > out_width ? out_width : start_w_out + block_size_w_out;
    }

    for (uint32_t c_out_idx=start_c_out; c_out_idx<stop_c_out; c_out_idx++) {
        _Float16 bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
        int c_offset_out = c_out_idx * out_height * out_width;
        for (uint32_t h_out_idx=start_h_out; h_out_idx<stop_h_out; h_out_idx++) {
            int h_out_offset = h_out_idx * out_width;
            for (uint32_t w_out_idx=start_w_out; w_out_idx<stop_w_out; w_out_idx++) {
                _Float16 tmp = 0.0f;
                int out_idx = w_out_idx + h_out_offset + c_offset_out;
                for (uint32_t h_ker_idx=0; h_ker_idx<kernel_h; h_ker_idx++) {
                    int h_act_idx = (h_out_idx + h_ker_idx - pad_top);
                    if (stride_h > 1) {
                        if (h_act_idx % stride_h != 0) continue;
                        h_act_idx /= stride_h;
                    }
                    if (h_act_idx < 0 || h_act_idx >= inp_height) continue;
                    for (uint32_t w_ker_idx=0; w_ker_idx<kernel_w; w_ker_idx++) {
                        int w_act_idx = (w_out_idx + w_ker_idx - pad_left);
                        if (stride_w > 1) {
                            if (w_act_idx % stride_w != 0) continue;
                            w_act_idx /= stride_w;
                        }
                        if (w_act_idx < 0 || w_act_idx >= inp_width) continue;
                        
                        for (uint32_t c_acts_idx=0; c_acts_idx<inp_ch; c_acts_idx++){
                            int params_idx = (kernel_h - 1 - h_ker_idx) * kernel_w + (kernel_w - 1 - w_ker_idx) + (c_acts_idx*kernel_h*kernel_w*out_ch_params) + (c_out_idx*kernel_h*kernel_w);
                            // int params_idx = (ker_spat_size-w_ker_idx-h_ker_idx*kernel_w) + c_out_idx*kernel_w*kernel_h + c_acts_idx*kernel_w*kernel_h*out_ch_params;
                            int act_idx = w_act_idx + h_act_idx*inp_width + c_acts_idx*inp_height*inp_width;
                            
                            tmp += parameters_pt[params_idx] * activations_pt[act_idx];
                        }
                    }
                }
                output_pt[out_idx] = tmp + bias_val;
            } 
        }
    }
}

#endif // __pulp_cluster__