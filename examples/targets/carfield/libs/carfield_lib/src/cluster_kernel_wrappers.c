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
    #ifdef dense
        case dense:             pulp_nn_dense_wrapper(ctx);             break;
    #endif
    #ifdef conv2d
        case conv2d:            pulp_nn_hoparallel_conv2d_wrapper(ctx); break;
    #endif
    #ifdef dense_out
        case dense_out:         pulp_nn_dense_out_int_wrapper(ctx);     break;
    #endif
    #ifdef depthwise_conv2d
        case depthwise_conv2d:  pulp_nn_dw_conv2d_wrapper(ctx);         break;
    #endif
    #ifdef pointwise_conv2d
        case pointwise_conv2d:  pulp_nn_pw_conv2d_wrapper(ctx);         break;
    #endif
    #ifdef add_requant
        case add_requant:       pulp_nn_add_wrapper(ctx);               break;
    #endif

    // Pulp NN fp16
    #ifdef dense_fp16
        case dense_fp16:        pulp_fp16_dense_wrapper(ctx);           break;
    #endif
    #ifdef conv2d_fp16
        case conv2d_fp16:       pulp_fp16_conv2d_wrapper(ctx);          break;
    #endif
    #ifdef avgpool2d_fp16
        case avgpool2d_fp16:    pulp_fp16_avgpool2d_wrapper(ctx);       break;
    #endif

        default:                wtf_wrapper(ctx);                       break;
    }
}


/* ======== PULP-NN int8 ======== */


void pulp_nn_dense_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1*2+1].size;
    int inp_ch = tensors[0].tiles[MEM_L1*2+1].size;

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
    int inp_ch = tensors[0].tiles[MEM_L1*2+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1*2+1].size;
    
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
    int out_width = tensors[num_tensors-1].tiles[MEM_L1*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[MEM_L1*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1*4+3].size; // out ch
    // Input
    int inp_width = tensors[0].tiles[MEM_L1*4+2].size; // out width
    int inp_height = tensors[0].tiles[MEM_L1*4+1].size; // out height
    int inp_ch = tensors[0].tiles[MEM_L1*4+3].size; // out ch
    // Padding
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1*4+2]));

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
    int out_width = tensors[num_tensors-1].tiles[MEM_L1*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[MEM_L1*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1*4+3].size; // out ch
    // Input
    int inp_width = tensors[0].tiles[MEM_L1*4+2].size; // out width
    int inp_height = tensors[0].tiles[MEM_L1*4+1].size; // out height
    int inp_ch = tensors[0].tiles[MEM_L1*4+3].size; // out ch
    // Padding
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1*4+2]));

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
    int out_width = tensors[num_tensors-1].tiles[MEM_L1*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[MEM_L1*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1*4+3].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[MEM_L1*4+2].size; // out width
    int inp_height = tensors[0].tiles[MEM_L1*4+1].size; // out height
    int inp_ch = tensors[0].tiles[MEM_L1*4+3].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1*4+2]));

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
    int out_width = tensors[num_tensors-1].tiles[MEM_L1*4+2].size;
    int out_height = tensors[num_tensors-1].tiles[MEM_L1*4+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1*4+3].size;

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
    int inp_ch = tensors[0].tiles[MEM_L1*2+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1*2+1].size;

    // TODO improve this - use RedMulE when supported
    if (inp_ch == 64 && out_ch == 10) {
        redmule_fp16_dense_wrapper(ctx);
        return;
    }
    
#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] pulp_fp16_linear: ");
    smp_printf("Out. tile (%d,) | ", out_ch);
    smp_printf("Inp. tile (%d,)\r\n", inp_ch);
#endif

    pulp_fp16_linear(
        tensors[0].pt,             // Activations pt
        tensors[1].pt,             // Weights pt
        tensors[num_tensors-1].pt, // Output pt
        num_tensors > 3 ? tensors[2].pt : NULL, // Bias ptr
        inp_ch,                    // Input Neurons
        out_ch                     // Output Neurons
    );
}


void pulp_fp16_conv2d_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;

    void *input = tensors[0].pt;
    void *weight = tensors[1].pt;
    void *bias = num_tensors > 3 ? tensors[2].pt : NULL;
    void *output = tensors[num_tensors-1].pt;
    void *im2col = im2col_pt_;

    int out_width = tensors[num_tensors-1].tiles[MEM_L1*4+2].size; 
    int out_height = tensors[num_tensors-1].tiles[MEM_L1*4+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1*4+3].size;

    int inp_width = tensors[0].tiles[MEM_L1*4+2].size; 
    int inp_height = tensors[0].tiles[MEM_L1*4+1].size;
    int inp_ch = tensors[0].tiles[MEM_L1*4+3].size;

    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1*4+2]));
    
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    int filter_width = conv_attrs->kernel_size[1];
    int filter_height = conv_attrs->kernel_size[0];
    int stride_x = conv_attrs->strides[1];
    int stride_y = conv_attrs->strides[0];

    int apply_relu = 0;

#if DEBUG_CLUSTER_LIB
    smp_printf("[PULP][KER] pulp_fp16_conv2d: ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Inp. tile (%d,%d,%d) | ", inp_ch, inp_height, inp_width);
    smp_printf("Pad ▲ %d ▼ %d ◄ %d ► %d\r\n", pad_top, pad_bottom, pad_left, pad_right);
#endif

    pulp_fp16_conv2d(
        input, weight, bias, output, im2col,
        inp_width, inp_height, inp_ch,
        out_width,out_height, out_ch, 
        filter_width, filter_height,
        pad_top, pad_bottom, pad_left, pad_right,
        stride_x, stride_y,
        apply_relu
    );
}



void pulp_fp16_avgpool2d_wrapper(MatchCtx* ctx){
    // TODO add support in MATCH
}




/* ======== RedMulE [WIP] ======== */


void redmule_fp16_dense_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int inp_neurons = tensors[0].tiles[MEM_L1*2+1].size;
    int out_neurons = tensors[num_tensors-1].tiles[MEM_L1*2+1].size;

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

#endif // __pulp_cluster__