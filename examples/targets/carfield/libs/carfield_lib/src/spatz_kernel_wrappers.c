#ifdef __spatz__

#include "carfield_lib/spatz.h"

#include <stdint.h>
#include <stddef.h>

#include "carfield_lib/printf.h"
#include "carfield_lib/utils.h"

#include "match/ctx.h"
#include "carfield.h"

#include "snrt.h"

#include "spatz_kernels/spatz_fp16_kernels.h"



static void wtf_wrapper(MatchCtx *ctx) {
    smp_printf("[SPATZ][KER] No kernel found for pattern ID %d. Crashing... :(\r\n", ctx->pattern_name);
    exit(1);
}


// Feed MatchCtx to library kernel APIs
void kernel_wrapper(MatchCtx* ctx) 
{    
    switch(ctx->pattern_name){

    // Pulp NN fp16
    #ifdef spatz_conv2d_fp16
        case spatz_conv2d_fp16: spatz_fp16_conv2d_wrapper(ctx); break;
    #endif
    #ifdef spatz_conv2d_bias_fp16
        case spatz_conv2d_bias_fp16: spatz_fp16_conv2d_wrapper(ctx); break;
    #endif
    #ifdef spatz_conv2d_bnorm_fp16
        case spatz_conv2d_bnorm_fp16: spatz_fp16_conv2d_wrapper(ctx); break;
    #endif

    #ifdef spatz_conv2d_grouped_fp16:
        case spatz_conv2d_grouped_fp16: spatz_fp16_conv2d_grouped_wrapper(ctx); break;
    #endif
    #ifdef spatz_conv2d_grouped_bias_fp16:
        case spatz_conv2d_grouped_bias_fp16: spatz_fp16_conv2d_grouped_wrapper(ctx); break;
    #endif
    #ifdef spatz_conv2d_grouped_bnorm_fp16:
        case spatz_conv2d_grouped_bnorm_fp16: spatz_fp16_conv2d_grouped_wrapper(ctx); break;
    #endif

    #ifdef spatz_dense_fp16
        case spatz_dense_fp16: spatz_fp16_dense_wrapper(ctx); break;
    #endif
    #ifdef spatz_dense_bias_fp16
        case spatz_dense_bias_fp16: spatz_fp16_dense_wrapper(ctx); break;
    #endif

        default: wtf_wrapper(ctx); break;
    }
    
}


void spatz_fp16_conv2d_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;

    void *input = tensors[0].pt;
    void *weight = tensors[1].pt;
    void *bias = num_tensors > 3 ? tensors[2].pt : NULL;
    void *bnorm_mul = num_tensors > 4 ? tensors[3].pt : NULL;
    void *bnorm_add = num_tensors > 4 ? tensors[4].pt : NULL;
    void *output = tensors[num_tensors-1].pt;
    // void *im2col = im2col_pt_;

    int out_width = tensors[num_tensors-1].tiles[MEM_L1_SPATZ*4+2].size; 
    int out_height = tensors[num_tensors-1].tiles[MEM_L1_SPATZ*4+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_SPATZ*4+3].size;

    int inp_width = tensors[0].tiles[MEM_L1_SPATZ*4+2].size; 
    int inp_height = tensors[0].tiles[MEM_L1_SPATZ*4+1].size;
    int inp_ch = tensors[0].tiles[MEM_L1_SPATZ*4+3].size;

    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_SPATZ*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_SPATZ*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_SPATZ*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_SPATZ*4+2]));
    
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    int filter_width = conv_attrs->kernel_size[1];
    int filter_height = conv_attrs->kernel_size[0];
    int stride_x = conv_attrs->strides[1];
    int stride_y = conv_attrs->strides[0];

    int apply_relu = 0;

#if DEBUG_SPATZ_LIB
    smp_printf("[SPATZ][KER] spatz_fp16_conv2d: ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Inp. tile (%d,%d,%d) | ", inp_ch, inp_height, inp_width);
    smp_printf("Pad ▲ %d ▼ %d ◄ %d ► %d\r\n", pad_top, pad_bottom, pad_left, pad_right);
#endif

    spatz_fp16_conv2d(
        input, weight, bias, bnorm_mul, bnorm_add, output, 0,
        inp_width, inp_height, inp_ch,
        out_width,out_height, out_ch, 
        filter_width, filter_height,
        pad_top, pad_bottom, pad_left, pad_right,
        stride_x, stride_y,
        apply_relu
    );
}



void spatz_fp16_conv2d_grouped_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;

    void *input = tensors[0].pt;
    void *weight = tensors[1].pt;
    void *bias = num_tensors > 3 ? tensors[2].pt : NULL;
    void *bnorm_mul = num_tensors > 4 ? tensors[3].pt : NULL;
    void *bnorm_add = num_tensors > 4 ? tensors[4].pt : NULL;
    void *output = tensors[num_tensors-1].pt;
    // void *im2col = im2col_pt_;

    int out_width = tensors[num_tensors-1].tiles[MEM_L1_SPATZ*4+2].size; 
    int out_height = tensors[num_tensors-1].tiles[MEM_L1_SPATZ*4+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_SPATZ*4+3].size;

    int inp_width = tensors[0].tiles[MEM_L1_SPATZ*4+2].size; 
    int inp_height = tensors[0].tiles[MEM_L1_SPATZ*4+1].size;
    int inp_ch = tensors[0].tiles[MEM_L1_SPATZ*4+3].size;

    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_SPATZ*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[MEM_L1_SPATZ*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_SPATZ*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[MEM_L1_SPATZ*4+2]));
    
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    int filter_width = conv_attrs->kernel_size[1];
    int filter_height = conv_attrs->kernel_size[0];
    int stride_x = conv_attrs->strides[1];
    int stride_y = conv_attrs->strides[0];
    int groups = conv_attrs->groups;
    int apply_relu = 0;

#if DEBUG_SPATZ_LIB
    smp_printf("[SPATZ][KER] spatz_fp16_conv2d_grouped: ");
    smp_printf("Out. tile (%d,%d,%d) | ", out_ch, out_height, out_width);
    smp_printf("Inp. tile (%d,%d,%d) | ", inp_ch, inp_height, inp_width);
    smp_printf("Pad ▲ %d ▼ %d ◄ %d ► %d | ", pad_top, pad_bottom, pad_left, pad_right);
    smp_printf("Groups %d\r\n", groups);
#endif

    spatz_fp16_conv2d_grouped(
        input, weight, bias, bnorm_mul, bnorm_add, output, 0,
        inp_width, inp_height, inp_ch,
        out_width,out_height, out_ch, 
        filter_width, filter_height,
        pad_top, pad_bottom, pad_left, pad_right,
        stride_x, stride_y,
        apply_relu,
        groups
    );
}



void spatz_fp16_dense_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int inp_ch = tensors[0].tiles[MEM_L1_SPATZ*2+1].size;
    int out_ch = tensors[num_tensors-1].tiles[MEM_L1_SPATZ*2+1].size;

#if DEBUG_SPATZ_LIB
    smp_printf("[SPATZ][KER] spatz_fp16_linear: ");
    smp_printf("Out. tile (%d,) | ", out_ch);
    smp_printf("Inp. tile (%d,)\r\n", inp_ch);
#endif

    spatz_fp16_linear(
        tensors[0].pt,             // Activations pt
        tensors[1].pt,             // Weights pt
        num_tensors > 3 ? tensors[2].pt : NULL, // Bias ptr
        tensors[num_tensors-1].pt, // Output pt
        inp_ch,                    // Input Neurons
        out_ch                     // Output Neurons
    );
}

#endif // __spatz__