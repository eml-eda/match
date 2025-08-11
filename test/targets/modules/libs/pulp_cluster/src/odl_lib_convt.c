#include <pulp_cluster/odl_lib.h>

void odl_naive_parallel_conv2d_transpose_stride_1_fp32(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1; // output tensor is always the last one
    MatchConv2DTransposeAttrs* conv_attrs = (MatchConv2DTransposeAttrs*)ctx->ops->ops[0].attrs;
    
    float * __restrict__ activations_pt = tensors[0].pt;
    float * __restrict__ parameters_pt = tensors[1].pt;
    float * __restrict__ output_pt = tensors[num_tensors-1].pt;
    float * __restrict__ bias_pt = NULL;
    if (num_tensors > 3) bias_pt = tensors[2].pt;

    int ch_idx = 1; // channel index
    int height_idx = 2; // height index
    int width_idx = 3; // width index
    int HWC_LAYOUT = 0;

    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW"){
        ch_idx = 3; // channel index
        height_idx = 1; // height index
        width_idx = 2; // width index
        HWC_LAYOUT = 1; // Choose if data layout is CHW (=0) or HWC (=1)
    }
    #endif

    // out chw
    int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size; // out batches
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch params
    // inp chw
    int inp_batches = tensors[0].tiles[L1_SCRATCHPAD*4+0].size; // in batches
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    // pad
    int idx_remainder_top_int_pad = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].idx_remainder!=0.0f?
        tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].idx_remainder>0.0f?
        -1: 1: 0; // int padding remainder for top pad
    int idx_remainder_left_int_pad = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].idx_remainder!=0.0f?
        tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].idx_remainder>0.0f?
        -1: 1: 0; // int padding remainder for left pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx])) + idx_remainder_top_int_pad;
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx])) + idx_remainder_left_int_pad;
    int kernel_h = conv_attrs->kernel_size[0];
    int kernel_w = conv_attrs->kernel_size[1];
    int is_dw = conv_attrs->depthwise;
    #ifdef CLUSTER_LIB_DEBUG
    printf("Out tile [%d %d %d %d] Inp tile [%d %d %d %d] pad ^ %d < %d\n",
        out_batches, out_ch, out_height, out_width,
        inp_batches, inp_ch, inp_height, inp_width,
        pad_top, pad_left
    );
    printf("Num tensors: %d\n", num_tensors);
    #endif
    const uint32_t ker_spat_size = kernel_h*kernel_w-1;

    uint32_t start_c_out = 0, stop_c_out = out_ch;
    uint32_t start_h_out = 0, stop_h_out = out_height;
    uint32_t start_w_out = 0, stop_w_out = out_width;
    #if NUM_CORES > 1
    int prefer_blocking_over = out_ch >= out_height && out_ch >= out_width ? 0 :
                                out_height >= out_width ? 1 : 2;
    if(!prefer_blocking_over) {
        int block_size_c_out = (out_ch+NUM_CORES-1) / NUM_CORES;
        start_c_out = pi_core_id() * block_size_c_out;
        stop_c_out = start_c_out + block_size_c_out > out_ch ? out_ch : start_c_out + block_size_c_out;
    } else if(prefer_blocking_over == 1) {
        int block_size_h_out = (out_height+NUM_CORES-1) / NUM_CORES;
        start_h_out = pi_core_id() * block_size_h_out;
        stop_h_out = start_h_out + block_size_h_out > out_height ? out_height : start_h_out + block_size_h_out;
    } else {
        int block_size_w_out = (out_width+NUM_CORES-1) / NUM_CORES;
        start_w_out = pi_core_id() * block_size_w_out;
        stop_w_out = start_w_out + block_size_w_out > out_width ? out_width : start_w_out + block_size_w_out;
    }
    #endif

    if(!is_dw){
        for (uint32_t c_out_idx=start_c_out; c_out_idx<stop_c_out; c_out_idx++) {
            float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            int c_offset_out = c_out_idx * out_height * out_width;
            for (uint32_t h_out_idx=start_h_out; h_out_idx<stop_h_out; h_out_idx++) {
                int h_out_offset = h_out_idx * out_width;
                for (uint32_t w_out_idx=start_w_out; w_out_idx<stop_w_out; w_out_idx++) {
                    float tmp = 0.0f;
                    int out_idx = w_out_idx + h_out_offset + c_offset_out;
                    for (uint32_t h_ker_idx=0; h_ker_idx<kernel_h; h_ker_idx++) {
                        int h_act_idx = (h_out_idx + h_ker_idx - pad_top);
                        if (h_act_idx<0 || h_act_idx >= inp_height) continue;
                        for (uint32_t w_ker_idx=0; w_ker_idx<kernel_w; w_ker_idx++) {
                            int w_act_idx = (w_out_idx + w_ker_idx - pad_left);
                            if (w_act_idx<0 || w_act_idx >= inp_width) continue;
                            #ifdef ODL_DEBUG_KERNEL
                            if(c_out_idx == 0)
                                printf("h_out_idx: %d, w_out_idx: %d, h_ker_idx: %d, w_ker_idx: %d, h_act_idx: %d, w_act_idx: %d\n", 
                                        h_out_idx, w_out_idx, h_ker_idx, w_ker_idx, h_act_idx, w_act_idx);
                            #endif
                            for (uint32_t c_acts_idx=0; c_acts_idx<inp_ch; c_acts_idx++){
                                int params_idx = (kernel_h - 1 - h_ker_idx) * kernel_w + (kernel_w - 1 - w_ker_idx) + (c_acts_idx*kernel_h*kernel_w*out_ch_params) + (c_out_idx*kernel_h*kernel_w);
                                // int params_idx = (ker_spat_size-w_ker_idx-h_ker_idx*kernel_w) + c_out_idx*kernel_w*kernel_h + c_acts_idx*kernel_w*kernel_h*out_ch_params;
                                int act_idx = w_act_idx + h_act_idx*inp_width + c_acts_idx*inp_height*inp_width;
                                #ifdef ODL_DEBUG_KERNEL
                                printf("C[%d] += A[%d] * B[%d] -> %f += %f * %f\n", 
                                    out_idx, params_idx, act_idx, tmp, parameters_pt[params_idx], activations_pt[act_idx]);
                                #endif
                                tmp += parameters_pt[params_idx] * activations_pt[act_idx];
                            }
                        }
                    }
                    #ifdef ODL_DEBUG_KERNEL
                    if(c_out_idx == 0) printf("C[%d] = %f\n", out_idx, tmp);
                    #endif
                    output_pt[out_idx] = tmp + bias_val;
                } 
            }
        }
    }
    else{
        for (uint32_t c_out_idx=start_c_out; c_out_idx<stop_c_out; c_out_idx++) {
            float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            int c_offset_out = c_out_idx * out_height * out_width;
            for (uint32_t h_out_idx=start_h_out; h_out_idx<stop_h_out; h_out_idx++) {
                int h_out_offset = h_out_idx * out_width;
                for (uint32_t w_out_idx=start_w_out; w_out_idx<stop_w_out; w_out_idx++) {
                    float tmp = 0.0f;
                    int out_idx = w_out_idx + h_out_offset + c_offset_out;
                    for (uint32_t h_ker_idx=0; h_ker_idx<kernel_h; h_ker_idx++) {
                        int h_act_idx = (h_out_idx + h_ker_idx - pad_top);
                        if (h_act_idx<0 || h_act_idx >= inp_height) continue;
                        for (uint32_t w_ker_idx=0; w_ker_idx<kernel_w; w_ker_idx++) {
                            int w_act_idx = (w_out_idx + w_ker_idx - pad_left);
                            if (w_act_idx<0 || w_act_idx >= inp_width) continue;
                            #ifdef ODL_DEBUG_KERNEL
                            if(c_out_idx == 0)
                                printf("h_out_idx: %d, w_out_idx: %d, h_ker_idx: %d, w_ker_idx: %d, h_act_idx: %d, w_act_idx: %d\n", 
                                        h_out_idx, w_out_idx, h_ker_idx, w_ker_idx, h_act_idx, w_act_idx);
                            #endif
                            int params_idx = (kernel_h - 1 - h_ker_idx) * kernel_w + (kernel_w - 1 - w_ker_idx) + (c_out_idx*kernel_h*kernel_w);
                            // int params_idx = (ker_spat_size-w_ker_idx-h_ker_idx*kernel_w) + c_out_idx*kernel_w*kernel_h;
                            int act_idx = w_act_idx + h_act_idx*inp_width + c_out_idx*inp_height*inp_width;
                            #ifdef ODL_DEBUG_KERNEL
                            printf("C[%d] += A[%d] * B[%d] -> %f += %f * %f\n", 
                                out_idx, params_idx, act_idx, tmp, parameters_pt[params_idx], activations_pt[act_idx]);
                            #endif
                            tmp += parameters_pt[params_idx] * activations_pt[act_idx];
                        }
                    }
                    #ifdef ODL_DEBUG_KERNEL
                    if(c_out_idx == 0) printf("C[%d] = %f\n", out_idx, tmp);
                    #endif
                    output_pt[out_idx] = tmp + bias_val;
                }
            }
        }
    }
}

void odl_naive_parallel_conv2d_transpose_stride_2_fp32(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1; // output tensor is always the last one
    MatchConv2DTransposeAttrs* conv_attrs = (MatchConv2DTransposeAttrs*)ctx->ops->ops[0].attrs;
    
    float * __restrict__ activations_pt = tensors[0].pt;
    float * __restrict__ parameters_pt = tensors[1].pt;
    float * __restrict__ output_pt = tensors[num_tensors-1].pt;
    float * __restrict__ bias_pt = NULL;
    if (num_tensors > 3) bias_pt = tensors[2].pt;

    int ch_idx = 1; // channel index
    int height_idx = 2; // height index
    int width_idx = 3; // width index
    int HWC_LAYOUT = 0;

    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW"){
        ch_idx = 3; // channel index
        height_idx = 1; // height index
        width_idx = 2; // width index
        HWC_LAYOUT = 1; // Choose if data layout is CHW (=0) or HWC (=1)
    }
    #endif

    // out chw
    int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size; // out batches
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch params
    // inp chw
    int inp_batches = tensors[0].tiles[L1_SCRATCHPAD*4+0].size; // in batches
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    // pad
    int idx_remainder_top_int_pad = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].idx_remainder!=0.0f?
        tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].idx_remainder>0.0f?
        -1: 1: 0; // int padding remainder for top pad
    int idx_remainder_left_int_pad = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].idx_remainder!=0.0f?
        tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].idx_remainder>0.0f?
        -1: 1: 0; // int padding remainder for left pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx])) + idx_remainder_top_int_pad;
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx])) + idx_remainder_left_int_pad;
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int groups = conv_attrs->groups;
    int dilation_h = conv_attrs->dilation[0];
    int dilation_w = conv_attrs->dilation[1];
    int kernel_h = conv_attrs->kernel_size[0];
    int kernel_w = conv_attrs->kernel_size[1];
    int is_dw = conv_attrs->depthwise;
    #ifdef CLUSTER_LIB_DEBUG
    printf("Out tile [%d %d %d %d] Inp tile [%d %d %d %d] pad ^ %d v %d < %d > %d Strides < %d %d > Dil < %d %d >\n",
        out_batches, out_ch, out_height, out_width,
        inp_batches, inp_ch, inp_height, inp_width,
        pad_top, pad_bottom, pad_left, pad_right,
        dilation_h, dilation_w
    );
    printf("Num tensors: %d\n", num_tensors);
    #endif
    const uint32_t ker_spat_size = kernel_h*kernel_w-1;

    uint32_t start_c_out = 0, stop_c_out = out_ch;
    uint32_t start_h_out = 0, stop_h_out = out_height;
    uint32_t start_w_out = 0, stop_w_out = out_width;
    #if NUM_CORES > 1
    int prefer_blocking_over = out_ch >= out_height && out_ch >= out_width ? 0 :
                                out_height >= out_width ? 1 : 2;
    if(!prefer_blocking_over) {
        int block_size_c_out = (out_ch+NUM_CORES-1) / NUM_CORES;
        start_c_out = pi_core_id() * block_size_c_out;
        stop_c_out = start_c_out + block_size_c_out > out_ch ? out_ch : start_c_out + block_size_c_out;
    } else if(prefer_blocking_over == 1) {
        int block_size_h_out = (out_height+NUM_CORES-1) / NUM_CORES;
        start_h_out = pi_core_id() * block_size_h_out;
        stop_h_out = start_h_out + block_size_h_out > out_height ? out_height : start_h_out + block_size_h_out;
    } else {
        int block_size_w_out = (out_width+NUM_CORES-1) / NUM_CORES;
        start_w_out = pi_core_id() * block_size_w_out;
        stop_w_out = start_w_out + block_size_w_out > out_width ? out_width : start_w_out + block_size_w_out;
    }
    #endif

    if(!is_dw){
        for (uint32_t c_out_idx=start_c_out; c_out_idx<stop_c_out; c_out_idx++) {
            float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            int c_offset_out = c_out_idx * out_height * out_width;
            for (uint32_t h_out_idx=start_h_out; h_out_idx<stop_h_out; h_out_idx++) {
                int h_out_offset = h_out_idx * out_width;
                for (uint32_t w_out_idx=start_w_out; w_out_idx<stop_w_out; w_out_idx++) {
                    float tmp = 0.0f;
                    int out_idx = w_out_idx + h_out_offset + c_offset_out;
                    for (uint32_t h_ker_idx=0; h_ker_idx<kernel_h; h_ker_idx++) {
                        int h_act_idx = (h_out_idx + h_ker_idx - pad_top);
                        if (h_act_idx & 1 || h_act_idx<0) continue;
                        h_act_idx >>= 1;
                        if (h_act_idx >= inp_height) continue;
                        for (uint32_t w_ker_idx=0; w_ker_idx<kernel_w; w_ker_idx++) {
                            int w_act_idx = (w_out_idx + w_ker_idx - pad_left);
                            if (w_act_idx & 1 || w_act_idx<0) continue;
                            w_act_idx >>= 1;
                            if (w_act_idx >= inp_width) continue;
                            #ifdef ODL_DEBUG_KERNEL
                            if(c_out_idx == 0)
                                printf("h_out_idx: %d, w_out_idx: %d, h_ker_idx: %d, w_ker_idx: %d, h_act_idx: %d, w_act_idx: %d\n", 
                                        h_out_idx, w_out_idx, h_ker_idx, w_ker_idx, h_act_idx, w_act_idx);
                            #endif
                            for (uint32_t c_acts_idx=0; c_acts_idx<inp_ch; c_acts_idx++){
                                int params_idx = (kernel_h - 1 - h_ker_idx) * kernel_w + (kernel_w - 1 - w_ker_idx) + (c_acts_idx*kernel_h*kernel_w*out_ch_params) + (c_out_idx*kernel_h*kernel_w);
                                // int params_idx = (ker_spat_size-w_ker_idx-h_ker_idx*kernel_w) + c_out_idx*kernel_w*kernel_h + c_acts_idx*kernel_w*kernel_h*out_ch_params;
                                int act_idx = w_act_idx + h_act_idx*inp_width + c_acts_idx*inp_height*inp_width;
                                #ifdef ODL_DEBUG_KERNEL
                                printf("C[%d] += A[%d] * B[%d] -> %f += %f * %f\n", 
                                    out_idx, params_idx, act_idx, tmp, parameters_pt[params_idx], activations_pt[act_idx]);
                                #endif
                                tmp += parameters_pt[params_idx] * activations_pt[act_idx];
                            }
                        }
                    }
                    #ifdef ODL_DEBUG_KERNEL
                    if(c_out_idx == 0) printf("C[%d] = %f\n", out_idx, tmp);
                    #endif
                    output_pt[out_idx] = tmp + bias_val;
                } 
            }
        }
    }
    else{
        for (uint32_t c_out_idx=start_c_out; c_out_idx<stop_c_out; c_out_idx++) {
            float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            int c_offset_out = c_out_idx * out_height * out_width;
            for (uint32_t h_out_idx=start_h_out; h_out_idx<stop_h_out; h_out_idx++) {
                int h_out_offset = h_out_idx * out_width;
                for (uint32_t w_out_idx=start_w_out; w_out_idx<stop_w_out; w_out_idx++) {
                    float tmp = 0.0f;
                    int out_idx = w_out_idx + h_out_offset + c_offset_out;
                    for (uint32_t h_ker_idx=0; h_ker_idx<kernel_h; h_ker_idx++) {
                        int h_act_idx = (h_out_idx + h_ker_idx - pad_top);
                        if (h_act_idx & 1 || h_act_idx<0) continue;
                        h_act_idx >>= 1;
                        if (h_act_idx >= inp_height) continue;
                        for (uint32_t w_ker_idx=0; w_ker_idx<kernel_w; w_ker_idx++) {
                            int w_act_idx = (w_out_idx + w_ker_idx - pad_left);
                            if (w_act_idx & 1 || w_act_idx<0) continue;
                            w_act_idx >>= 1;
                            if (w_act_idx >= inp_width) continue;
                            #ifdef ODL_DEBUG_KERNEL
                            if(c_out_idx == 0)
                                printf("h_out_idx: %d, w_out_idx: %d, h_ker_idx: %d, w_ker_idx: %d, h_act_idx: %d, w_act_idx: %d\n", 
                                        h_out_idx, w_out_idx, h_ker_idx, w_ker_idx, h_act_idx, w_act_idx);
                            #endif
                            int params_idx = (kernel_h - 1 - h_ker_idx) * kernel_w + (kernel_w - 1 - w_ker_idx) + (c_out_idx*kernel_h*kernel_w);
                            // int params_idx = (ker_spat_size-w_ker_idx-h_ker_idx*kernel_w) + c_out_idx*kernel_w*kernel_h;
                            int act_idx = w_act_idx + h_act_idx*inp_width + c_out_idx*inp_height*inp_width;
                            #ifdef ODL_DEBUG_KERNEL
                            printf("C[%d] += A[%d] * B[%d] -> %f += %f * %f\n", 
                                out_idx, params_idx, act_idx, tmp, parameters_pt[params_idx], activations_pt[act_idx]);
                            #endif
                            tmp += parameters_pt[params_idx] * activations_pt[act_idx];
                        }
                    }
                    #ifdef ODL_DEBUG_KERNEL
                    if(c_out_idx == 0) printf("C[%d] = %f\n", out_idx, tmp);
                    #endif
                    output_pt[out_idx] = tmp + bias_val;
                }
            }
        }
    }
}


void odl_naive_parallel_conv2d_transpose_pw_stride_1_fp32(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1; // output tensor is always the last one
    MatchConv2DTransposeAttrs* conv_attrs = (MatchConv2DTransposeAttrs*)ctx->ops->ops[0].attrs;
    
    float * __restrict__ activations_pt = tensors[0].pt;
    float * __restrict__ parameters_pt = tensors[1].pt;
    float * __restrict__ output_pt = tensors[num_tensors-1].pt;
    float * __restrict__ bias_pt = NULL;
    if (num_tensors > 3) bias_pt = tensors[2].pt;

    int ch_idx = 1; // channel index
    int height_idx = 2; // height index
    int width_idx = 3; // width index
    int HWC_LAYOUT = 0;

    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW"){
        ch_idx = 3; // channel index
        height_idx = 1; // height index
        width_idx = 2; // width index
        HWC_LAYOUT = 1; // Choose if data layout is CHW (=0) or HWC (=1)
    }
    #endif

    // out chw
    int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size; // out batches
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch params
    // inp chw
    int inp_batches = tensors[0].tiles[L1_SCRATCHPAD*4+0].size; // in batches
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    
    int is_dw = conv_attrs->depthwise;
    
    const i_spat_size = inp_height*inp_width;
    uint32_t start_c_out = 0, stop_c_out = out_ch;
    int start_spat_out = 0, stop_spat_out = out_height*out_width;
    #if NUM_CORES > 1
    // currently theres a bug with blocking over spatial dimension
    // int prefer_blocking_over = stop_c_out >= stop_spat_out ? 0 : 1;
    int prefer_blocking_over = 0;
    if(!prefer_blocking_over) {
        int block_size_c_out = (out_ch+NUM_CORES-1) / NUM_CORES;
        start_c_out = pi_core_id() * block_size_c_out;
        stop_c_out = start_c_out + block_size_c_out > out_ch ? out_ch : start_c_out + block_size_c_out;
    } else {
        int block_size_spat_out = (stop_spat_out+NUM_CORES-1) / NUM_CORES;
        start_spat_out = pi_core_id() * block_size_spat_out;
        stop_spat_out = start_spat_out + block_size_spat_out > stop_spat_out ? stop_spat_out : start_spat_out + block_size_spat_out;
    }
    #endif
    // #ifdef CLUSTER_LIB_DEBUG
    // printf("[PW] Out tile [%d %d %d %d] Inp tile [%d %d %d %d] start c out %d stop c out %d start spat out %d stop spat out %d\n",
    //     out_batches, out_ch, out_height, out_width,
    //     inp_batches, inp_ch, inp_height, inp_width,
    //     start_c_out, stop_c_out, start_spat_out, stop_spat_out
    // );
    // #endif
    // version 1
    // int out_idx = start_c_out * out_height * out_width;
    // float tmp = 0.0f;
    // int params_idx = start_c_out;
    // int act_idx = start_spat_out;
    // int c_acts_idx = 0;
    // for (uint32_t c_out_idx=start_c_out; c_out_idx<stop_c_out; c_out_idx++) {
    //     for (uint32_t spat_out_idx=start_spat_out; spat_out_idx<stop_spat_out; spat_out_idx++) {
    //         tmp = 0.0f; params_idx = c_out_idx; act_idx = spat_out_idx; c_acts_idx = 0;
    //         for (; c_acts_idx<(inp_ch&(~3)); c_acts_idx+=4){
    //             tmp += parameters_pt[params_idx] * activations_pt[act_idx];
    //             tmp += parameters_pt[params_idx+ out_ch_params] * activations_pt[act_idx + i_spat_size];
    //             tmp += parameters_pt[params_idx + 2 * out_ch_params] * activations_pt[act_idx + 2 * i_spat_size];
    //             tmp += parameters_pt[params_idx + 3 * out_ch_params] * activations_pt[act_idx + 3 * i_spat_size];
    //             params_idx += 4*out_ch_params;
    //             act_idx += 4*i_spat_size;
    //         }
    //         for (; c_acts_idx<inp_ch; c_acts_idx++){
    //             tmp += parameters_pt[params_idx] * activations_pt[act_idx];
    //             params_idx += out_ch_params;
    //             act_idx += i_spat_size;
    //         }
    //         output_pt[out_idx] = tmp;
    //         out_idx++;
    //     }
    // }
    // version 2
    const int unrolled_ch_in = inp_ch&(~3);
    // const int unrolled_stop_spat_out = stop_spat_out &(~1);
    const int out_ch_params_by_2 = out_ch_params << 1; // out_ch_params * 2
    const int out_ch_params_by_3 = out_ch_params_by_2 + out_ch_params; // out_ch_params * 3
    const int i_spat_size_by_2 = i_spat_size << 1; // i_spat_size * 2
    const int i_spat_size_by_3 = i_spat_size_by_2 + i_spat_size; // i_spat_size * 3
    // first zero and then compute the output tensor
    for (uint32_t c_out_idx=start_c_out; c_out_idx<stop_c_out; c_out_idx++) {
        int out_idx = c_out_idx * out_height * out_width + start_spat_out;
        int c_acts_idx=0;
        // zero out the output tensor
        for (uint32_t spat_out_idx=start_spat_out; spat_out_idx<stop_spat_out; spat_out_idx++)
            output_pt[out_idx + spat_out_idx] = 0.0f;
        for (; c_acts_idx<unrolled_ch_in; c_acts_idx+=4){
            int act_idx = c_acts_idx * i_spat_size;
            int ch_in_off = c_acts_idx * out_ch_params;
            float ker_val_1 = parameters_pt[ch_in_off + c_out_idx];
            float ker_val_2 = parameters_pt[ch_in_off + c_out_idx + out_ch_params];
            float ker_val_3 = parameters_pt[ch_in_off + c_out_idx + out_ch_params_by_2];
            float ker_val_4 = parameters_pt[ch_in_off + c_out_idx + out_ch_params_by_3];
            // unrolled version
            // uint32_t spat_out_idx=start_spat_out;
            // for (; spat_out_idx<unrolled_stop_spat_out; spat_out_idx+=2){
            //     float tmp_1 = output_pt[out_idx + spat_out_idx];
            //     float tmp_2 = output_pt[out_idx + spat_out_idx + 1];
            //     float act_val_1_1 = activations_pt[act_idx + spat_out_idx];
            //     float act_val_2_1 = activations_pt[act_idx + spat_out_idx + 1];
            //     tmp_1 += ker_val_1 * act_val_1_1;
            //     tmp_2 += ker_val_1 * act_val_2_1;
            //     float act_val_1_2 = activations_pt[act_idx + spat_out_idx + i_spat_size];
            //     float act_val_2_2 = activations_pt[act_idx + spat_out_idx + i_spat_size + 1];
            //     tmp_1 += ker_val_2 * act_val_1_2;
            //     tmp_2 += ker_val_2 * act_val_2_2;
            //     float act_val_1_3 = activations_pt[act_idx + spat_out_idx + i_spat_size_by_2];
            //     float act_val_2_3 = activations_pt[act_idx + spat_out_idx + i_spat_size_by_2 + 1];
            //     tmp_1 += ker_val_3 * act_val_1_3;
            //     tmp_2 += ker_val_3 * act_val_2_3;
            //     float act_val_1_4 = activations_pt[act_idx + spat_out_idx + i_spat_size_by_3];
            //     float act_val_2_4 = activations_pt[act_idx + spat_out_idx + i_spat_size_by_3 + 1];
            //     tmp_1 += ker_val_4 * act_val_1_4;
            //     tmp_2 += ker_val_4 * act_val_2_4;
            //     output_pt[out_idx + spat_out_idx] = tmp_1;
            //     output_pt[out_idx + spat_out_idx + 1] = tmp_2;
            // }
            for (uint32_t spat_out_idx=start_spat_out; spat_out_idx<stop_spat_out; spat_out_idx++){
                float tmp = output_pt[out_idx + spat_out_idx];
                float act_val_1 = activations_pt[act_idx + spat_out_idx];
                float act_val_2 = activations_pt[act_idx + spat_out_idx + i_spat_size];
                float act_val_3 = activations_pt[act_idx + spat_out_idx + i_spat_size_by_2];
                float act_val_4 = activations_pt[act_idx + spat_out_idx + i_spat_size_by_3];
                tmp += ker_val_1 * act_val_1;
                tmp += ker_val_2 * act_val_2;
                tmp += ker_val_3 * act_val_3;
                tmp += ker_val_4 * act_val_4;
                output_pt[out_idx + spat_out_idx] = tmp;
            }
        }
        for (; c_acts_idx<inp_ch; c_acts_idx++){
            int act_idx = c_acts_idx * i_spat_size;
            float ker_val = parameters_pt[c_acts_idx * out_ch_params + c_out_idx];
            for (uint32_t spat_out_idx=start_spat_out; spat_out_idx<stop_spat_out; spat_out_idx++)
                output_pt[out_idx + spat_out_idx] += ker_val * activations_pt[act_idx + spat_out_idx];
        }
    }
}

// void odl_pulp_trainlib_pw_transpose_fp32(void* args){
//     MatchCtx* ctx = (MatchCtx*)args;
//     MatchTensor* tensors = ctx->tensors->tensors;
//     int num_ops = ctx->ops->num_ops;
//     int num_tensors = ctx->tensors->num_tensors;
//     int output_tensor_idx = num_tensors - 1; // output tensor is always the last one
//     MatchConv2DTransposeAttrs* conv_attrs = (MatchConv2DTransposeAttrs*)ctx->ops->ops[0].attrs;
    
//     float * __restrict__ activations_pt = tensors[0].pt;
//     float * __restrict__ parameters_pt = tensors[1].pt;
//     float * __restrict__ output_pt = tensors[num_tensors-1].pt;
//     float * __restrict__ bias_pt = NULL;
//     if (num_tensors > 3) bias_pt = tensors[2].pt;

//     int ch_idx = 1; // channel index
//     int height_idx = 2; // height index
//     int width_idx = 3; // width index
//     int HWC_LAYOUT = 0;

//     #ifdef ODL_SUPPORT_NHWC
//     if (conv_attrs->data_layout != "NCHW"){
//         ch_idx = 3; // channel index
//         height_idx = 1; // height index
//         width_idx = 2; // width index
//         HWC_LAYOUT = 1; // Choose if data layout is CHW (=0) or HWC (=1)
//     }
//     #endif

//     // out chw
//     int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size; // out batches
//     int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
//     int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
//     int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
//     int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch params
//     // inp chw
//     int inp_batches = tensors[0].tiles[L1_SCRATCHPAD*4+0].size; // in batches
//     int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
//     int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
//     int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch

//     struct transp_args tr_args;
//     tr_args.matrix = parameters_pt;
//     void* tr_buffer = parameters_pt;
//     tr_args.transp_matrix = tr_buffer ;
//     tr_args.N = inp_ch;
//     tr_args.M = out_ch;
//     pi_cl_team_fork(NUM_CORES, transpose, &tr_args);

//     struct matMul_args matMul_args;
//     // COMPUTE ACTIV_GRAD
//     matMul_args.A = tr_buffer;
//     matMul_args.B = activations_pt;
//     matMul_args.C = output_pt;
//     matMul_args.N = out_ch;
//     matMul_args.M = out_height * out_width;
//     matMul_args.K = inp_ch;
//     matMul_args.trans_B = 0;

//     struct mm_manager_args man_args;
//     man_args.mm_args = &matMul_args;
//     man_args.layer_type = LAYER_PW_CONV;
//     man_args.step_type = STEP_IN_GRAD;
//     man_args.matmul_type = 20; //MATMUL_TYPE;
//     pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
// }