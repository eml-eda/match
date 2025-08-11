#include <pulp_cluster/odl_lib.h>

void odl_naive_parallel_conv2d_transpose_fp32(void* args){
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
    int stride_h = conv_attrs->strides[0];
    int stride_w = conv_attrs->strides[1];
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
        stride_h, stride_w, dilation_h, dilation_w
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

// right now used only for dw conv2d bw, otherwise it is not optimized enough and host is faster
void odl_naive_parallel_conv2d_bw_fp32(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1; // output tensor is always the last one
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    
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
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int stride_h = conv_attrs->strides[0];
    int stride_w = conv_attrs->strides[1];
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
        stride_h, stride_w, dilation_h, dilation_w
    );
    printf("Num tensors: %d\n", num_tensors);
    #endif
    uint32_t start_bc = 0, stop_bc = out_batches * out_ch;
    uint32_t start_h_out = 0, stop_h_out = out_height;
    uint32_t start_w_out = 0, stop_w_out = out_width;
    #if NUM_CORES > 1
    int prefer_blocking_over = stop_bc >= out_height && stop_bc >= out_width ? 0 : out_height >= out_width ? 1 : 2;
    if(!prefer_blocking_over) {
        int block_size_bc = (stop_bc+NUM_CORES-1) / NUM_CORES;
        start_bc = pi_core_id() * block_size_bc;
        stop_bc = start_bc + block_size_bc > stop_bc ? stop_bc : start_bc + block_size_bc;
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
        for (int bc_idx = start_bc; bc_idx < stop_bc; ++bc_idx) {
            int b_out_idx = bc_idx / out_ch;
            int c_out_idx = bc_idx % out_ch;
            int params_idx_base = c_out_idx * inp_ch * kernel_h * kernel_w;
            int b_c_offset = b_out_idx * out_ch * out_height * out_width + c_out_idx * out_height * out_width;
            float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            for (uint32_t h_out_idx=start_h_out; h_out_idx<stop_h_out; h_out_idx++) {
                int h_out_offset = h_out_idx * out_width;
                int h_act_idx_base = h_out_idx * stride_h - pad_top;
                for (uint32_t w_out_idx=start_w_out; w_out_idx<stop_w_out; w_out_idx++) {
                    float tmp = 0.0f;
                    int w_act_idx_base = w_out_idx * stride_w - pad_left;
                    int out_idx = w_out_idx + h_out_offset + b_c_offset;
                    for (uint32_t h_ker_idx=0; h_ker_idx<kernel_h; h_ker_idx++) {
                        int h_act_idx = h_act_idx_base + h_ker_idx * dilation_h;
                        if ((h_act_idx < 0) || (h_act_idx >= inp_height))
                            continue;
                        for (uint32_t w_ker_idx=0; w_ker_idx<kernel_w; w_ker_idx++) {
                            int w_act_idx = w_act_idx_base + w_ker_idx * dilation_w;
                            // check if the indices are within the bounds of the input
                            if ((w_act_idx < 0) || (w_act_idx >= inp_width))
                                continue;
                            #ifdef ODL_DEBUG_KERNEL
                            if(c_out_idx == 0)
                                printf("h_out_idx: %d, w_out_idx: %d, h_ker_idx: %d, w_ker_idx: %d, h_act_idx: %d, w_act_idx: %d\n", 
                                        h_out_idx, w_out_idx, h_ker_idx, w_ker_idx, h_act_idx, w_act_idx);
                            #endif
                            for (uint32_t c_acts_idx=0; c_acts_idx<inp_ch; c_acts_idx++){
                                int params_idx = params_idx_base + c_acts_idx * kernel_w * kernel_h + h_ker_idx * kernel_w + w_ker_idx;
                                int act_idx = w_act_idx + inp_width * (h_act_idx + inp_height * (c_acts_idx + inp_ch * b_out_idx));
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
        for (int bc_idx = start_bc; bc_idx < stop_bc; ++bc_idx) {
            int b_out_idx = bc_idx / out_ch;
            int c_out_idx = bc_idx % out_ch;
            int params_idx_base = c_out_idx * inp_ch * kernel_h * kernel_w;
            int b_c_offset = b_out_idx * out_ch * out_height * out_width + c_out_idx * out_height * out_width;
            float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            for (uint32_t h_out_idx=start_h_out; h_out_idx<stop_h_out; h_out_idx++) {
                int h_out_offset = h_out_idx * out_width;
                int h_act_idx_base = h_out_idx * stride_h - pad_top;
                for (uint32_t w_out_idx=start_w_out; w_out_idx<stop_w_out; w_out_idx++) {
                    float tmp = 0.0f;
                    int w_act_idx_base = w_out_idx * stride_w - pad_left;
                    int out_idx = w_out_idx + h_out_offset + b_c_offset;
                    for (uint32_t h_ker_idx=0; h_ker_idx<kernel_h; h_ker_idx++) {
                        int h_act_idx = h_act_idx_base + h_ker_idx * dilation_h;
                        if ((h_act_idx < 0) || (h_act_idx >= inp_height))
                            continue;
                        for (uint32_t w_ker_idx=0; w_ker_idx<kernel_w; w_ker_idx++) {
                            int w_act_idx = w_act_idx_base + w_ker_idx * dilation_w;
                            // check if the indices are within the bounds of the input
                            if ((w_act_idx < 0) || (w_act_idx >= inp_width))
                                continue;
                            #ifdef ODL_DEBUG_KERNEL
                            if(c_out_idx == 0)
                                printf("h_out_idx: %d, w_out_idx: %d, h_ker_idx: %d, w_ker_idx: %d, h_act_idx: %d, w_act_idx: %d\n", 
                                        h_out_idx, w_out_idx, h_ker_idx, w_ker_idx, h_act_idx, w_act_idx);
                            #endif
                            int params_idx = c_out_idx*kernel_h*kernel_w + h_ker_idx*kernel_w + w_ker_idx;
                            int act_idx = w_act_idx + inp_width * (h_act_idx + inp_height * (c_out_idx + inp_ch * b_out_idx));
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

// Highly optimized standard convolution backward kernel 
void odl_fast_parallel_conv2d_bw_fp32(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    
    float * __restrict__ activations_pt = tensors[0].pt;
    float * __restrict__ parameters_pt = tensors[1].pt;
    float * __restrict__ output_pt = tensors[num_tensors-1].pt;
    float * __restrict__ bias_pt = NULL;
    if (num_tensors > 3) bias_pt = tensors[2].pt;

    int ch_idx = 1, height_idx = 2, width_idx = 3;
    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW"){
        ch_idx = 3; height_idx = 1; width_idx = 2;
    }
    #endif

    // Extract dimensions - using registers for frequently accessed values
    register const int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size;
    register const int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size;
    register const int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size;
    register const int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size;
    register const int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size;
    register const int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size;
    register const int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size;
    
    register const int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    register const int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    register const int stride_h = conv_attrs->strides[0];
    register const int stride_w = conv_attrs->strides[1];
    register const int dilation_h = conv_attrs->dilation[0];
    register const int dilation_w = conv_attrs->dilation[1];
    register const int kernel_h = conv_attrs->kernel_size[0];
    register const int kernel_w = conv_attrs->kernel_size[1];

    // Pre-compute all loop invariants
    const int out_spatial_size = out_height * out_width;
    const int inp_spatial_size = inp_height * inp_width;
    const int kernel_size = kernel_h * kernel_w;
    const int out_ch_spatial = out_ch * out_spatial_size;
    const int inp_ch_spatial = inp_ch * inp_spatial_size;
    const int kernel_ch_size = kernel_size * inp_ch;
    
    // Optimized work distribution - prefer spatial distribution for better cache locality
    int start_batch = 0, stop_batch = out_batches;
    int start_ch = 0, stop_ch = out_ch;
    int start_h = 0, stop_h = out_height;
    int start_w = 0, stop_w = out_width;
    
    #if NUM_CORES > 1
    // Choose dimension with most work for distribution
    const int total_spatial = out_height * out_width;
    const int total_channels = out_ch;
    const int total_batches = out_batches;
    
    if (total_spatial >= total_channels && total_spatial >= total_batches) {
        // Distribute across spatial dimensions
        if (out_height >= out_width) {
            int block_h = (out_height + NUM_CORES - 1) / NUM_CORES;
            start_h = pi_core_id() * block_h;
            stop_h = (start_h + block_h > out_height) ? out_height : start_h + block_h;
        } else {
            int block_w = (out_width + NUM_CORES - 1) / NUM_CORES;
            start_w = pi_core_id() * block_w;
            stop_w = (start_w + block_w > out_width) ? out_width : start_w + block_w;
        }
    } else if (total_channels >= total_batches) {
        // Distribute across channels
        int block_ch = (out_ch + NUM_CORES - 1) / NUM_CORES;
        start_ch = pi_core_id() * block_ch;
        stop_ch = (start_ch + block_ch > out_ch) ? out_ch : start_ch + block_ch;
    } else {
        // Distribute across batches
        int block_batch = (out_batches + NUM_CORES - 1) / NUM_CORES;
        start_batch = pi_core_id() * block_batch;
        stop_batch = (start_batch + block_batch > out_batches) ? out_batches : start_batch + block_batch;
    }
    #endif

    // Main computation with optimized loop ordering and memory access
    for (int b_idx = start_batch; b_idx < stop_batch; b_idx++) {
        const int batch_out_base = b_idx * out_ch_spatial;
        const int batch_inp_base = b_idx * inp_ch_spatial;
        
        for (int c_out_idx = start_ch; c_out_idx < stop_ch; c_out_idx++) {
            const float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            const int c_out_base = batch_out_base + c_out_idx * out_spatial_size;
            const int c_param_base = c_out_idx * kernel_ch_size;
            
            // Process output in blocks for better cache utilization
            for (int h_out_idx = start_h; h_out_idx < stop_h; h_out_idx++) {
                const int h_out_base = c_out_base + h_out_idx * out_width;
                const int h_inp_base = h_out_idx * stride_h - pad_top;
                
                // Inner loop optimization - process multiple pixels when possible
                int w_out_idx = start_w;
                for (; w_out_idx < stop_w; w_out_idx++) {
                    register float acc = 0.0f;
                    const int out_idx = h_out_base + w_out_idx;
                    const int w_inp_base = w_out_idx * stride_w - pad_left;
                    
                    // Reorganized loops for better memory access pattern
                    for (int c_inp_idx = 0; c_inp_idx < inp_ch; c_inp_idx++) {
                        const int c_inp_base = batch_inp_base + c_inp_idx * inp_spatial_size;
                        const int c_param_offset = c_param_base + c_inp_idx * kernel_size;
                        
                        for (int h_ker_idx = 0; h_ker_idx < kernel_h; h_ker_idx++) {
                            const int h_inp_idx = h_inp_base + h_ker_idx * dilation_h;
                            if (h_inp_idx < 0 || h_inp_idx >= inp_height) continue;
                            
                            const int h_inp_offset = c_inp_base + h_inp_idx * inp_width;
                            const int h_param_offset = c_param_offset + h_ker_idx * kernel_w;
                            
                            // Unroll small kernels for better performance
                            if (kernel_w == 3 && dilation_w == 1) {
                                // Unrolled 3x3 kernel
                                int w_inp_idx0 = w_inp_base;
                                int w_inp_idx1 = w_inp_base + 1;
                                int w_inp_idx2 = w_inp_base + 2;
                                
                                if (w_inp_idx0 >= 0 && w_inp_idx0 < inp_width) {
                                    acc += parameters_pt[h_param_offset] * activations_pt[h_inp_offset + w_inp_idx0];
                                }
                                if (w_inp_idx1 >= 0 && w_inp_idx1 < inp_width) {
                                    acc += parameters_pt[h_param_offset + 1] * activations_pt[h_inp_offset + w_inp_idx1];
                                }
                                if (w_inp_idx2 >= 0 && w_inp_idx2 < inp_width) {
                                    acc += parameters_pt[h_param_offset + 2] * activations_pt[h_inp_offset + w_inp_idx2];
                                }
                            } else {
                                // General case
                                for (int w_ker_idx = 0; w_ker_idx < kernel_w; w_ker_idx++) {
                                    const int w_inp_idx = w_inp_base + w_ker_idx * dilation_w;
                                    if (w_inp_idx < 0 || w_inp_idx >= inp_width) continue;
                                    
                                    acc += parameters_pt[h_param_offset + w_ker_idx] * 
                                           activations_pt[h_inp_offset + w_inp_idx];
                                }
                            }
                        }
                    }
                    
                    output_pt[out_idx] = acc + bias_val;
                }
            }
        }
    }
}

// Optimized depthwise convolution backward kernel for batched processing
void odl_optimized_parallel_conv2d_bw_dw_fp32(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    
    float * __restrict__ activations_pt = tensors[0].pt;
    float * __restrict__ parameters_pt = tensors[1].pt;
    float * __restrict__ output_pt = tensors[num_tensors-1].pt;
    float * __restrict__ bias_pt = NULL;
    if (num_tensors > 3) bias_pt = tensors[2].pt;

    int ch_idx = 1; // channel index
    int height_idx = 2; // height index
    int width_idx = 3; // width index

    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW"){
        ch_idx = 3; // channel index
        height_idx = 1; // height index
        width_idx = 2; // width index
    }
    #endif

    // Extract dimensions
    const int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size;
    const int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size;
    const int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size;
    const int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size;
    const int inp_batches = tensors[0].tiles[L1_SCRATCHPAD*4+0].size;
    const int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size;
    const int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size;
    const int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size;
    
    const int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    const int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    const int stride_h = conv_attrs->strides[0];
    const int stride_w = conv_attrs->strides[1];
    const int dilation_h = conv_attrs->dilation[0];
    const int dilation_w = conv_attrs->dilation[1];
    const int kernel_h = conv_attrs->kernel_size[0];
    const int kernel_w = conv_attrs->kernel_size[1];

    // Pre-compute constants
    const int out_spatial_size = out_height * out_width;
    const int inp_spatial_size = inp_height * inp_width;
    const int kernel_size = kernel_h * kernel_w;
    const int out_ch_spatial = out_ch * out_spatial_size;
    const int inp_ch_spatial = inp_ch * inp_spatial_size;

    // Work distribution across cores - distribute across batches and channels
    const int total_work = out_batches * out_ch;
    const int block_size = (total_work + NUM_CORES - 1) / NUM_CORES;
    const int start_work = pi_core_id() * block_size;
    const int stop_work = (start_work + block_size > total_work) ? total_work : start_work + block_size;

    // Main computation loop - optimized for depthwise with batching
    for (int work_idx = start_work; work_idx < stop_work; work_idx++) {
        const int b_idx = work_idx / out_ch;
        const int c_idx = work_idx % out_ch;
        
        const float bias_val = bias_pt ? bias_pt[c_idx] : 0.0f;
        const int batch_out_offset = b_idx * out_ch_spatial;
        const int batch_inp_offset = b_idx * inp_ch_spatial;
        const int c_out_offset = batch_out_offset + c_idx * out_spatial_size;
        const int c_inp_offset = batch_inp_offset + c_idx * inp_spatial_size;
        const int c_param_offset = c_idx * kernel_size;
        
        // For each output spatial position
        for (int h_out_idx = 0; h_out_idx < out_height; h_out_idx++) {
            const int h_out_offset = h_out_idx * out_width;
            const int h_inp_base = h_out_idx * stride_h - pad_top;
            
            for (int w_out_idx = 0; w_out_idx < out_width; w_out_idx++) {
                float tmp = 0.0f;
                const int out_idx = c_out_offset + h_out_offset + w_out_idx;
                const int w_inp_base = w_out_idx * stride_w - pad_left;
                
                // Kernel convolution - optimized inner loops
                for (int h_ker_idx = 0; h_ker_idx < kernel_h; h_ker_idx++) {
                    const int h_inp_idx = h_inp_base + h_ker_idx * dilation_h;
                    
                    // Bounds check for height
                    if (h_inp_idx < 0 || h_inp_idx >= inp_height) continue;
                    
                    const int h_param_offset = h_ker_idx * kernel_w;
                    const int h_inp_offset = c_inp_offset + h_inp_idx * inp_width;
                    
                    for (int w_ker_idx = 0; w_ker_idx < kernel_w; w_ker_idx++) {
                        const int w_inp_idx = w_inp_base + w_ker_idx * dilation_w;
                        
                        // Bounds check for width
                        if (w_inp_idx < 0 || w_inp_idx >= inp_width) continue;
                        
                        // Calculate indices using pre-computed offsets
                        const int params_idx = c_param_offset + h_param_offset + w_ker_idx;
                        const int act_idx = h_inp_offset + w_inp_idx;
                        
                        tmp += parameters_pt[params_idx] * activations_pt[act_idx];
                    }
                }
                
                output_pt[out_idx] = tmp + bias_val;
            }
        }
    }
}