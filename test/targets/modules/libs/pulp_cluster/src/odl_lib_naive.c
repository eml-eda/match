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