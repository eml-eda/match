#include <pulp_cluster/odl_lib.h>

void odl_optimized_conv2d_bw_fp32_im2col(void* args) {
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;

    float* __restrict__ activations_pt = tensors[0].pt;
    float* __restrict__ parameters_pt = tensors[1].pt;
    float* __restrict__ output_pt = tensors[output_tensor_idx].pt;
    float* __restrict__ bias_pt = (num_tensors > 3) ? tensors[2].pt : NULL;
    float* __restrict__ im2col_pt = get_im2col_pt();

    int ch_idx = 1, height_idx = 2, width_idx = 3;
    if (conv_attrs->data_layout != "NCHW") {
        ch_idx = 3; height_idx = 1; width_idx = 2;
    }

    int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + 0].size;
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + width_idx].size;
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + height_idx].size;
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;

    int inp_width = tensors[0].tiles[L1_SCRATCHPAD * 4 + width_idx].size;
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD * 4 + height_idx].size;
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;

    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD * 4 + height_idx]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD * 4 + width_idx]));

    int stride_h = conv_attrs->strides[0];
    int stride_w = conv_attrs->strides[1];
    int dilation_h = conv_attrs->dilation[0];
    int dilation_w = conv_attrs->dilation[1];
    int kernel_h = conv_attrs->kernel_size[0];
    int kernel_w = conv_attrs->kernel_size[1];

    int im2col_rows = kernel_h * kernel_w * inp_ch;
    int im2col_cols = out_height * out_width;
    int total_c = out_ch;
    int block_size_c = (total_c + NUM_CORES - 1) / NUM_CORES;
    int c_start = pi_core_id() * block_size_c;
    int c_end = (c_start + block_size_c > total_c) ? total_c : c_start + block_size_c;
    
    for (int b = 0; b < out_batches; ++b) {
        int col_idx = 0;
        int total_cols = out_height * out_width;
        int col_per_core = (total_cols + NUM_CORES - 1) / NUM_CORES;
        int col_start = pi_core_id() * col_per_core;
        int col_end = ODL_MIN(col_start + col_per_core, total_cols);

        for (int col_idx = col_start; col_idx < col_end; ++col_idx) {
            int h_out = col_idx / out_width;
            int w_out = col_idx % out_width;

            int row_idx = 0;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int h_in = h_out * stride_h + kh * dilation_h - pad_top;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = w_out * stride_w + kw * dilation_w - pad_left;
                    for (int c = 0; c < inp_ch; ++c) {
                        float val = 0.0f;
                        if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                            int act_idx = w_in + inp_width * (h_in + inp_height * (c + inp_ch * b));
                            val = activations_pt[act_idx];
                        }
                        im2col_pt[row_idx * total_cols + col_idx] = val;
                        row_idx++;
                    }
                }
            }
        }

        pi_cl_team_barrier(0); // Ensure all cores have completed im2col before proceeding

        for (int c_out = c_start; c_out < c_end; ++c_out) {
            float bias_val = bias_pt ? bias_pt[c_out] : 0.0f;
            int base_out = c_out * out_height * out_width + b * out_ch * out_height * out_width;
            for (int i = 0; i < im2col_cols; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < im2col_rows; ++j) {
                    int weight_idx = c_out * im2col_rows + j;
                    acc += parameters_pt[weight_idx] * im2col_pt[j * im2col_cols + i];
                }
                output_pt[base_out + i] = acc + bias_val;
            }
        }
    }
}

void odl_fast_conv2d_bw_fp32_im2col(void* args) {
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;

    float* __restrict__ activations_pt = tensors[0].pt;
    float* __restrict__ parameters_pt = tensors[1].pt;
    float* __restrict__ output_pt = tensors[output_tensor_idx].pt;
    float* __restrict__ bias_pt = (num_tensors > 3) ? tensors[2].pt : NULL;
    float* __restrict__ im2col_pt = get_im2col_pt();

    int ch_idx = 1, height_idx = 2, width_idx = 3;
    if (conv_attrs->data_layout != "NCHW") {
        ch_idx = 3; height_idx = 1; width_idx = 2;
    }

    int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + 0].size;
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + width_idx].size;
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + height_idx].size;
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;

    int inp_width = tensors[0].tiles[L1_SCRATCHPAD * 4 + width_idx].size;
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD * 4 + height_idx].size;
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;

    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD * 4 + height_idx]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD * 4 + width_idx]));

    int stride_h = conv_attrs->strides[0];
    int stride_w = conv_attrs->strides[1];
    int dilation_h = conv_attrs->dilation[0];
    int dilation_w = conv_attrs->dilation[1];
    int kernel_h = conv_attrs->kernel_size[0];
    int kernel_w = conv_attrs->kernel_size[1];

    int im2col_rows = kernel_h * kernel_w * inp_ch;
    int im2col_cols = out_height * out_width;
    
    // Pre-compute for better performance  
    int inp_hw = inp_height * inp_width;
    int total_cols = out_height * out_width;
    int col_per_core = (total_cols + NUM_CORES - 1) / NUM_CORES;
    int col_start = pi_core_id() * col_per_core;
    int col_end = ODL_MIN(col_start + col_per_core, total_cols);
    
    for (int b = 0; b < out_batches; ++b) {
        // Simplified im2col construction - keep it close to original for stability
        for (int col_idx = col_start; col_idx < col_end; ++col_idx) {
            int h_out = col_idx / out_width;
            int w_out = col_idx % out_width;

            int row_idx = 0;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int h_in = h_out * stride_h + kh * dilation_h - pad_top;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = w_out * stride_w + kw * dilation_w - pad_left;
                    
                    // Simple bounds check
                    if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                        int base_act_idx = w_in + inp_width * h_in + inp_hw * b;
                        
                        // Sequential channel access for better cache locality
                        for (int c = 0; c < inp_ch; ++c) {
                            int act_idx = base_act_idx + c * inp_hw;
                            im2col_pt[row_idx * total_cols + col_idx] = activations_pt[act_idx];
                            row_idx++;
                        }
                    } else {
                        // Fill zeros for padding
                        for (int c = 0; c < inp_ch; ++c) {
                            im2col_pt[row_idx * total_cols + col_idx] = 0.0f;
                            row_idx++;
                        }
                    }
                }
            }
        }

        pi_cl_team_barrier(0);

        // Optimized matrix multiplication with modest unrolling
        int total_c = out_ch;
        int block_size_c = (total_c + NUM_CORES - 1) / NUM_CORES;
        int c_start = pi_core_id() * block_size_c;
        int c_end = (c_start + block_size_c > total_c) ? total_c : c_start + block_size_c;
        
        for (int c_out = c_start; c_out < c_end; ++c_out) {
            float bias_val = bias_pt ? bias_pt[c_out] : 0.0f;
            int base_out = c_out * out_height * out_width + b * out_ch * out_height * out_width;
            
            // Process 4 outputs at a time for better throughput
            int i = 0;
            for (; i < (im2col_cols & (~3)); i += 4) {
                register float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                
                // Inner loop with modest unrolling
                for (int j = 0; j < im2col_rows; ++j) {
                    int weight_idx = c_out * im2col_rows + j;
                    float weight = parameters_pt[weight_idx];
                    int im2col_base = j * im2col_cols + i;
                    
                    acc0 += weight * im2col_pt[im2col_base];
                    acc1 += weight * im2col_pt[im2col_base + 1];
                    acc2 += weight * im2col_pt[im2col_base + 2];
                    acc3 += weight * im2col_pt[im2col_base + 3];
                }
                
                output_pt[base_out + i] = acc0 + bias_val;
                output_pt[base_out + i + 1] = acc1 + bias_val;
                output_pt[base_out + i + 2] = acc2 + bias_val;
                output_pt[base_out + i + 3] = acc3 + bias_val;
            }
            
            // Handle remaining outputs
            for (; i < im2col_cols; ++i) {
                register float acc = 0.0f;
                for (int j = 0; j < im2col_rows; ++j) {
                    int weight_idx = c_out * im2col_rows + j;
                    acc += parameters_pt[weight_idx] * im2col_pt[j * im2col_cols + i];
                }
                output_pt[base_out + i] = acc + bias_val;
            }
        }
    }
}

void odl_optimized_conv2d_transpose_3x3_stride1_fp32_im2col(void* args) {
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1;
    MatchConv2DTransposeAttrs* conv_attrs = (MatchConv2DTransposeAttrs*)ctx->ops->ops[0].attrs;

    float* __restrict__ activations_pt = tensors[0].pt;
    float* __restrict__ parameters_pt = tensors[1].pt;
    float* __restrict__ output_pt = tensors[output_tensor_idx].pt;
    float* __restrict__ bias_pt = (num_tensors > 3) ? tensors[2].pt : NULL;
    float* __restrict__ im2col_pt = get_im2col_pt();

    int ch_idx = 1, height_idx = 2, width_idx = 3;
    
    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW") {
        ch_idx = 3; height_idx = 1; width_idx = 2;
    }
    #endif

    // Get dimensions
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + width_idx].size;
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + height_idx].size;
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;
    int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD * 4 + width_idx].size;
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD * 4 + height_idx].size;
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;

    // Get padding
    int idx_remainder_top_int_pad = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].idx_remainder!=0.0f?
        tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].idx_remainder>0.0f? -1: 1: 0; 
    int idx_remainder_left_int_pad = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].idx_remainder!=0.0f?
        tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].idx_remainder>0.0f? -1: 1: 0; 
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx])) + idx_remainder_top_int_pad;
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx])) + idx_remainder_left_int_pad;

    int is_dw = conv_attrs->depthwise;

    // Simple 3x3 stride=1 - no tiling complexity
    int im2col_rows = 9 * inp_ch;
    int im2col_cols = out_height * out_width;
    int inp_hw = inp_height * inp_width;
    
    // Simple core parallelization for im2col construction
    int col_per_core = (im2col_cols + NUM_CORES - 1) / NUM_CORES;
    int col_start = pi_core_id() * col_per_core;
    int col_end = (col_start + col_per_core > im2col_cols) ? im2col_cols : col_start + col_per_core;

    #ifdef ODL_DEBUG
    printf("col per core: %d, col start: %d, col end: %d\n", col_per_core, col_start, col_end);
    printf("Out[%d %d %d] Inp[%d %d %d] pad ^%d <%d\n",
        out_ch, out_height, out_width,
        inp_ch, inp_height, inp_width,
        pad_top, pad_left
    );
    #endif

    // Build single im2col buffer
    for (int col_idx = col_start; col_idx < col_end; ++col_idx) {
        int h_out = col_idx / out_width;
        int w_out = col_idx % out_width;

        int im2col_idx = 0;
        
        // Completely unrolled 3x3 stride=1 im2col
        for (int c = 0; c < inp_ch; ++c) {
            int c_base = c * inp_hw;
            
            // kh=0, kw=0
            int h_in = h_out - pad_top;
            int w_in = w_out - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
            
            // kh=0, kw=1
            w_in = w_out + 1 - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
            
            // kh=0, kw=2
            w_in = w_out + 2 - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
            
            // kh=1, kw=0
            h_in = h_out + 1 - pad_top;
            w_in = w_out - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
            
            // kh=1, kw=1
            w_in = w_out + 1 - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
            
            // kh=1, kw=2
            w_in = w_out + 2 - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
            
            // kh=2, kw=0
            h_in = h_out + 2 - pad_top;
            w_in = w_out - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
            
            // kh=2, kw=1
            w_in = w_out + 1 - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
            
            // kh=2, kw=2
            w_in = w_out + 2 - pad_left;
            im2col_pt[im2col_idx * im2col_cols + col_idx] = (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) ? 
                activations_pt[c_base + h_in * inp_width + w_in] : 0.0f;
            im2col_idx++;
        }
    }

    pi_cl_team_barrier(0);

    // Simple matrix multiplication - no tiling
    int ch_per_core = (out_ch + NUM_CORES - 1) / NUM_CORES;
    int ch_start = pi_core_id() * ch_per_core;
    int ch_end = (ch_start + ch_per_core > out_ch) ? out_ch : ch_start + ch_per_core;
    if(!is_dw) {
        for (int c_out_idx = ch_start; c_out_idx < ch_end; c_out_idx++) {
            float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            int c_weight_base_ = c_out_idx * 9;
            int out_idx_ch_base = c_out_idx * out_height * out_width;
            int col = 0;
            for (; col < (im2col_cols&(~3)); col+=4) {
                register float acc = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;
                int out_idx_base = out_idx_ch_base + col;
                for (int c_idx = 0; c_idx < inp_ch; c_idx+=9) {
                    int weight_idx = (c_idx * out_ch_params + c_weight_base_) + 8;
                    int im2col_pt_base = (c_idx * im2col_cols) + col;
                    float p_1 = parameters_pt[weight_idx];
                    // Completely unrolled 3x3 with flipped indices
                    acc += p_1 * im2col_pt[im2col_pt_base];
                    acc2 += p_1 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_1 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_1 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_2 = parameters_pt[weight_idx];
                    acc += p_2 * im2col_pt[im2col_pt_base];
                    acc2 += p_2 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_2 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_2 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_3 = parameters_pt[weight_idx];
                    acc += p_3 * im2col_pt[im2col_pt_base];
                    acc2 += p_3 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_3 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_3 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_4 = parameters_pt[weight_idx];
                    acc += p_4 * im2col_pt[im2col_pt_base];
                    acc2 += p_4 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_4 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_4 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_5 = parameters_pt[weight_idx];
                    acc += p_5 * im2col_pt[im2col_pt_base];
                    acc2 += p_5 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_5 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_5 * im2col_pt[im2col_pt_base + 3];
                    
                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_6 = parameters_pt[weight_idx];
                    acc += p_6 * im2col_pt[im2col_pt_base];
                    acc2 += p_6 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_6 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_6 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_7 = parameters_pt[weight_idx];
                    acc += p_7 * im2col_pt[im2col_pt_base];
                    acc2 += p_7 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_7 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_7 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_8 = parameters_pt[weight_idx];
                    acc += p_8 * im2col_pt[im2col_pt_base];
                    acc2 += p_8 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_8 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_8 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_9 = parameters_pt[weight_idx];
                    acc += p_9 * im2col_pt[im2col_pt_base];
                    acc2 += p_9 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_9 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_9 * im2col_pt[im2col_pt_base + 3];
                }
                
                output_pt[out_idx_base] = acc;
                output_pt[out_idx_base + 1] = acc2;
                output_pt[out_idx_base + 2] = acc3;
                output_pt[out_idx_base + 3] = acc4;
            }
            for (; col < im2col_cols; ++col) {
                register float acc = 0.0f;
                for (int c_idx = 0; c_idx < inp_ch; ++c_idx) {
                    int c_weight_base = c_idx * out_ch_params + c_weight_base_;
                    int im2col_pt_base = (c_idx * im2col_cols) + col;
                    // Completely unrolled 3x3 with flipped indices
                    acc += parameters_pt[c_weight_base + 8] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 7] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 6] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 5] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 4] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 3] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 2] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 1] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 0] * im2col_pt[im2col_pt_base];
                }
                
                output_pt[out_idx_ch_base + col] = acc;
            }
        }
    } else {
        for (int c_out_idx = ch_start; c_out_idx < ch_end; c_out_idx++) {
            int out_idx_base_ch = c_out_idx * out_height * out_width;
            int im2col_base = c_out_idx * 9;
            int col = 0;
            register float acc = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;
            for (; col < (im2col_cols&(~3)); col+=4) {
                float p_1 = parameters_pt[im2col_base + 8];
                int im2col_pt_base = (im2col_base * im2col_cols) + col;
                int out_idx_base = out_idx_base_ch + col;
                // Completely unrolled 3x3 with flipped indices
                acc += p_1 * im2col_pt[im2col_pt_base];
                acc2 += p_1 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_1 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_1 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_2 = parameters_pt[im2col_base + 7];
                acc += p_2 * im2col_pt[im2col_pt_base];
                acc2 += p_2 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_2 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_2 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_3 = parameters_pt[im2col_base + 6];
                acc += p_3 * im2col_pt[im2col_pt_base];
                acc2 += p_3 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_3 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_3 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_4 = parameters_pt[im2col_base + 5];
                acc += p_4 * im2col_pt[im2col_pt_base];
                acc2 += p_4 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_4 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_4 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_5 = parameters_pt[im2col_base + 4];
                acc += p_5 * im2col_pt[im2col_pt_base];
                acc2 += p_5 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_5 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_5 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_6 = parameters_pt[im2col_base + 3];
                acc += p_6 * im2col_pt[im2col_pt_base];
                acc2 += p_6 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_6 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_6 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_7 = parameters_pt[im2col_base + 2];
                acc += p_7 * im2col_pt[im2col_pt_base];
                acc2 += p_7 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_7 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_7 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_8 = parameters_pt[im2col_base + 1];
                acc += p_8 * im2col_pt[im2col_pt_base];
                acc2 += p_8 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_8 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_8 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_9 = parameters_pt[im2col_base];
                acc += p_9 * im2col_pt[im2col_pt_base];
                acc2 += p_9 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_9 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_9 * im2col_pt[im2col_pt_base + 3];
                output_pt[out_idx_base] = acc;
                output_pt[out_idx_base + 1] = acc2;
                output_pt[out_idx_base + 2] = acc3;
                output_pt[out_idx_base + 3] = acc4;
            }
            for (; col < im2col_cols; ++col) {
                acc = 0.0f;
                int im2col_pt_base = (im2col_base * im2col_cols) + col;
                acc += parameters_pt[im2col_base + 8] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 7] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 6] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 5] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 4] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 3] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 2] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 1] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base] * im2col_pt[im2col_pt_base];
                output_pt[out_idx_base_ch + col] = acc;
            }
        }
    }
}

void odl_optimized_conv2d_transpose_3x3_stride2_fp32_im2col(void* args) {
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1;
    MatchConv2DTransposeAttrs* conv_attrs = (MatchConv2DTransposeAttrs*)ctx->ops->ops[0].attrs;

    float* __restrict__ activations_pt = tensors[0].pt;
    float* __restrict__ parameters_pt = tensors[1].pt;
    float* __restrict__ output_pt = tensors[output_tensor_idx].pt;
    float* __restrict__ bias_pt = (num_tensors > 3) ? tensors[2].pt : NULL;
    float* __restrict__ im2col_pt = get_im2col_pt();

    int ch_idx = 1, height_idx = 2, width_idx = 3;
    
    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW") {
        ch_idx = 3; height_idx = 1; width_idx = 2;
    }
    #endif

    // Get dimensions
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + width_idx].size;
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + height_idx].size;
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;
    int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD * 4 + width_idx].size;
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD * 4 + height_idx].size;
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD * 4 + ch_idx].size;

    // Get padding
    int idx_remainder_top_int_pad = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].idx_remainder!=0.0f?
        tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].idx_remainder>0.0f? -1: 1: 0; 
    int idx_remainder_left_int_pad = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].idx_remainder!=0.0f?
        tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].idx_remainder>0.0f? -1: 1: 0; 
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx])) + idx_remainder_top_int_pad;
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx])) + idx_remainder_left_int_pad;

    int is_dw = conv_attrs->depthwise;

    // Simple 3x3 stride=2 - no tiling complexity
    int im2col_rows = 9 * inp_ch;
    int im2col_cols = out_height * out_width;
    int inp_hw = inp_height * inp_width;
    
    // Simple core parallelization for im2col construction
    int col_per_core = (im2col_cols + NUM_CORES - 1) / NUM_CORES;
    int col_start = pi_core_id() * col_per_core;
    int col_end = (col_start + col_per_core > im2col_cols) ? im2col_cols : col_start + col_per_core;

    // Build single im2col buffer
    for (int col_idx = col_start; col_idx < col_end; ++col_idx) {
        int h_out = col_idx / out_width;
        int w_out = col_idx % out_width;

        int im2col_idx = 0;
        
        // Completely unrolled 3x3 stride=2 im2col
        for (int c = 0; c < inp_ch; ++c) {
            int c_base = c * inp_hw;
            
            // kh=0, kw=0
            int h_in_base = h_out - pad_top;
            int w_in_base = w_out - pad_left;
            float val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) { // Check if even (divisible by 2)
                int h_in = h_in_base >> 1; // Divide by 2 using bit shift
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
            
            // kh=0, kw=1
            w_in_base = w_out + 1 - pad_left;
            val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) {
                int h_in = h_in_base >> 1;
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
            
            // kh=0, kw=2
            w_in_base = w_out + 2 - pad_left;
            val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) {
                int h_in = h_in_base >> 1;
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
            
            // kh=1, kw=0
            h_in_base = h_out + 1 - pad_top;
            w_in_base = w_out - pad_left;
            val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) {
                int h_in = h_in_base >> 1;
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
            
            // kh=1, kw=1
            w_in_base = w_out + 1 - pad_left;
            val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) {
                int h_in = h_in_base >> 1;
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
            
            // kh=1, kw=2
            w_in_base = w_out + 2 - pad_left;
            val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) {
                int h_in = h_in_base >> 1;
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
            
            // kh=2, kw=0
            h_in_base = h_out + 2 - pad_top;
            w_in_base = w_out - pad_left;
            val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) {
                int h_in = h_in_base >> 1;
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
            
            // kh=2, kw=1
            w_in_base = w_out + 1 - pad_left;
            val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) {
                int h_in = h_in_base >> 1;
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
            
            // kh=2, kw=2
            w_in_base = w_out + 2 - pad_left;
            val = 0.0f;
            if (!(h_in_base & 1) && !(w_in_base & 1)) {
                int h_in = h_in_base >> 1;
                int w_in = w_in_base >> 1;
                if (h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width) {
                    val = activations_pt[c_base + h_in * inp_width + w_in];
                }
            }
            im2col_pt[im2col_idx * im2col_cols + col_idx] = val;
            im2col_idx++;
        }
    }

    pi_cl_team_barrier(0);

    // Simple matrix multiplication - no tiling
    int ch_per_core = (out_ch + NUM_CORES - 1) / NUM_CORES;
    int ch_start = pi_core_id() * ch_per_core;
    int ch_end = (ch_start + ch_per_core > out_ch) ? out_ch : ch_start + ch_per_core;

    if(!is_dw) {
        for (int c_out_idx = ch_start; c_out_idx < ch_end; c_out_idx++) {
            float bias_val = bias_pt ? bias_pt[c_out_idx] : 0.0f;
            int c_weight_base_ = c_out_idx * 9;
            int out_idx_ch_base = c_out_idx * out_height * out_width;
            int col = 0;
            for (; col < (im2col_cols&(~3)); col+=4) {
                register float acc = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;
                int out_idx_base = out_idx_ch_base + col;
                for (int c_idx = 0; c_idx < inp_ch; c_idx+=9) {
                    int weight_idx = (c_idx * out_ch_params + c_weight_base_) + 8;
                    int im2col_pt_base = (c_idx * im2col_cols) + col;
                    float p_1 = parameters_pt[weight_idx];
                    // Completely unrolled 3x3 with flipped indices
                    acc += p_1 * im2col_pt[im2col_pt_base];
                    acc2 += p_1 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_1 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_1 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_2 = parameters_pt[weight_idx];
                    acc += p_2 * im2col_pt[im2col_pt_base];
                    acc2 += p_2 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_2 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_2 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_3 = parameters_pt[weight_idx];
                    acc += p_3 * im2col_pt[im2col_pt_base];
                    acc2 += p_3 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_3 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_3 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_4 = parameters_pt[weight_idx];
                    acc += p_4 * im2col_pt[im2col_pt_base];
                    acc2 += p_4 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_4 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_4 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_5 = parameters_pt[weight_idx];
                    acc += p_5 * im2col_pt[im2col_pt_base];
                    acc2 += p_5 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_5 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_5 * im2col_pt[im2col_pt_base + 3];
                    
                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_6 = parameters_pt[weight_idx];
                    acc += p_6 * im2col_pt[im2col_pt_base];
                    acc2 += p_6 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_6 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_6 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_7 = parameters_pt[weight_idx];
                    acc += p_7 * im2col_pt[im2col_pt_base];
                    acc2 += p_7 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_7 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_7 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_8 = parameters_pt[weight_idx];
                    acc += p_8 * im2col_pt[im2col_pt_base];
                    acc2 += p_8 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_8 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_8 * im2col_pt[im2col_pt_base + 3];

                    im2col_pt_base += im2col_cols;
                    weight_idx--;

                    float p_9 = parameters_pt[weight_idx];
                    acc += p_9 * im2col_pt[im2col_pt_base];
                    acc2 += p_9 * im2col_pt[im2col_pt_base + 1];
                    acc3 += p_9 * im2col_pt[im2col_pt_base + 2];
                    acc4 += p_9 * im2col_pt[im2col_pt_base + 3];
                }
                
                output_pt[out_idx_base] = acc;
                output_pt[out_idx_base + 1] = acc2;
                output_pt[out_idx_base + 2] = acc3;
                output_pt[out_idx_base + 3] = acc4;
            }
            for (; col < im2col_cols; ++col) {
                register float acc = 0.0f;
                for (int c_idx = 0; c_idx < inp_ch; ++c_idx) {
                    int c_weight_base = c_idx * out_ch_params + c_weight_base_;
                    int im2col_pt_base = (c_idx * im2col_cols) + col;
                    // Completely unrolled 3x3 with flipped indices
                    acc += parameters_pt[c_weight_base + 8] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 7] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 6] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 5] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 4] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 3] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 2] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 1] * im2col_pt[im2col_pt_base];
                    im2col_pt_base += im2col_cols;
                    acc += parameters_pt[c_weight_base + 0] * im2col_pt[im2col_pt_base];
                }
                
                output_pt[out_idx_ch_base + col] = acc;
            }
        }
    } else {
        for (int c_out_idx = ch_start; c_out_idx < ch_end; c_out_idx++) {
            int out_idx_base_ch = c_out_idx * out_height * out_width;
            int im2col_base = c_out_idx * 9;
            int col = 0;
            register float acc = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;
            for (; col < (im2col_cols&(~3)); col+=4) {
                float p_1 = parameters_pt[im2col_base + 8];
                int im2col_pt_base = (im2col_base * im2col_cols) + col;
                int out_idx_base = out_idx_base_ch + col;
                // Completely unrolled 3x3 with flipped indices
                acc += p_1 * im2col_pt[im2col_pt_base];
                acc2 += p_1 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_1 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_1 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_2 = parameters_pt[im2col_base + 7];
                acc += p_2 * im2col_pt[im2col_pt_base];
                acc2 += p_2 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_2 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_2 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_3 = parameters_pt[im2col_base + 6];
                acc += p_3 * im2col_pt[im2col_pt_base];
                acc2 += p_3 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_3 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_3 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_4 = parameters_pt[im2col_base + 5];
                acc += p_4 * im2col_pt[im2col_pt_base];
                acc2 += p_4 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_4 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_4 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_5 = parameters_pt[im2col_base + 4];
                acc += p_5 * im2col_pt[im2col_pt_base];
                acc2 += p_5 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_5 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_5 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_6 = parameters_pt[im2col_base + 3];
                acc += p_6 * im2col_pt[im2col_pt_base];
                acc2 += p_6 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_6 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_6 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_7 = parameters_pt[im2col_base + 2];
                acc += p_7 * im2col_pt[im2col_pt_base];
                acc2 += p_7 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_7 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_7 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_8 = parameters_pt[im2col_base + 1];
                acc += p_8 * im2col_pt[im2col_pt_base];
                acc2 += p_8 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_8 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_8 * im2col_pt[im2col_pt_base + 3];
                im2col_pt_base += im2col_cols;
                float p_9 = parameters_pt[im2col_base];
                acc += p_9 * im2col_pt[im2col_pt_base];
                acc2 += p_9 * im2col_pt[im2col_pt_base + 1];
                acc3 += p_9 * im2col_pt[im2col_pt_base + 2];
                acc4 += p_9 * im2col_pt[im2col_pt_base + 3];
                output_pt[out_idx_base] = acc;
                output_pt[out_idx_base + 1] = acc2;
                output_pt[out_idx_base + 2] = acc3;
                output_pt[out_idx_base + 3] = acc4;
            }
            for (; col < im2col_cols; ++col) {
                acc = 0.0f;
                int im2col_pt_base = (im2col_base * im2col_cols) + col;
                acc += parameters_pt[im2col_base + 8] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 7] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 6] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 5] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 4] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 3] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 2] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base + 1] * im2col_pt[im2col_pt_base];
                im2col_pt_base += im2col_cols;
                acc += parameters_pt[im2col_base] * im2col_pt[im2col_pt_base];
                output_pt[out_idx_base_ch + col] = acc;
            }
        }
    }
}