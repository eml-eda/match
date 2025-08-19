#include <pulp_cluster/odl_lib.h>

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

    #ifdef CLUSTER_LIB_DEBUG
    if (pi_core_id() == 0) {
        printf("Fast Conv2D BW FP32: Batches %d, Channels %d, Height %d, Width %d\n",
            out_batches, out_ch, out_height, out_width);
        printf("Input: C %d H %d W %d\n",
               inp_ch, inp_height, inp_width);
        printf("Start: B %d C %d H %d W %d, Stop: B %d C %d H %d W %d\n",
            start_batch, start_ch, start_h, start_w, stop_batch, stop_ch, stop_h, stop_w);
        printf("Parameters: Kernel %dx%d, Stride %dx%d, Dilation %dx%d, Padding %d,%d\n",
                kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left);
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
    
    #ifdef CLUSTER_LIB_DEBUG
    if (pi_core_id() == 0) {
        printf("Fast Conv2D BW FP32 Im2Col: Batches %d, Channels %d, Height %d, Width %d\n",
            out_batches, out_ch, out_height, out_width);
        printf("Input: C %d H %d W %d\n",
               inp_ch, inp_height, inp_width);
        printf("Parameters: Kernel %dx%d, Stride %dx%d, Dilation %dx%d, Padding %d,%d\n",
                kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left);
    }
    #endif

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