#include <pulp_cluster/odl_lib.h>
#define ODL_MIN(a,b) ((a)<(b)?(a):(b))

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