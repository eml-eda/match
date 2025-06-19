#ifdef __pulp_cluster__

#include <stdint.h>
#include <stddef.h>

#include "pulp_nn_fp16/pulp_nn_fp16_defines.h"
#include "pulp_nn_fp16/pulp_nn_fp16_kernels.h"



static void pulp_nn_fp16_conv2d_nhwc_ohwi_par_oc_naive(
    const fp16   *input,      // NHWC
    const fp16   *weight,     // OHWI
    const fp16   *bias,       // O
    fp16         *output,     // NHWC
    fp16         *im2col,     // im2col buffer (not used)
    uint32_t      dim_ix,     // input width
    uint32_t      dim_iy,     // input height
    uint32_t      dim_ic,     // input channels
    uint32_t      dim_ox,     // output width
    uint32_t      dim_oy,     // output height
    uint32_t      dim_oc,     // output channels
    uint32_t      dim_fx,     // kernel width
    uint32_t      dim_fy,     // kernel height
    uint32_t      pad_t,
    uint32_t      pad_b,
    uint32_t      pad_l,
    uint32_t      pad_r,
    uint32_t      stride_x,
    uint32_t      stride_y,
    uint32_t      apply_relu
) {
    // Parallelize on output channel (dim_oc)
    int oc_chunk = (dim_oc + nthreads - 1) / nthreads;
    int oc_start = min(tid * oc_chunk, dim_oc);
    int oc_end = min(oc_start + oc_chunk, dim_oc);

    for (int oy = 0; oy < dim_oy; ++oy) {   // output height
        for (int ox = 0; ox < dim_ox; ++ox) { // output width
            for (int oc = oc_start; oc < oc_end; ++oc) { // output channel
                // Add Bias
                fp16 dot = (bias != NULL) ? bias[oc] : (fp16)0.0f;
                // For each kernel position
                for (int ic = 0; ic < dim_ic; ++ic) { // input channel
                    for (int fy = 0; fy < dim_fy; ++fy) { // filter height
                        for (int fx = 0; fx < dim_fx; ++fx) { // filter width
                            int iy = oy * stride_y + fy - pad_t;
                            int ix = ox * stride_x + fx - pad_l;

                            if (iy >= 0 && iy < dim_iy && ix >= 0 && ix < dim_ix) {
                                int input_idx  = idx_NHWC(iy, ix, ic, dim_iy, dim_ix, dim_ic);
                                int weight_idx = idx_OHWI(fy, fx, ic, oc, dim_fy, dim_fx, dim_ic, dim_oc);
                                dot += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                // Apply ReLU
                if (apply_relu && dot < (fp16)0.0f) {
                    dot = 0.0f;
                }
                int output_idx = idx_NHWC(oy, ox, oc, dim_oy, dim_ox, dim_oc);
                output[output_idx] = dot;
            }
        }
    }
}


void pulp_nn_fp16_conv2d(
    const fp16 *__restrict__ input,         // Pointer to the input feature map
    const fp16 *__restrict__ weight,        // Pointer to the weights
    const fp16 *__restrict__ bias,          // Pointer to the bias vector
    fp16 *__restrict__       output,        // Pointer to the output feature map
    fp16 *__restrict__       im2col,        // Pointer to the im2col buffer
    uint32_t                 dim_ix,        // Input Width
    uint32_t                 dim_iy,        // Input Height
    uint32_t                 dim_ic,        // Input Channels
    uint32_t                 dim_ox,        // Output Width
    uint32_t                 dim_oy,        // Output Height
    uint32_t                 dim_oc,        // Output Channels
    uint32_t                 dim_fx,        // Kernel Width
    uint32_t                 dim_fy,        // Kernel Height 
    uint32_t                 pad_t,         // Padding Top
    uint32_t                 pad_b,         // Padding Bottom
    uint32_t                 pad_l,         // Padding Left
    uint32_t                 pad_r,         // Padding Right
    uint32_t                 stride_x,      // Stride Horizontal
    uint32_t                 stride_y,      // Stride Vertical
    uint32_t                 apply_relu     // Apply ReLU activation
) {
    pulp_nn_fp16_conv2d_nhwc_ohwi_par_oc_naive(
        input, weight, bias, output, im2col,
        dim_ix, dim_iy, dim_ic,
        dim_ox, dim_oy, dim_oc,
        dim_fx, dim_fy,
        pad_t, pad_b, pad_l, pad_r,
        stride_x, stride_y, 
        apply_relu
    );
}


/*

static void pulp_nn_fp16_conv2d_par_c(
    const fp16 *__restrict__ input,
    fp16 *__restrict__       im2col,
    const fp16 *__restrict__ bias,
    fp16 *__restrict__       output,
    const fp16 *__restrict__ weight,
    const uint16_t           dim_w,
    const uint16_t           dim_h,
    const uint16_t           dim_c,
    const uint16_t           dim_p,
    const uint16_t           dim_q,
    const uint16_t           dim_k,
    const uint16_t           dim_r,
    const uint16_t           dim_s,
    const uint16_t           pad_t,
    const uint16_t           pad_b,
    const uint16_t           pad_l,
    const uint16_t           pad_r,
    const uint16_t           stride_x,
    const uint16_t           stride_y,
    int                      apply_relu
)
{
    // Parallelize on output channel (dim_k)
    int k_chunk = (dim_k + nthreads - 1) / nthreads;
    int k_start = min(tid * k_chunk, dim_k);
    int k_end = min(k_start + k_chunk, dim_k);

    for (int k = k_start; k < k_end; ++k) { // output channel
        for (int q = 0; q < dim_q; ++q) {   // output height
            for (int p = 0; p < dim_p; ++p) { // output width
                float acc = (bias != NULL) ? bias[k] : (fp16)0.0;

                // For each kernel position
                for (int c = 0; c < dim_c; ++c) { // input channel
                    for (int r = 0; r < dim_s; ++r) { // kernel height
                        for (int s = 0; s < dim_r; ++s) { // kernel width
                            int in_y = q * stride_y + r - pad_t;
                            int in_x = p * stride_x + s - pad_l;

                            if (in_y >= 0 && in_y < dim_h && in_x >= 0 && in_x < dim_w) {
                                size_t input_idx  = (c * dim_h + in_y) * dim_w + in_x;
                                size_t weight_idx = ((k * dim_c + c) * dim_s + r) * dim_r + s;
                                acc += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }

                if (apply_relu && acc < (fp16)0.0f) {
                    acc = 0.0f;
                }

                size_t out_idx = (k * dim_q + q) * dim_p + p;
                output[out_idx] = acc;
            }
        }
    }
}

static void pulp_nn_fp16_conv2d_par_c_(
    const fp16 *__restrict__ input,         // Pointer to the input feature map
    fp16 *__restrict__       im2col,        // Pointer to the shared im2col buffer
    const fp16 *__restrict__ bias,          // Pointer to the bias vector
    fp16 *__restrict__       output,        // Pointer to the output feature map
    const fp16 *__restrict__ weight,        // Pointer to the weights
    const uint16_t           dim_w,         // Input Width
    const uint16_t           dim_h,         // Input Height
    const uint16_t           dim_c,         // Input Channels
    const uint16_t           dim_p,         // Output Width
    const uint16_t           dim_q,         // Output Height
    const uint16_t           dim_k,         // Output Channels
    const uint16_t           dim_r,         // Kernel Width
    const uint16_t           dim_s,         // Kernel Height 
    const uint16_t           pad_t,         // Padding Top
    const uint16_t           pad_b,         // Padding Bottom
    const uint16_t           pad_l,         // Padding Left
    const uint16_t           pad_r,         // Padding Right
    const uint16_t           stride_x,      // Stride Horizontal
    const uint16_t           stride_y,      // Stride Vertical
    int                      apply_relu     // Apply ReLU activation
)
{
    // Calculate output channels per thread
    uint32_t chunk = (dim_k + nthreads - 1) / nthreads;
    uint32_t start_k = tid * chunk;
    uint32_t stop_k = (start_k + chunk > dim_k) ? dim_k : (start_k + chunk);

    // Each thread gets a private im2col region
    uint32_t im2col_size = dim_r * dim_s * dim_c;
    fp16 *im2col_thread = im2col + tid * im2col_size;

    for (uint16_t out_y = 0; out_y < dim_q; out_y++) {
        for (uint16_t out_x = 0; out_x < dim_p; out_x++) {
            // Prepare im2col for this (out_y, out_x)
            fp16 *im2col_ptr = im2col_thread;
            for (uint16_t ker_y = 0; ker_y < dim_s; ker_y++) {
                int32_t in_y = (int32_t)out_y * stride_y - pad_t + ker_y;
                for (uint16_t ker_x = 0; ker_x < dim_r; ker_x++) {
                    int32_t in_x = (int32_t)out_x * stride_x - pad_l + ker_x;
                    if (in_y < 0 || in_y >= dim_h || in_x < 0 || in_x >= dim_w) {
                        for (uint16_t c = 0; c < dim_c; c++) {
                            *im2col_ptr++ = (fp16)0.0f;
                        }
                    } else {
                        const fp16 *in_pixel = input + (in_y * dim_w + in_x) * dim_c;
                        for (uint16_t c = 0; c < dim_c; c++) {
                            *im2col_ptr++ = in_pixel[c];
                        }
                    }
                }
            }

            // Compute only this thread's output channels
            for (uint16_t k = start_k; k < stop_k; k++) {
                const fp16 *w = weight + k * im2col_size;
                fp16 acc = bias ? bias[k] : (fp16)0.0f;
                for (uint32_t i = 0; i < im2col_size; i++) {
                    acc += im2col_thread[i] * w[i];
                }
                if (apply_relu && acc < (fp16)0.0f) acc = (fp16)0.0f;
                output[((out_y * dim_p + out_x) * dim_k) + k] = acc;
            }
        }
    }
}
*/


#endif // __pulp_cluster__