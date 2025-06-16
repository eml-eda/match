#ifdef __pulp_cluster__

#include "pulp_nn_fp16/pulp_nn_fp16_defines.h"
#include "pulp_nn_fp16/pulp_nn_fp16_kernels.h"

#include <stdint.h>

static void pulp_nn_fp16_conv2d_par_c(
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
    // Calculate channels per thread (output channels parallelism)
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


void __attribute__((noinline)) pulp_nn_fp16_conv2d(
    const fp16 *__restrict__ input,         // Pointer to the input feature map
    fp16 *__restrict__       im2col,        // Pointer to the im2col buffer
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
) {
    pulp_nn_fp16_conv2d_par_c(
        input, im2col, bias, output, weight,
        dim_w, dim_h, dim_c, dim_p, dim_q, dim_k,
        dim_r, dim_s, pad_t, pad_b, pad_l, pad_r,
        stride_x, stride_y, apply_relu
    );
}

#endif // __pulp_cluster__