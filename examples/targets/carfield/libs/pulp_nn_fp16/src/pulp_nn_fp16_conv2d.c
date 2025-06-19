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


static void pulp_nn_fp16_conv2d_nhwc_ohwi_par_oc_im2col(
    const fp16   *input,      // NHWC
    const fp16   *weight,     // OHWI
    const fp16   *bias,       // O
    fp16         *output,     // NHWC
    fp16         *im2col,     // im2col buffer (private per thread)
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
)
{
    // Parallelize on output channel (dim_oc)
    uint32_t chunk = (dim_oc + nthreads - 1) / nthreads;
    uint32_t start_oc = tid * chunk;
    uint32_t stop_oc  = (start_oc + chunk > dim_oc) ? dim_oc : (start_oc + chunk);

    uint32_t im2col_size = dim_fx * dim_fy * dim_ic;
    fp16 *im2col_thread = im2col + tid * im2col_size;

    for (uint32_t oy = 0; oy < dim_oy; oy++) {
        for (uint32_t ox = 0; ox < dim_ox; ox++) {
            // Prepare im2col for (oy, ox)
            fp16 *im2col_ptr = im2col_thread;
            for (uint32_t fy = 0; fy < dim_fy; fy++) {
                int32_t iy = (int32_t)oy * stride_y - pad_t + fy;
                for (uint32_t fx = 0; fx < dim_fx; fx++) {
                    int32_t ix = (int32_t)ox * stride_x - pad_l + fx;
                    if (iy < 0 || iy >= (int32_t)dim_iy || ix < 0 || ix >= (int32_t)dim_ix) {
                        for (uint32_t ic = 0; ic < dim_ic; ic++) {
                            *im2col_ptr++ = (fp16)0.0f;
                        }
                    } else {
                        const fp16 *in_pixel = input + ((iy * dim_ix + ix) * dim_ic);
                        for (uint32_t ic = 0; ic < dim_ic; ic++) {
                            *im2col_ptr++ = in_pixel[ic];
                        }
                    }
                }
            }

            // Compute
            for (uint32_t oc = start_oc; oc < stop_oc; oc++) {
                const fp16 *w = weight + oc * dim_fy * dim_fx * dim_ic;
                fp16 acc = bias ? bias[oc] : (fp16)0.0f;
                for (uint32_t i = 0; i < im2col_size; i++) {
                    acc += im2col_thread[i] * w[i];
                }
                if (apply_relu && acc < (fp16)0.0f) acc = (fp16)0.0f;
                output[((oy * dim_ox + ox) * dim_oc) + oc] = acc;
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
    pulp_nn_fp16_conv2d_nhwc_ohwi_par_oc_im2col(
        input, weight, bias, output, im2col,
        dim_ix, dim_iy, dim_ic,
        dim_ox, dim_oy, dim_oc,
        dim_fx, dim_fy,
        pad_t, pad_b, pad_l, pad_r,
        stride_x, stride_y, 
        apply_relu
    );
}


#endif // __pulp_cluster__