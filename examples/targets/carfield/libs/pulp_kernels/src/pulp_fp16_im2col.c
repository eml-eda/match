#include "pulp_kernels/pulp_fp16_defines.h"
#include "pulp_kernels/pulp_fp16_kernels.h"


static void pulp_fp16_im2col_nhwc_par_ox(
    const fp16 *input,         // NHWC
    fp16 *im2col,
    uint32_t dim_ix,           // Input Width
    uint32_t dim_iy,           // Input Height
    uint32_t dim_ic,           // Input Channels
    uint32_t dim_ox,           // Output Width
    uint32_t dim_oy,           // Output Height
    uint32_t dim_fx,           // Kernel Width
    uint32_t dim_fy,           // Kernel Height
    uint32_t pad_t,            // Padding Top
    uint32_t pad_l,            // Padding Left
    uint32_t stride_x,         // Stride Horizontal
    uint32_t stride_y          // Stride Vertical
)
{
    int ox_chunk = (dim_ox + nthreads - 1) / nthreads;
    int ox_start = tid * ox_chunk;
    int ox_end = ox_start + ox_chunk;
    if (ox_end > dim_ox) ox_end = dim_ox;

    for (int oy = 0; oy < dim_oy; ++oy) {
        for (int ox = ox_start; ox < ox_end; ++ox) {
            int col = (oy * dim_ox + ox);
            for (int fy = 0; fy < dim_fy; ++fy) {
                for (int fx = 0; fx < dim_fx; ++fx) {
                    for (int ic = 0; ic < dim_ic; ++ic) {
                        int iy = oy * stride_y + fy - pad_t;
                        int ix = ox * stride_x + fx - pad_l;
                        int im2col_idx = col * (dim_fy * dim_fx * dim_ic) +
                                         ((fy * dim_fx + fx) * dim_ic + ic);

                        if (iy >= 0 && iy < dim_iy && ix >= 0 && ix < dim_ix)
                            im2col[im2col_idx] = input[idx_NHWC(iy, ix, ic, dim_iy, dim_ix, dim_ic)];
                        else
                            im2col[im2col_idx] = (fp16)0.0f;
                    }
                }
            }
        }
    }
}


static void pulp_fp16_im2col_nchw_par_ox(
    const fp16 *input,         // NCHW
    fp16 *im2col,
    uint32_t dim_ix,           // Input Width
    uint32_t dim_iy,           // Input Height
    uint32_t dim_ic,           // Input Channels
    uint32_t dim_ox,           // Output Width
    uint32_t dim_oy,           // Output Height
    uint32_t dim_fx,           // Kernel Width
    uint32_t dim_fy,           // Kernel Height
    uint32_t pad_t,            // Padding Top
    uint32_t pad_l,            // Padding Left
    uint32_t stride_x,         // Stride Horizontal
    uint32_t stride_y          // Stride Vertical
)
{
    int ox_chunk = (dim_ox + nthreads - 1) / nthreads;
    int ox_start = tid * ox_chunk;
    int ox_end = ox_start + ox_chunk;
    if (ox_end > dim_ox) ox_end = dim_ox;

    for (int oy = 0; oy < dim_oy; ++oy) {
        for (int ox = ox_start; ox < ox_end; ++ox) {
            int col = (oy * dim_ox + ox);
            for (int fy = 0; fy < dim_fy; ++fy) {
                for (int fx = 0; fx < dim_fx; ++fx) {
                    for (int ic = 0; ic < dim_ic; ++ic) {
                        int iy = oy * stride_y + fy - pad_t;
                        int ix = ox * stride_x + fx - pad_l;
                        int im2col_idx = col * (dim_fy * dim_fx * dim_ic) +
                                         ((fy * dim_fx + fx) * dim_ic + ic);

                        if (iy >= 0 && iy < dim_iy && ix >= 0 && ix < dim_ix)
                            im2col[im2col_idx] = input[idx_NCHW(iy, ix, ic, dim_iy, dim_ix, dim_ic)];
                        else
                            im2col[im2col_idx] = (fp16)0.0f;
                    }
                }
            }
        }
    }
}


void pulp_fp16_im2col(
    const fp16 *__restrict__ input,        
    fp16 *__restrict__       im2col,
    uint32_t dim_ix,           // Input Width
    uint32_t dim_iy,           // Input Height
    uint32_t dim_ic,           // Input Channels
    uint32_t dim_ox,           // Output Width
    uint32_t dim_oy,           // Output Height
    uint32_t dim_fx,           // Kernel Width
    uint32_t dim_fy,           // Kernel Height
    uint32_t pad_t,            // Padding Top
    uint32_t pad_l,            // Padding Left
    uint32_t stride_x,         // Stride Horizontal
    uint32_t stride_y          // Stride Vertical
)
{
    pulp_fp16_im2col_nhwc_par_ox(
        input, im2col,
        dim_ix, dim_iy, dim_ic,
        dim_ox, dim_oy,
        dim_fx, dim_fy,
        pad_t, pad_l,
        stride_x, stride_y
    );
}

