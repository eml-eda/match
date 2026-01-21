#ifdef __pulp_cluster__

#include "pulp_kernels/pulp_fp16_defines.h"
#include "pulp_kernels/pulp_fp16_kernels.h"

#include <stdint.h>

// Parallelize over output channels
static inline void pulp_fp16_avgpool2d_nhwc_par_oc(
    const fp16 *input,
    fp16 *output,
    uint16_t dim_iy, uint16_t dim_ix, uint16_t dim_ic,
    uint16_t dim_oy, uint16_t dim_ox, uint16_t dim_oc,
    uint16_t dim_fy, uint16_t dim_fx,
    uint16_t pad_t, uint16_t pad_b,
    uint16_t pad_l, uint16_t pad_r,
    uint16_t stride_y, uint16_t stride_x
) {
    uint16_t ch_per_core = (dim_oc + nthreads - 1) / nthreads;
    uint16_t ch_start = tid * ch_per_core;
    uint16_t ch_end = ch_start + ch_per_core;
    if (ch_end > dim_oc) ch_end = dim_oc;

    for (uint16_t oc = ch_start; oc < ch_end; ++oc) {
        for (uint16_t oy = 0; oy < dim_oy; ++oy) {
            for (uint16_t ox = 0; ox < dim_ox; ++ox) {
                int in_iy_start = oy * stride_y - pad_t;
                int in_ix_start = ox * stride_x - pad_l;
                int in_iy_end = in_iy_start + dim_fy;
                int in_ix_end = in_ix_start + dim_fx;

                int iy_start = in_iy_start < 0 ? 0 : in_iy_start;
                int ix_start = in_ix_start < 0 ? 0 : in_ix_start;
                int iy_end = in_iy_end > dim_iy ? dim_iy : in_iy_end;
                int ix_end = in_ix_end > dim_ix ? dim_ix : in_ix_end;

                int pool_area = (iy_end - iy_start) * (ix_end - ix_start);

                fp16 acc = 0.0f;
                for (int iy = iy_start; iy < iy_end; ++iy) {
                    for (int ix = ix_start; ix < ix_end; ++ix) {
                        int input_idx = idx_NHWC(iy, ix, oc, dim_iy, dim_ix, dim_ic);
                        acc += input[input_idx];
                    }
                }
                fp16 avg = (pool_area > 0) ? acc / pool_area : 0.0f;
                int out_idx = idx_NHWC(oy, ox, oc, dim_oy, dim_ox, dim_oc);
                output[out_idx] = avg;
            }
        }
    }
}

// Parallelize over output rows
static inline void pulp_fp16_avgpool2d_nhwc_par_oy(
    const fp16 *input, fp16 *output,
    uint16_t dim_ix, uint16_t dim_iy, uint16_t dim_ic,
    uint16_t dim_ox, uint16_t dim_oy, uint16_t dim_oc,
    uint16_t dim_fx, uint16_t dim_fy,
    uint16_t pad_t, uint16_t pad_b, uint16_t pad_l, uint16_t pad_r,
    uint16_t stride_y, uint16_t stride_x
) {
    uint16_t out_rows_per_core = (dim_oy + nthreads - 1) / nthreads;
    uint16_t oy_start = tid * out_rows_per_core;
    uint16_t oy_end = oy_start + out_rows_per_core;
    if (oy_end > dim_oy) oy_end = dim_oy;

    for (uint16_t oy = oy_start; oy < oy_end; ++oy) {
        for (uint16_t ox = 0; ox < dim_ox; ++ox) {
            int in_iy_start = oy * stride_y - pad_t;
            int in_ix_start = ox * stride_x - pad_l;
            int in_iy_end = in_iy_start + dim_fy;
            int in_ix_end = in_ix_start + dim_fx;

            int iy_start = in_iy_start < 0 ? 0 : in_iy_start;
            int ix_start = in_ix_start < 0 ? 0 : in_ix_start;
            int iy_end = in_iy_end > dim_iy ? dim_iy : in_iy_end;
            int ix_end = in_ix_end > dim_ix ? dim_ix : in_ix_end;

            int pool_area = (iy_end - iy_start) * (ix_end - ix_start);

            for (uint16_t oc = 0; oc < dim_oc; ++oc) {
                fp16 acc = 0.0f;
                for (int iy = iy_start; iy < iy_end; ++iy) {
                    for (int ix = ix_start; ix < ix_end; ++ix) {
                        int input_idx = idx_NHWC(iy, ix, oc, dim_iy, dim_ix, dim_ic);
                        acc += input[input_idx];
                    }
                }
                fp16 avg = (pool_area > 0) ? acc / pool_area : 0.0f;
                int out_idx = idx_NHWC(oy, ox, oc, dim_oy, dim_ox, dim_oc);
                output[out_idx] = avg;
            }
        }
    }
}

void __attribute__((noinline)) pulp_fp16_avgpool2d(
    const fp16 *__restrict__ input,
    fp16 *__restrict__ output,
    uint16_t dim_ix, 
    uint16_t dim_iy, 
    uint16_t dim_ic,
    uint16_t dim_ox, 
    uint16_t dim_oy, 
    uint16_t dim_oc,
    uint16_t dim_fx, 
    uint16_t dim_fy,
    uint16_t pad_t, 
    uint16_t pad_b,
    uint16_t pad_l, 
    uint16_t pad_r,
    uint16_t stride_x,
    uint16_t stride_y 
) {
    uint32_t out_elems = (uint32_t)dim_oy * dim_ox;

    if (dim_oc >= out_elems) {
        pulp_fp16_avgpool2d_nhwc_par_oc(
            input, output,
            dim_ix, dim_iy, dim_ic,
            dim_ox, dim_oy, dim_oc,
            dim_fx, dim_fy,
            pad_t, pad_b, pad_l, pad_r,
            stride_x, stride_y
        );
    } else {
        pulp_fp16_avgpool2d_nhwc_par_oy(
            input, output,
            dim_ix, dim_iy, dim_ic,
            dim_ox, dim_oy, dim_oc,
            dim_fx, dim_fy,
            pad_t, pad_b, pad_l, pad_r,
            stride_x, stride_y
        );
    }
}

#endif // __pulp_cluster__
