#ifdef __pulp_cluster__

#include <stdint.h>

#include "pulp_kernels/pulp_fp16_defines.h"
#include "pulp_kernels/pulp_fp16_kernels.h"

static inline void pulp_fp16_maxpool2d_nhwc(
    const fp16 *input, fp16 *output,
    uint16_t dim_ix, uint16_t dim_iy, uint16_t dim_ic,
    uint16_t dim_ox, uint16_t dim_oy, uint16_t dim_oc,
    uint16_t dim_fx, uint16_t dim_fy,
    uint16_t pad_t, uint16_t pad_l,
    uint16_t stride_x, uint16_t stride_y
) {
    uint32_t work = (uint32_t)dim_oy * dim_ox * dim_oc;
    uint32_t chunk = (work + nthreads - 1) / nthreads;
    uint32_t start = min(tid * chunk, work);
    uint32_t end = min(start + chunk, work);

    for (uint32_t linear = start; linear < end; ++linear) {
        uint16_t oc = linear % dim_oc;
        uint32_t spatial = linear / dim_oc;
        uint16_t ox = spatial % dim_ox;
        uint16_t oy = spatial / dim_ox;
        int iy_start = (int)oy * stride_y - pad_t;
        int ix_start = (int)ox * stride_x - pad_l;
        fp16 value = -__FLT_MAX__;

        for (uint16_t fy = 0; fy < dim_fy; ++fy) {
            int iy = iy_start + fy;
            if (iy < 0 || iy >= dim_iy) continue;
            for (uint16_t fx = 0; fx < dim_fx; ++fx) {
                int ix = ix_start + fx;
                if (ix < 0 || ix >= dim_ix) continue;
                value = max(value, input[idx_NHWC(iy, ix, oc, dim_iy, dim_ix, dim_ic)]);
            }
        }
        output[idx_NHWC(oy, ox, oc, dim_oy, dim_ox, dim_oc)] = value;
    }
}

void __attribute__((noinline)) pulp_fp16_maxpool2d(
    const fp16 *__restrict__ input, fp16 *__restrict__ output,
    uint16_t dim_ix, uint16_t dim_iy, uint16_t dim_ic,
    uint16_t dim_ox, uint16_t dim_oy, uint16_t dim_oc,
    uint16_t dim_fx, uint16_t dim_fy,
    uint16_t pad_t, uint16_t pad_b, uint16_t pad_l, uint16_t pad_r,
    uint16_t stride_x, uint16_t stride_y
) {
    (void)pad_b;
    (void)pad_r;
    pulp_fp16_maxpool2d_nhwc(input, output, dim_ix, dim_iy, dim_ic,
                             dim_ox, dim_oy, dim_oc, dim_fx, dim_fy,
                             pad_t, pad_l, stride_x, stride_y);
}

#endif