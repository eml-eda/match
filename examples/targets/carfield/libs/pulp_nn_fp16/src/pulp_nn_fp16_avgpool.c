#ifdef __pulp_cluster__

#include "pulp_nn_fp16/pulp_nn_fp16_defines.h"
#include "pulp_nn_fp16/pulp_nn_fp16_kernels.h"

#include <stdint.h>


// Parallelize over dim_c
static void pulp_nn_fp16_avgpool_par_c(
    const fp16 *input,
    fp16 *output,
    uint16_t dim_h, uint16_t dim_w, uint16_t dim_c,
    uint16_t dim_p, uint16_t dim_q,
    uint16_t dim_r, uint16_t dim_s,
    uint16_t pad_t, uint16_t pad_b,
    uint16_t pad_l, uint16_t pad_r,
    uint16_t stride_y, uint16_t stride_x
) {
    uint16_t ch_per_core = (dim_c + nthreads - 1) / nthreads;
    uint16_t ch_start = tid * ch_per_core;
    uint16_t ch_end = ch_start + ch_per_core;
    if (ch_end > dim_c) ch_end = dim_c;

    for (uint16_t c = ch_start; c < ch_end; ++c) {
        for (uint16_t oh = 0; oh < dim_p; ++oh) {
            for (uint16_t ow = 0; ow < dim_q; ++ow) {
                int in_h_start = oh * stride_y - pad_t;
                int in_w_start = ow * stride_x - pad_l;
                int in_h_end = in_h_start + dim_r;
                int in_w_end = in_w_start + dim_s;

                int h_start = in_h_start < 0 ? 0 : in_h_start;
                int w_start = in_w_start < 0 ? 0 : in_w_start;
                int h_end = in_h_end > dim_h ? dim_h : in_h_end;
                int w_end = in_w_end > dim_w ? dim_w : in_w_end;

                int pool_area = (h_end - h_start) * (w_end - w_start);

                float acc = 0.0f;
                for (int ih = h_start; ih < h_end; ++ih) {
                    for (int iw = w_start; iw < w_end; ++iw) {
                        int in_idx = (ih * dim_w + iw) * dim_c + c;
                        acc += (float)input[in_idx];
                    }
                }
                float avg = (pool_area > 0) ? acc / pool_area : 0.0f;
                int out_idx = (oh * dim_q + ow) * dim_c + c;
                output[out_idx] = (fp16)avg;
            }
        }
    }
}

// Parallelize over output rows
static void pulp_nn_fp16_avgpool_par_oh(
    const fp16 *input,
    fp16 *output,
    uint16_t dim_h, uint16_t dim_w, uint16_t dim_c,
    uint16_t dim_p, uint16_t dim_q,
    uint16_t dim_r, uint16_t dim_s,
    uint16_t pad_t, uint16_t pad_b,
    uint16_t pad_l, uint16_t pad_r,
    uint16_t stride_y, uint16_t stride_x
) {
    uint16_t out_rows_per_core = (dim_p + nthreads - 1) / nthreads;
    uint16_t oh_start = tid * out_rows_per_core;
    uint16_t oh_end = oh_start + out_rows_per_core;
    if (oh_end > dim_p) oh_end = dim_p;

    for (uint16_t oh = oh_start; oh < oh_end; ++oh) {
        for (uint16_t ow = 0; ow < dim_q; ++ow) {
            int in_h_start = oh * stride_y - pad_t;
            int in_w_start = ow * stride_x - pad_l;
            int in_h_end = in_h_start + dim_r;
            int in_w_end = in_w_start + dim_s;

            int h_start = in_h_start < 0 ? 0 : in_h_start;
            int w_start = in_w_start < 0 ? 0 : in_w_start;
            int h_end = in_h_end > dim_h ? dim_h : in_h_end;
            int w_end = in_w_end > dim_w ? dim_w : in_w_end;

            int pool_area = (h_end - h_start) * (w_end - w_start);

            for (uint16_t c = 0; c < dim_c; ++c) {
                float acc = 0.0f;
                for (int ih = h_start; ih < h_end; ++ih) {
                    for (int iw = w_start; iw < w_end; ++iw) {
                        int in_idx = (ih * dim_w + iw) * dim_c + c;
                        acc += (float)input[in_idx];
                    }
                }
                float avg = (pool_area > 0) ? acc / pool_area : 0.0f;
                int out_idx = (oh * dim_q + ow) * dim_c + c;
                output[out_idx] = (fp16)avg;
            }
        }
    }
}

void __attribute__((noinline)) pulp_nn_fp16_avgpool(
    const fp16 *input,
    fp16 *output,
    uint16_t dim_h, uint16_t dim_w, uint16_t dim_c,
    uint16_t dim_p, uint16_t dim_q,
    uint16_t dim_r, uint16_t dim_s,
    uint16_t pad_t, uint16_t pad_b,
    uint16_t pad_l, uint16_t pad_r,
    uint16_t stride_y, uint16_t stride_x
) {
    uint32_t out_elems = (uint32_t)dim_p * dim_q;

    if (dim_c >= out_elems) {
        pulp_nn_fp16_avgpool_par_c(
            input, output,
            dim_h, dim_w, dim_c,
            dim_p, dim_q,
            dim_r, dim_s,
            pad_t, pad_b,
            pad_l, pad_r,
            stride_y, stride_x
        );
    } else {
        pulp_nn_fp16_avgpool_par_oh(
            input, output,
            dim_h, dim_w, dim_c,
            dim_p, dim_q,
            dim_r, dim_s,
            pad_t, pad_b,
            pad_l, pad_r,
            stride_y, stride_x
        );
    }
}

#endif // __pulp_cluster__