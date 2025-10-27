#ifdef __spatz__

#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#include "carfield_lib/printf.h"
#include "carfield_lib/spatz.h"
#include "spatz_kernels/spatz_fp16_kernels.h"




static void spatz_conv2d_fp16_nhwc_ohwi_rvv(
    const fp16* input,      // NHWC
    const fp16* weight,     // OHWI
    const fp16* bias,       // [O] or NULL
    const fp16* bnorm_mul,  // [O] or NULL
    const fp16*  bnorm_add, // [O] or NULL
    fp16* output,           // NHWC
    uint32_t dim_ix,                 // input width
    uint32_t dim_iy,                 // input height
    uint32_t dim_ic,                 // input channels
    uint32_t dim_ox,                 // output width
    uint32_t dim_oy,                 // output height
    uint32_t dim_oc,                 // output channels
    uint32_t dim_fx,                 // kernel width
    uint32_t dim_fy,                 // kernel height
    uint32_t pad_t, uint32_t pad_b, uint32_t pad_l, uint32_t pad_r,
    uint32_t stride_x, uint32_t stride_y, uint32_t apply_relu
) {
    int oc_chunk = (dim_oc + nthreads - 1) / nthreads;
    int oc_start = min(tid * oc_chunk, dim_oc);
    int oc_end = min(oc_start + oc_chunk, dim_oc);

    if (dim_oc <= 2) {
        if (tid != 0) goto sync;
        oc_start = 0;
        oc_end = dim_oc;
    } 

    for (uint32_t oy = 0; oy < dim_oy; ++oy) {
        for (uint32_t ox = 0; ox < dim_ox; ++ox) {
            // Convlution
    
            for (uint32_t oc = oc_start; oc < oc_end; ++oc) {
                _Float16 dot = 0.0f;

                asm volatile("vsetvli zero, %0, e16, m8, ta, ma" :: "r"(dim_ic));
                asm volatile("vxor.vv v24, v24, v24");
                asm volatile("vmv.s.x v0, zero");

                uint32_t res_ic = 0;
                while (res_ic < dim_ic) {
                    uint32_t remaining = dim_ic - res_ic;
                    uint32_t vl;

                    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(remaining));

                    for (int fy = 0; fy < (int)dim_fy; ++fy) {
                        for (int fx = 0; fx < (int)dim_fx; ++fx) {
                            int iy = oy * stride_y + fy - pad_t;
                            int ix = ox * stride_x + fx - pad_l;

                            if (iy >= 0 && iy < dim_iy && ix >= 0 && ix < dim_ix) {

                                const _Float16* in_ptr =
                                    &input[idx_NHWC(iy, ix, res_ic, dim_iy, dim_ix, dim_ic)];
                                const _Float16* w_ptr =
                                    &weight[idx_OHWI(fy, fx, res_ic, oc, dim_fy, dim_fx, dim_ic, dim_oc)];


                                asm volatile("vle16.v v8, (%0)"  :: "r"(in_ptr) : "v8",  "memory");
                                asm volatile("vle16.v v16, (%0)" :: "r"(w_ptr)  : "v16", "memory");

                                asm volatile("vfmacc.vv v24, v8, v16" ::: "v24", "v8", "v16");
                            }
                        }
                        //asm volatile("fence rw, rw" ::: "memory");
                    }
                    //asm volatile("fence rw, rw" ::: "memory");
                    res_ic = res_ic + vl;
                }

                asm volatile("vsetvli zero, %0, e16, m8, ta, ma" :: "r"(dim_ic));
                asm volatile("vfredsum.vs v0, v24, v0" );
                asm volatile("vfmv.f.s %0, v0" : "=f"(dot));

                int output_idx = idx_NHWC(oy, ox, oc, dim_oy, dim_ox, dim_oc);
                output[output_idx] = dot;
            }  // oc

            // Post-process

            uint32_t res_oc = 0;
            fp16* out_ptr = &output[idx_NHWC(oy, ox, oc_start, dim_oy, dim_ox, dim_oc)];

            while (res_oc < oc_end - oc_start && (bias || bnorm_mul || bnorm_add || apply_relu)) {
                uint32_t remaining = oc_end - oc_start - res_oc;
                uint32_t vl;

                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(remaining));

                asm volatile("vle16.v v0, (%0)" :: "r"(&out_ptr[res_oc]) : "v0", "memory");

                if (bias) {
                    asm volatile("vle16.v v8, (%0)" :: "r"(&bias[oc_start + res_oc]) : "v8", "memory");
                    asm volatile("vfadd.vv v0, v0, v8" ::: "v0", "v8");
                }

                if (bnorm_mul && bnorm_add) {
                    asm volatile("vle16.v v8, (%0)" :: "r"(&bnorm_mul[oc_start + res_oc]) : "v8", "memory");
                    asm volatile("vle16.v v16, (%0)" :: "r"(&bnorm_add[oc_start + res_oc]) : "v16", "memory");
                    asm volatile("vfmul.vv v0, v0, v8" ::: "v0", "v8");
                    asm volatile("vfadd.vv v0, v0, v16" ::: "v0", "v16");
                }

                if (apply_relu) {
                    asm volatile("vfmax.vf v0, v0, %0" :: "f"(0.0f) : "v0");
                }

                asm volatile("vse16.v v0, (%0)" :: "r"(&out_ptr[res_oc]) : "v0", "memory");

                res_oc += vl;
            }
        }      // ox
    }          // oy

    sync:
    spatz_sync_cores(NULL);        
}





static void spatz_conv2d_fp16_nhwc_ohwi_fdotp(
    const fp16* input,          // NHWC
    const fp16* weight,         // OHWI
    const fp16* bias,           // [O] or NULL
    const fp16* bnorm_mul,      // [O] or NULL
    const fp16*  bnorm_add,     // [O] or NULL
    fp16* output,               // NHWC
    uint32_t dim_ix,            // input width
    uint32_t dim_iy,            // input height
    uint32_t dim_ic,            // input channels
    uint32_t dim_ox,            // output width
    uint32_t dim_oy,            // output height
    uint32_t dim_oc,            // output channels
    uint32_t dim_fx,            // kernel width
    uint32_t dim_fy,            // kernel height
    uint32_t pad_t, uint32_t pad_b, uint32_t pad_l, uint32_t pad_r,
    uint32_t stride_x, uint32_t stride_y, 
    uint32_t apply_relu
) {

    int oc_chunk = (dim_oc + nthreads - 1) / nthreads;
    int oc_start = min(tid * oc_chunk, dim_oc);
    int oc_end = min(oc_start + oc_chunk, dim_oc);

    if (dim_oc <= 2) {
        if (tid != 0) return;
        oc_start = 0;
        oc_end = dim_oc;
    } 

    for (uint32_t oy = 0; oy < dim_oy; ++oy) {
        for (uint32_t ox = 0; ox < dim_ox; ++ox) {
            for (uint32_t oc = 0; oc < dim_oc; ++oc) {
            
                _Float16 dot = 0.0f;

                for (int fy = 0; fy < (int)dim_fy; ++fy) {
                    for (int fx = 0; fx < (int)dim_fx; ++fx) {
                        int iy = oy * stride_y + fy - pad_t;
                        int ix = ox * stride_x + fx - pad_l;

                        if (iy >= 0 && iy < dim_iy && ix >= 0 && ix < dim_ix) {

                            const _Float16* in_ptr =
                                &input[idx_NHWC(iy, ix, 0, dim_iy, dim_ix, dim_ic)];
                            const _Float16* w_ptr =
                                &weight[idx_OHWI(fy, fx, 0, oc, dim_fy, dim_fx, dim_ic, dim_oc)];

                            _Float16 tmp = spatz_fp16_fdotp(in_ptr, w_ptr, dim_ic);

                            dot = dot + tmp;
                        }
                    }
                }
                
                int output_idx = idx_NHWC(oy, ox, oc, dim_oy, dim_ox, dim_oc);
                output[output_idx] = dot;

            }  // oc
        }      // ox
    }          // oy

    end:
    spatz_sync_cores(NULL);        
}





static void spatz_fp16_conv2d_nhwc_ohwi_naive(
    const fp16   *input,      // NHWC
    const fp16   *weight,     // OHWI
    const fp16   *bias,       // O
    const fp16   *bnorm_mul,  // O
    const fp16   *bnorm_add,  // O
    fp16         *output,     // NHWC
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
    if (tid != 0) goto end_1;
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" :: "r"(1));

    // Parallelize on output channel (dim_oc)
    for (int oy = 0; oy < dim_oy; ++oy) {   // output height
        for (int ox = 0; ox < dim_ox; ++ox) { // output width
            for (int oc = 0; oc < dim_oc; ++oc) { // output channel
                // Add Bias
                asm volatile("vmv.s.x v0, zero");
                // For each kernel position
                for (int ic = 0; ic < dim_ic; ++ic) { // input channel
                    for (int fy = 0; fy < dim_fy; ++fy) { // filter height
                        for (int fx = 0; fx < dim_fx; ++fx) { // filter width
                            int iy = oy * stride_y + fy - pad_t;
                            int ix = ox * stride_x + fx - pad_l;

                            if (iy >= 0 && iy < dim_iy && ix >= 0 && ix < dim_ix) {
                                int input_idx  = idx_NHWC(iy, ix, ic, dim_iy, dim_ix, dim_ic);
                                int weight_idx = idx_OHWI(fy, fx, ic, oc, dim_fy, dim_fx, dim_ic, dim_oc);
                                // dot += input[input_idx] * weight[weight_idx];
                                asm volatile("vle16.v v8, (%0)" :: "r"(&input[input_idx]) : "v8", "memory");
                                asm volatile("vle16.v v16, (%0)" :: "r"(&weight[weight_idx]) : "v16", "memory");
                                asm volatile("vfmacc.vv v0, v8, v16" ::: "v0", "v8", "v16");
                            }
                        }
                    }
                }
                
                int output_idx = idx_NHWC(oy, ox, oc, dim_oy, dim_ox, dim_oc);
                //output[output_idx] = dot;
                asm volatile("vse16.v v0, (%0)" :: "r"(&output[output_idx]) : "v0", "memory");
            }
        }
    }
    end_1:
        spatz_sync_cores(NULL);        
}



void spatz_fp16_conv2d(
    const fp16* __restrict__ input,      // Pointer to the input feature map
    const fp16* __restrict__ weight,     // Pointer to the weights
    const fp16* __restrict__ bias,       // Pointer to the bias vector
    const fp16* __restrict__ bnorm_mul,  // Pointer to the batch normalization scale
    const fp16* __restrict__ bnorm_add,  // Pointer to the batch normalization offset
    fp16* __restrict__ output,           // Pointer to the output feature map
    fp16* __restrict__ im2col,           // Pointer to the im2col buffer
    uint32_t dim_ix,                     // Input Width
    uint32_t dim_iy,                     // Input Height
    uint32_t dim_ic,                     // Input Channels
    uint32_t dim_ox,                     // Output Width
    uint32_t dim_oy,                     // Output Height
    uint32_t dim_oc,                     // Output Channels
    uint32_t dim_fx,                     // Kernel Width
    uint32_t dim_fy,                     // Kernel Height
    uint32_t pad_t,                      // Padding Top
    uint32_t pad_b,                      // Padding Bottom
    uint32_t pad_l,                      // Padding Left
    uint32_t pad_r,                      // Padding Right
    uint32_t stride_x,                   // Stride Horizontal
    uint32_t stride_y,                   // Stride Vertical
    uint32_t apply_relu                  // Apply ReLU activation
) {
    spatz_conv2d_fp16_nhwc_ohwi_rvv(input, weight, bias, bnorm_mul, bnorm_add, output,  // im2col,
                              dim_ix, dim_iy, dim_ic, dim_ox, dim_oy, dim_oc, dim_fx, dim_fy, pad_t,
                              pad_b, pad_l, pad_r, stride_x, stride_y, apply_relu);
}




// Grouped



static void spatz_conv2d_fp16_nhwc_ohwi_rvv_grouped(
    const fp16* input,       // NHWC
    const fp16* weight,      // OHWI (CO, H, W, group_ic)
    const fp16* bias,        // [CO]
    const fp16* bnorm_mul,   // [CO]
    const fp16* bnorm_add,   // [CO]
    fp16* output,            // NHWC
    uint32_t dim_ix,         // input width
    uint32_t dim_iy,         // input height
    uint32_t dim_ic,         // input channels
    uint32_t dim_ox,         // output width
    uint32_t dim_oy,         // output height
    uint32_t dim_oc,         // output channels
    uint32_t dim_fx,         // kernel width
    uint32_t dim_fy,         // kernel height
    uint32_t pad_t, uint32_t pad_b, uint32_t pad_l, uint32_t pad_r,
    uint32_t stride_x, uint32_t stride_y, uint32_t apply_relu,
    uint32_t groups          // number of groups
) {
    uint32_t group_ic = dim_ic / groups;
    uint32_t group_oc = dim_oc / groups;

    int oc_chunk = (dim_oc + nthreads - 1) / nthreads;
    int oc_start = min(tid * oc_chunk, dim_oc);
    int oc_end = min(oc_start + oc_chunk, dim_oc);

    if (dim_oc <= 2) {
        if (tid != 0) goto sync;
        oc_start = 0;
        oc_end = dim_oc;
    } 

    for (uint32_t oy = 0; oy < dim_oy; ++oy) {
        for (uint32_t ox = 0; ox < dim_ox; ++ox) {
            for (uint32_t oc = oc_start; oc < oc_end; ++oc) {
                uint32_t group = oc / group_oc;
                uint32_t oc_in_group = oc % group_oc;

                _Float16 dot = 0.0f;

                asm volatile("vsetvli zero, %0, e16, m8, ta, ma" :: "r"(group_ic));
                asm volatile("vxor.vv v24, v24, v24");
                asm volatile("vmv.s.x v0, zero");

                uint32_t res_ic_in_group = 0;
                while (res_ic_in_group < group_ic) {
                    uint32_t remaining = group_ic - res_ic_in_group;
                    uint32_t vl;
                    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(remaining));

                    for (int fy = 0; fy < (int)dim_fy; ++fy) {
                        for (int fx = 0; fx < (int)dim_fx; ++fx) {
                            int iy = oy * stride_y + fy - pad_t;
                            int ix = ox * stride_x + fx - pad_l;
                            if (iy >= 0 && iy < dim_iy && ix >= 0 && ix < dim_ix) {
                                // Input: NHWC, index into correct group channels
                                const _Float16* in_ptr =
                                    &input[idx_NHWC(iy, ix, group * group_ic + res_ic_in_group, dim_iy, dim_ix, dim_ic)];
                                // Weight: [oc, fy, fx, ic_in_group]
                                const _Float16* w_ptr =
                                    &weight[oc * dim_fy * dim_fx * group_ic +
                                            fy * dim_fx * group_ic +
                                            fx * group_ic +
                                            res_ic_in_group];
                                asm volatile("vle16.v v8, (%0)"  :: "r"(in_ptr) : "v8",  "memory");
                                asm volatile("vle16.v v16, (%0)" :: "r"(w_ptr)  : "v16", "memory");
                                asm volatile("vfmacc.vv v24, v8, v16" ::: "v24", "v8", "v16");
                            }
                        }
                    }
                    res_ic_in_group += vl;
                }

                asm volatile("vsetvli zero, %0, e16, m8, ta, ma" :: "r"(group_ic));
                asm volatile("vfredsum.vs v0, v24, v0" );
                asm volatile("vfmv.f.s %0, v0" : "=f"(dot));

                int output_idx = idx_NHWC(oy, ox, oc, dim_oy, dim_ox, dim_oc);
                output[output_idx] = dot;
            }

            // Post-process (same as before)
            uint32_t res_oc = 0;
            fp16* out_ptr = &output[idx_NHWC(oy, ox, oc_start, dim_oy, dim_ox, dim_oc)];

            while (res_oc < oc_end - oc_start && (bias || bnorm_mul || bnorm_add || apply_relu)) {
                uint32_t remaining = oc_end - oc_start - res_oc;
                uint32_t vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(remaining));
                asm volatile("vle16.v v0, (%0)" :: "r"(&out_ptr[res_oc]) : "v0", "memory");

                if (bias) {
                    asm volatile("vle16.v v8, (%0)" :: "r"(&bias[oc_start + res_oc]) : "v8", "memory");
                    asm volatile("vfadd.vv v0, v0, v8" ::: "v0", "v8");
                }
                if (bnorm_mul && bnorm_add) {
                    asm volatile("vle16.v v8, (%0)" :: "r"(&bnorm_mul[oc_start + res_oc]) : "v8", "memory");
                    asm volatile("vle16.v v16, (%0)" :: "r"(&bnorm_add[oc_start + res_oc]) : "v16", "memory");
                    asm volatile("vfmul.vv v0, v0, v8" ::: "v0", "v8");
                    asm volatile("vfadd.vv v0, v0, v16" ::: "v0", "v16");
                }
                if (apply_relu) {
                    asm volatile("vfmax.vf v0, v0, %0" :: "f"(0.0f) : "v0");
                }
                asm volatile("vse16.v v0, (%0)" :: "r"(&out_ptr[res_oc]) : "v0", "memory");
                res_oc += vl;
            }
        }
    }
sync:
    spatz_sync_cores(NULL);
}


void spatz_fp16_conv2d_grouped(
    const fp16* __restrict__ input,      // Pointer to the input feature map
    const fp16* __restrict__ weight,     // Pointer to the weights
    const fp16* __restrict__ bias,       // Pointer to the bias vector
    const fp16* __restrict__ bnorm_mul,  // Pointer to the batch normalization scale
    const fp16* __restrict__ bnorm_add,  // Pointer to the batch normalization offset
    fp16* __restrict__ output,           // Pointer to the output feature map
    fp16* __restrict__ im2col,           // Pointer to the im2col buffer
    uint32_t dim_ix,                     // Input Width
    uint32_t dim_iy,                     // Input Height
    uint32_t dim_ic,                     // Input Channels
    uint32_t dim_ox,                     // Output Width
    uint32_t dim_oy,                     // Output Height
    uint32_t dim_oc,                     // Output Channels
    uint32_t dim_fx,                     // Kernel Width
    uint32_t dim_fy,                     // Kernel Height
    uint32_t pad_t,                      // Padding Top
    uint32_t pad_b,                      // Padding Bottom
    uint32_t pad_l,                      // Padding Left
    uint32_t pad_r,                      // Padding Right
    uint32_t stride_x,                   // Stride Horizontal
    uint32_t stride_y,                   // Stride Vertical
    uint32_t apply_relu,                  // Apply ReLU activation
    uint32_t groups                      // Number of groups
) {
    spatz_conv2d_fp16_nhwc_ohwi_rvv_grouped(input, weight, bias, bnorm_mul, bnorm_add, output,  // im2col,
                              dim_ix, dim_iy, dim_ic, dim_ox, dim_oy, dim_oc, dim_fx, dim_fy, pad_t,
                              pad_b, pad_l, pad_r, stride_x, stride_y, apply_relu, groups);
}


#endif  // __spatz__