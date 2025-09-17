#ifdef __spatz__
#ifndef __SPATZ_KERNELS_FP16_KERNELS_H__
#define __SPATZ_KERNELS_FP16_KERNELS_H__

#include <stddef.h>
#include <stdint.h>

#include <spatz_kernels/spatz_fp16_defines.h>


void spatz_fp16_gemm(
  const fp16 *__restrict__ input,   // M x N
  const fp16 *__restrict__ weight,  // N x K
  const fp16 *__restrict__ bias,    // M x K
  fp16 *__restrict__ output,        // M x K
  uint32_t dim_m,
  uint32_t dim_n,
  uint32_t dim_k
);


void spatz_fp16_matmul(
  const fp16 *__restrict__ input,   // M x N
  const fp16 *__restrict__ weight,  // N x K
  fp16 *__restrict__ output,        // M x K
  uint32_t dim_m,
  uint32_t dim_n,
  uint32_t dim_k
);


void spatz_fp16_linear(
    fp16 *__restrict__ input,
    fp16 *__restrict__ weight,
    fp16 *__restrict__ output,
    fp16 *__restrict__ bias,
    uint32_t dim_i,
    uint32_t dim_o
);

void spatz_fp16_conv2d(
    const fp16 *__restrict__ input,         // Pointer to the input feature map
    const fp16 *__restrict__ weight,        // Pointer to the weights
    const fp16 *__restrict__ bias,          // Pointer to the bias vector
    const fp16 *__restrict__ bnorm_mul,     // Pointer to the batch normalization scale
    const fp16 *__restrict__ bnorm_add,     // Pointer to the batch normalization offset
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
);


fp16 spatz_fp16_fdotp(
    const fp16* a, 
    const fp16* b, 
    unsigned int avl
);

#endif // __SPATZ_KERNELS_FP16_KERNELS_H__
#endif // __spatz__