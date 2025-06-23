#ifdef __pulp_cluster__
#ifndef __PULP_KERNELS_FP16_KERNELS_H__
#define __PULP_KERNELS_FP16_KERNELS_H__

#include <stdint.h>

#include <pulp_kernels/pulp_fp16_defines.h>


void pulp_fp16_gemm(
  const fp16 *__restrict__ input,           // Pointer to the input matrix  (M x K)
  const fp16 *__restrict__ weight,          // Pointer to the weight matrix (K x N)
  const fp16 *__restrict__ bias,            // Pointer to the bias matrix   (M x N)
  fp16 *__restrict__ output,                // Pointer to the output matrix (M x N)
  uint32_t dim_m,                           // M
  uint32_t dim_n,                           // N
  uint32_t dim_k                            // K
);

void pulp_fp16_linear(
  fp16 *__restrict__ input,                 // Pointer to the input vector  (1 x K)
  fp16 *__restrict__ weight,                // Pointer to the weight matrix (K x N)
  fp16 *__restrict__ output,                // Pointer to the output vector (1 x N)
  fp16 *__restrict__ bias,                  // Pointer to the bias vector   (1 x N)
  uint32_t dim_i,                           // Input dimension  (K)
  uint32_t dim_o                            // Output dimension (N)
);

void pulp_fp16_add(
    fp16 *__restrict__ input_a,             // Pointer to the first input vector  (length)
    fp16 *__restrict__ input_b,             // Pointer to the second input vector (length)
    fp16 *__restrict__ output,              // Pointer to the output vector       (length)
    uint32_t length                         // Length of the vectors
);

void pulp_fp16_conv2d(
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
);


void pulp_fp16_im2col(
    const fp16 *__restrict__ input,         
    fp16 *__restrict__ im2col,
    uint32_t dim_ix,                        // Input Width
    uint32_t dim_iy,                        // Input Height
    uint32_t dim_ic,                        // Input Channels
    uint32_t dim_ox,                        // Output Width
    uint32_t dim_oy,                        // Output Height
    uint32_t dim_fx,                        // Kernel Width
    uint32_t dim_fy,                        // Kernel Height
    uint32_t pad_t,                         // Padding Top
    uint32_t pad_l,                         // Padding Left
    uint32_t stride_x,                      // Stride Horizontal
    uint32_t stride_y                       // Stride Vertical
);


void pulp_fp16_avgpool2d(
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
);

void pulp_fp16_copy(
    const fp16 *input,
    fp16 *output,
    uint32_t length
);


#endif // __PULP_KERNELS_FP16_KERNELS_H__
#endif // __pulp_cluster__