#ifdef __pulp_cluster__
#ifndef __PULP_NN_FP16_KERNELS_H__
#define __PULP_NN_FP16_KERNELS_H__

#include <stdint.h>

#include <pulp_nn_fp16/pulp_nn_fp16_defines.h>


void pulp_nn_fp16_gemm(
  const fp16 *__restrict__ input,           // Pointer to the input matrix  (M x K)
  const fp16 *__restrict__ weight,          // Pointer to the weight matrix (K x N)
  const fp16 *__restrict__ bias,            // Pointer to the bias matrix   (M x N)
  fp16 *__restrict__ output,                // Pointer to the output matrix (M x N)
  uint32_t dim_m,                           // M
  uint32_t dim_n,                           // N
  uint32_t dim_k                            // K
);

void pulp_nn_fp16_linear(
  fp16 *__restrict__ input,                 // Pointer to the input vector  (1 x K)
  fp16 *__restrict__ weight,                // Pointer to the weight matrix (K x N)
  fp16 *__restrict__ output,                // Pointer to the output vector (1 x N)
  fp16 *__restrict__ bias,                  // Pointer to the bias vector   (1 x N)
  uint32_t dim_i,                           // Input dimension  (K)
  uint32_t dim_o                            // Output dimension (N)
);

void pulp_nn_fp16_add(
    fp16 *__restrict__ input_a,             // Pointer to the first input vector  (length)
    fp16 *__restrict__ input_b,             // Pointer to the second input vector (length)
    fp16 *__restrict__ output,              // Pointer to the output vector       (length)
    uint32_t length                         // Length of the vectors
);

void pulp_nn_fp16_conv2d(
    const fp16 *__restrict__ input,         // Pointer to the input feature map   (dim_w x dim_h x dim_c)
    fp16 *__restrict__       im2col,        // Pointer to the im2col buffer       
    const fp16 *__restrict__ bias,          // Pointer to the bias matrix         (dim_p x dim_q x dim_k)
    fp16 *__restrict__       output,        // Pointer to the output feature map  (dim_p x dim_q x dim_k)
    const fp16 *__restrict__ weight,        // Pointer to the weights             (dim_r x dim_s x dim_c x dim_k)
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
);

void pulp_nn_fp16_avgpool(
    const fp16 *input,
    fp16 *output,
    uint16_t dim_h, uint16_t dim_w, uint16_t dim_c,
    uint16_t dim_p, uint16_t dim_q,
    uint16_t dim_r, uint16_t dim_s,
    uint16_t pad_t, uint16_t pad_b,
    uint16_t pad_l, uint16_t pad_r,
    uint16_t stride_y, uint16_t stride_x
);


void pulp_nn_fp16_copy(
    const fp16 *input,
    fp16 *output,
    uint32_t size
);

#endif // __PULP_NN_FP16_KERNELS_H__
#endif // __pulp_cluster__