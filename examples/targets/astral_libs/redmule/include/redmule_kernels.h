#ifdef __pulp_cluster__
#ifndef __REDMULE_KERNELS_H__
#define __REDMULE_KERNELS_H__

#include "redmule/redmule_arch.h"
#include "redmule/redmule_hal.h"
#include "redmule/redmule_defines.h"


void redmule_fp16_gemm_async(
    const fp16 *__restrict__ x,
    const fp16 *__restrict__ w,
    fp16 *__restrict__ yz,
    uint16_t dim_m,
    uint16_t dim_n,
    uint16_t dim_k
);


void redmule_fp16_gemm(
    const fp16 *__restrict__ x,
    const fp16 *__restrict__ w,
    fp16 *__restrict__ yz,
    uint16_t dim_m,
    uint16_t dim_n,
    uint16_t dim_k
);


void redmule_fp16_matmul_async(
    const fp16 *__restrict__ x,
    const fp16 *__restrict__ w,
    fp16 *__restrict__ z, 
    uint16_t dim_m,
    uint16_t dim_n,
    uint16_t dim_k
);


void redmule_fp16_matmul(
    const fp16 *__restrict__ x,
    const fp16 *__restrict__ w,
    fp16 *__restrict__ z, 
    uint16_t dim_m,
    uint16_t dim_n,
    uint16_t dim_k
);

void redmule_fp16_conv3d_rd(
    const fp16  *input,       // Input 3D tensor (Shape: C_in x D x H x W)
    const fp16  *weights,     // Pre-flattened Weight Matrix (Shape: N x K)
    fp16  *output_matrix,     // Destination Matrix for GEMM result (Shape: M x K)
    fp16  *im2col_buf,        // Allocated L1 TCDM buffer for vol2col (Shape: M x N)
    const fp16  *bias,              // Bias vector (Shape: K)
    int apply_relu,
    int c_in, int d_in, int h_in, int w_in,
    int c_out, int d_out, int h_out, int w_out,
    int kd, int kh, int kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d_low, int pad_d_high,   // Depth padding (front, back)
    int pad_h_low, int pad_h_high,   // Height padding (top, bottom)
    int pad_w_low, int pad_w_high    // Width padding (left, right)
);

void redmule_fp16_conv3d_dhwn_rd(
    const fp16  *input,       // Input 3D tensor (Shape: D x H x W x C_in)
    const fp16  *weights,     // Pre-flattened Weight Matrix (Shape: K x N) -> [C_out x (Dk*Hk*Wk*C_in)]
    fp16  *output_matrix,     // Destination Matrix for GEMM result (Shape: M x K) -> [M x C_out]
    fp16  *im2col_buf,        // Allocated L1 TCDM buffer for vol2col (Shape: M x N) -> [M x (Dk*Hk*Wk*C_in)]
    const fp16  *bias,        // Bias vector (Shape: K) -> [C_out]
    int apply_relu,
    int c_in, int d_in, int h_in, int w_in,
    int c_out, int d_out, int h_out, int w_out,
    int kd, int kh, int kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d_low, int pad_d_high,   // Depth padding (front, back)
    int pad_h_low, int pad_h_high,   // Height padding (top, bottom)
    int pad_w_low, int pad_w_high    // Width padding (left, right)
);

void redmule_fp16_conv3d_pc(
    const fp16  *input,       // Input 3D tensor (Shape: C_in x D x H x W)
    const fp16  *weights,     // Pre-flattened Weight Matrix (Shape: N x K)
    fp16  *output_matrix,     // Destination Matrix for GEMM result (Shape: M x K)
    fp16  *im2col_buf,        // Allocated L1 TCDM buffer for vol2col (Shape: M x N)
    const fp16  *bias,              // Bias vector (Shape: K)
    int apply_relu,
    uint16_t c_in, uint16_t d_in, uint16_t h_in, uint16_t w_in,
    uint16_t c_out, uint16_t d_out, uint16_t h_out, uint16_t w_out,
    uint16_t kd, uint16_t kh, uint16_t kw,
    uint16_t stride_d, uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_d_low, uint16_t pad_d_high,   // Depth padding (front, back)
    uint16_t pad_h_low, uint16_t pad_h_high,   // Height padding (top, bottom)
    uint16_t pad_w_low, uint16_t pad_w_high    // Width padding (left, right)
);

#endif // __REDMULE_KERNELS_H__
#endif // __pulp_cluster__