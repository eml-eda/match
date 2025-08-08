#ifndef __PULP_CLUSTER_ODL_LIB_H__
#define __PULP_CLUSTER_ODL_LIB_H__

#include <pulp_cluster/match_dev.h>

#define ODL_MIN(a,b) ((a)<(b)?(a):(b))

void odl_naive_parallel_conv2d_transpose_fp32(void* args);

void odl_naive_parallel_conv2d_bw_fp32(void* args);

void odl_optimized_parallel_conv2d_bw_dw_fp32(void* args);

void odl_optimized_conv2d_bw_fp32_im2col(void* args);

void odl_fast_parallel_conv2d_bw_fp32(void* args);

void odl_fast_conv2d_bw_fp32_im2col(void* args);
// naive version seems better for stride 1
void odl_naive_parallel_conv2d_transpose_stride_1_fp32(void* args);

void odl_naive_parallel_conv2d_transpose_stride_2_fp32(void* args);

void odl_naive_parallel_conv2d_transpose_pw_stride_1_fp32(void* args);

void odl_optimized_conv2d_transpose_3x3_stride1_fp32_im2col(void* args);

void odl_optimized_conv2d_transpose_3x3_stride2_fp32_im2col(void* args);

void odl_optimized_dilation2_conv2d_fp32(void* args);

#endif // __PULP_CLUSTER_ODL_LIB_H__