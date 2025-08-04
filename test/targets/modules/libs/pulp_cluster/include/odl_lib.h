#ifndef __PULP_CLUSTER_ODL_LIB_H__
#define __PULP_CLUSTER_ODL_LIB_H__

#include <pulp_cluster/match_dev.h>

void odl_naive_parallel_conv2d_transpose_fp32(void* args);

void odl_naive_parallel_conv2d_bw_fp32(void* args);

void odl_optimized_conv2d_bw_fp32_im2col(void* args);

#endif // __PULP_CLUSTER_ODL_LIB_H__