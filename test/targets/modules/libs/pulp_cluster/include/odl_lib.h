#ifndef __PULP_CLUSTER_ODL_LIB_H__
#define __PULP_CLUSTER_ODL_LIB_H__

#include <pulp_cluster/match_dev.h>

#define ODL_MIN(a,b) ((a)<(b)?(a):(b))

// #define ODL_USE_UNSAFE_FAST_MATH

#define odl_max_f32(a, b)	__builtin_pulp_f32max((a), (b))
#define odl_min_f32(a, b)	__builtin_pulp_f32min((a), (b))

static inline float odl_clip_f32(float x, float lower, float upper) {
    return (float) odl_min_f32(odl_max_f32(x, lower), upper);
}

static inline  float odl_faster_pow2 (float p)
{
    float clipp = odl_clip_f32(p, -126.0f, 126.0f);
    union { unsigned int i; float f; } v = { (unsigned int) ( (1 << 23) * (clipp + 126.94269504f) ) };
    return v.f;
}

static inline float odl_fast_rsqrt(float x)
{
    const float threehalfs = 1.5F;
    float x2 = x * 0.5F;
    float y  = x;
    int i  = * ( int * ) &y;
    i  = 0x5f375a86 - ( i >> 1 );
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );

    return y;
}

void odl_naive_parallel_conv2d_transpose_fp32(void* args);

// not used because of poor performances, but kept for reference
void odl_naive_parallel_conv2d_bw_fp32(void* args);

void odl_optimized_parallel_conv2d_bw_dw_fp32(void* args);

// older version of bw im2col, not used because of not optimal performance
// void odl_optimized_conv2d_bw_fp32_im2col(void* args);

void odl_fast_parallel_conv2d_bw_fp32(void* args);

void odl_fast_conv2d_bw_fp32_im2col(void* args);
// naive version seems better for stride 1
// void odl_naive_parallel_conv2d_transpose_stride_1_fp32(void* args);

void odl_naive_parallel_conv2d_transpose_stride_2_fp32(void* args);

void odl_naive_parallel_conv2d_transpose_pw_stride_1_fp32(void* args);

// not used because of poor performances
// void odl_optimized_conv2d_transpose_3x3_stride1_fp32_im2col(void* args);
// not used because of poor performances
// void odl_optimized_conv2d_transpose_3x3_stride2_fp32_im2col(void* args);

void odl_bw_instance_norm_tail_fp32(void* args);

void odl_fw_instance_norm_tail_fp32(void* args);

#endif // __PULP_CLUSTER_ODL_LIB_H__