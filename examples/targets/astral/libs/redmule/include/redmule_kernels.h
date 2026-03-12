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

#endif // __REDMULE_KERNELS_H__
#endif // __pulp_cluster__