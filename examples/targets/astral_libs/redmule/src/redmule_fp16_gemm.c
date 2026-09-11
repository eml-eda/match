#ifdef __pulp_cluster__

#include <stdint.h>

#include "pulp.h"

#include "redmule/redmule_arch.h"
#include "redmule/redmule_hal.h"
#include "redmule/redmule_kernels.h"
#include "redmule/redmule_defines.h"


void redmule_fp16_gemm_async(
    const fp16 *__restrict__ x,
    const fp16 *__restrict__ w,
    fp16 *__restrict__ yz, 
    uint16_t dim_m,
    uint16_t dim_n,
    uint16_t dim_k
) {
    if (tid != 0) return; 
    
    redmule_init();

    redmule_cfg(x, w, yz, dim_m, dim_n, dim_k, (uint8_t)GEMM, (uint8_t)Float16); 

    redmule_start();
}


void redmule_fp16_gemm(
    const fp16 *__restrict__ x,
    const fp16 *__restrict__ w,
    fp16 *__restrict__ yz, 
    uint16_t dim_m,
    uint16_t dim_n,
    uint16_t dim_k
) {
    if (tid != 0) return; 
    
    redmule_fp16_gemm_async(x, w, yz, dim_m, dim_n, dim_k);

    redmule_wait();
}

#endif // __pulp_cluster__