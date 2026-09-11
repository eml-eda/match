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

    redmule_config(x, w, yz, dim_m, dim_n, dim_k, REDMULE_OP_GEMM, REDMULE_OP_FMT_FP16); 

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