#ifdef __spatz__

#include <stddef.h>
#include <stdint.h>

#include "spatz_kernels/spatz_fp16_kernels.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

/* Kernel Implementations */


// Assume (M x N) * (N x K) = (M x K)
// dim_k has to be even
static void spatz_fp16_gemm_mn_nk_mk_rvv(
  const fp16 *__restrict__ input,   // M x N
  const fp16 *__restrict__ weight,  // N x K
  const fp16 *__restrict__ bias,    // M x K
  fp16 *__restrict__ output,        // M x K
  uint32_t dim_m,
  uint32_t dim_n,
  uint32_t dim_k
) {

    int m_chunk = (dim_m + nthreads - 1) / nthreads;
    int m_start = min(tid * m_chunk, dim_k);
    int m_end = min(m_start + m_chunk, dim_k);

    if (dim_m <= 2) {
        if (tid != 0) goto sync;
        m_start = 0;
        m_end = dim_m;
    }

    uint32_t vl = 0;
    fp16 a;
    
    for (int m = m_start; m < m_end; ++m) {
        for (int k = 0; k < dim_k; k += vl) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(dim_k - k));

            asm volatile("vle16.v v8, (%0)"  :: "r"(&weight[0 * dim_k + k]) : "v8",  "memory");
            a = input[m * dim_n + 0];

            if (bias) {
                asm volatile("vle16.v v16, (%0)"  :: "r"(&bias[m * dim_k + k]) : "v16",  "memory");
                asm volatile("vfmacc.vf v16, %0, v8" ::"f"(a) : "v16", "v8");
            } else {
                asm volatile("vfmul.vf v16, v8, %0" ::"f"(a) : "v16", "v8");
            }

            for (int n = 1; n < dim_n; ++n) {
                a = input[m * dim_n + n];
                asm volatile("vle16.v v8, (%0)"  :: "r"(&weight[n * dim_k + k]) : "v8",  "memory");
                asm volatile("vfmacc.vf v16, %0, v8" ::"f"(a) : "v16", "v8");
            }

            asm volatile("vse16.v v16, (%0);" ::"r"(&output[m * dim_k + k]) : "v16", "memory");
        }
    }

    sync:
    barrier();

    return;
}


/* Kernel APIs */


void spatz_fp16_gemm(
  const fp16 *__restrict__ input,   // M x N
  const fp16 *__restrict__ weight,  // N x K
  const fp16 *__restrict__ bias,    // M x K
  fp16 *__restrict__ output,        // M x K
  uint32_t dim_m,
  uint32_t dim_n,
  uint32_t dim_k
) {
    spatz_fp16_gemm_mn_nk_mk_rvv(
        input,
        weight,
        bias,
        output,
        dim_m,
        dim_n,
        dim_k
    );
}


void spatz_fp16_matmul(
  const fp16 *__restrict__ input,   // M x N
  const fp16 *__restrict__ weight,  // N x K
  fp16 *__restrict__ output,        // M x K
  uint32_t dim_m,
  uint32_t dim_n,
  uint32_t dim_k
) {
    spatz_fp16_gemm_mn_nk_mk_rvv(
        input,
        weight,
        NULL,
        output,
        dim_m,
        dim_n,
        dim_k
    );
}


void spatz_fp16_linear(
    fp16 *__restrict__ input,
    fp16 *__restrict__ weight,
    fp16 *__restrict__ output,
    fp16 *__restrict__ bias,
    uint32_t dim_i,
    uint32_t dim_o
) {
    spatz_fp16_gemm_mn_nk_mk_rvv(
        input,
        weight,
        bias,
        output,
        1,
        dim_i,
        dim_o
    );
}

#endif  // __spatz__