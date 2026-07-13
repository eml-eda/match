#ifdef __pulp_cluster__

#include <stdint.h>

#include <pulp.h>

#include "pulp_kernels/pulp_fp16_kernels.h"
#include "pulp_kernels/pulp_fp16_defines.h"


// Assume (M x N) * (N x K) = (M x K)
void pulp_fp16_gemm(
  const fp16 *__restrict__ input,   // M x N
  const fp16 *__restrict__ weight,  // N x K
  const fp16 *__restrict__ bias,    // M x K
  fp16 *__restrict__ output,        // M x K
  uint32_t dim_m,
  uint32_t dim_n,
  uint32_t dim_k
) {
    int chunk = (dim_k + nthreads - 1) / nthreads;
    int start = min(chunk * tid, dim_k);
    int stop = min(start + chunk, dim_k);

    for (int k = start; k < stop; ++k) {
        for (int m = 0; m < dim_m; ++m) {
            fp16 sum = bias ? bias[m * dim_k + k] : 0;
            for (int n = 0; n < dim_n; ++n) {
                sum += input[m * dim_n + n] * weight[n * dim_k + k];
            }
            output[m * dim_k + k] = sum;
        }
    }
}

#endif