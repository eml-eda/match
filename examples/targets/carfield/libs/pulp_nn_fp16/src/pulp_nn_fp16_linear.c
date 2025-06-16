#ifdef __pulp_cluster__


#include <stdint.h>

#include <pulp.h>

#include "pulp_nn_fp16/pulp_nn_fp16_kernels.h"
#include "pulp_nn_fp16/pulp_nn_fp16_defines.h"


void pulp_nn_fp16_linear(
    fp16 *__restrict__ input,
    fp16 *__restrict__ weight,
    fp16 *__restrict__ output,
    fp16 *__restrict__ bias,
    uint32_t dim_i,
    uint32_t dim_o
)
{
    int chunk = (dim_o + nthreads - 1) / nthreads;
    int start = min(chunk * tid, dim_o);
    int stop = min(start + chunk, dim_o);

    for (int j = start; j < stop; j++) {
        fp16 sum = bias ? bias[j] : 0.0f;
        for (int k = 0; k < dim_i; k++) {
            sum += input[k] * weight[j * dim_i + k];
        }
        output[j] = sum;
    }
}


#endif