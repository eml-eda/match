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

    for (int o = start; o < stop; o++) {
        fp16 sum = bias ? bias[o] : 0.0f;
        for (int i = 0; i < dim_i; i++) {
            int weight_idx = idx_IO(i, o, dim_i, dim_o); // CN in TVM
            sum += input[i] * weight[weight_idx];
        }
        output[o] = sum;
    }
}


#endif