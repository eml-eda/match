#ifdef __pulp_cluster__

#include "pulp_nn_fp16/pulp_nn_kernels_fp16.h"

#include <stdint.h>

#include <pulp.h>

#define log2(x)   __builtin_pulp_fl1(x)
#define min(a,b)  ((a)<(b)?(a):(b))


void pulp_nn_linear_fp16(
    float16 *__restrict__ input,
    float16 *__restrict__ weight,
    float16 *__restrict__ output,
    float16 *__restrict__ bias,
    uint32_t dim_i,
    uint32_t dim_o
)
{
    const int NUM_CORES = get_core_num();

    int chunk = (dim_o >> log2(NUM_CORES)) + ((dim_o & (NUM_CORES - 1)) != 0);
    int start = min(chunk * rt_core_id(), dim_o);
    int stop = min(start + chunk, dim_o);

    for (int j = start; j < stop; j++) {
        float16 sum = bias ? bias[j] : 0;
        for (int k = 0; k < dim_i; k++) {
            sum += input[k] * weight[j * dim_i + k];
        }
        output[j] = sum;
    }
}

#endif