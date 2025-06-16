#ifdef __pulp_cluster__

#include <stdint.h>
#include <stddef.h>

#include <pulp.h>

#include "pulp_nn_fp16/pulp_nn_fp16_defines.h"
#include "pulp_nn_fp16/pulp_nn_fp16_kernels.h"


void __attribute__ ((noinline))  pulp_nn_fp16_add(
    fp16 *__restrict__ input_a,
    fp16 *__restrict__ input_b,
    fp16 *__restrict__ output,
    uint32_t length
) {
    uint32_t chunk = (length + nthreads - 1) / nthreads;
    uint32_t start = min(tid * chunk, length);
    uint32_t end = min(start + chunk, length);

    for (uint32_t i = start; i < end; i++) {
        output[i] = input_a[i] + input_b[i];
    }
}


#endif // __pulp_cluster__
