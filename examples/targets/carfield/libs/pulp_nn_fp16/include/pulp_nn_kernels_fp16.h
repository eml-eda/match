#ifdef __pulp_cluster__

#include <stdint.h>

void pulp_nn_linear_fp16(
  float16 *__restrict__ input,
  float16 *__restrict__ weight,
  float16 *__restrict__ output,
  float16 *__restrict__ bias,
  uint32_t dim_i,
  uint32_t dim_o
);

#endif