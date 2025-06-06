/*
 * pulp_nn_dw_linear_out_32.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef CLUSTER_COMPILATION

#include <pulp.h>
#include <pulp_nn/pulp_nn_utils.h>

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)

void pulp_nn_linear_out_32(
  uint8_t *pInBuffer,
  int8_t *bias,
  int32_t *pOutBuffer,
  int8_t *pWeights,
  uint16_t dim_vec,
  uint16_t num_o_neurons
)
{
  int core_id = rt_core_id();
  int Log2Core = log2(get_core_num());
  int chunk = (num_o_neurons >> Log2Core) + ((num_o_neurons & (get_core_num()-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  v4u vecA;
  v4s vecB;

  int32_t *pOut = (int32_t *) pOutBuffer + start;

  for(int i=start; i<stop; i++)
  {
    int sum = 0;

    if (bias != NULL)
    {
      sum = *(int32_t *)(bias + 4*i);
    }

    uint8_t *pA = pInBuffer;
    int8_t *pB = pWeights + (i * dim_vec);

    for (int j=0; j<(dim_vec >> 2); j++)
    {
      vecA = *((v4u*)pA);
      vecB = *((v4s*)pB);
      sum = SumDotp(vecA, vecB, sum);
      pA+=4;
      pB+=4;
    }
    uint16_t col_cnt = dim_vec & 0x3;
    while (col_cnt)
    {
      uint8_t inA = *pA;
      pA++;
      int8_t inB = *pB;
      pB++;
      sum += inA * inB;
      col_cnt--;
    }
    *pOut = sum;
    pOut++;
  }
}

#endif