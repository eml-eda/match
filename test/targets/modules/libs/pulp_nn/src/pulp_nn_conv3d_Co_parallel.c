/*
 * --------------------------------------------------------------------------
 *  Copyright (c) 2025 Politecnico di Torino, Italy
 *  SPDX-License-Identifier: Apache-2.0
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *  
 *  http://www.apache.org/licenses/LICENSE-2.0
 *  
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 * 
 *  Author: Francesco Daghero francesco.daghero@polito.it
 * --------------------------------------------------------------------------
 */


#include <pmsis.h>
#include <pulp_nn/pulp_nn_utils.h>
#include <pulp_nn/pulp_nn_kernels.h>

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)
 
void pulp_nn_conv3d_Co_parallel(
    const uint8_t * pInBuffer,
    uint8_t *       pIm2ColBuffer,
    const int8_t *  bias,
    uint8_t *       pOutBuffer,
    const int8_t *  pWeight,
    int32_t *       k,
    int32_t *       lambda,
    const uint16_t  out_mult,
    const uint16_t  out_shift,
    const uint16_t  dim_in_x,
    const uint16_t  dim_in_y,
    const uint16_t  dim_in_d,
    const uint16_t  ch_in,
    const uint16_t  dim_out_x,
    const uint16_t  dim_out_y,
    const uint16_t  dim_out_d,
    const uint16_t  ch_out,
    const uint16_t  dim_kernel_x,
    const uint16_t  dim_kernel_y,
    const uint16_t  dim_kernel_d,
    const uint16_t  padding_y_top,
    const uint16_t  padding_y_bottom,
    const uint16_t  padding_x_left,
    const uint16_t  padding_x_right,
    const uint16_t  padding_d_front,
    const uint16_t  padding_d_back,
    const uint16_t  stride_x,
    const uint16_t  stride_y,
    const uint16_t  stride_d,
    int             flag_relu,
    int             flag_batch_norm
){
   int core_id = pi_core_id();
 
   // local vars
   int i_out_d, i_out_y, i_out_x,i_ker_d, i_ker_y, i_ker_x;
 
   int Log2Core = log2(NUM_CORES);
   /*chunks are built along the spatial dimension of the OFM */
   int chunk = (ch_out >> Log2Core) + ((ch_out & (NUM_CORES - 1)) != 0);
 
   /* defining the specific channels computed by each core */
   int start_channel, stop_channel;
   start_channel = min(chunk * core_id, ch_out);
   stop_channel = min(start_channel + chunk, ch_out);
 
   int eff_chunk = stop_channel - start_channel;
 
   uint8_t *pIm2ColBase = pIm2ColBuffer + (2 * core_id * ch_in * dim_kernel_x * dim_kernel_y * dim_kernel_d);
   uint8_t *pIm2Col = pIm2ColBase;
   int8_t *pW = pWeight + (start_channel * ch_in * dim_kernel_x * dim_kernel_y * dim_kernel_d);
 
   uint8_t *pOut = pOutBuffer + start_channel;
   int32_t *k0 = k + start_channel;
   int32_t *lambda0 = lambda + start_channel;
 
   if(eff_chunk)
   {
 
     for (i_out_d = 0; i_out_d < dim_out_d; i_out_d++) {
       for (i_out_y = 0; i_out_y < dim_out_y; i_out_y++) {
         for (i_out_x = 0; i_out_x < dim_out_x; i_out_x++) {
           if(i_out_d < padding_d_front) {
             /* Front padding region - need to handle depth padding */
             if(i_out_y < padding_y_top) {
               /* Top-left-front corner region */
               for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                 for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                   for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                     if (i_ker_d < 0 || i_ker_d >= dim_in_d || i_ker_y < 0 || i_ker_y >= dim_in_y || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                       pulp_zero_mem(pIm2Col, ch_in);
                     }
                     else {
                       pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                     }
                     pIm2Col += ch_in;
                   }
                 }
               }
             }
             else if(i_out_y < dim_out_y - padding_y_bottom) {
               /* Middle-front region */
               if(i_out_x < padding_x_left) {
                 /* Left-front region */
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                       if (i_ker_d < 0 || i_ker_d >= dim_in_d || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                         pulp_zero_mem(pIm2Col, ch_in);
                       }
                       else {
                         pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                       }
                       pIm2Col += ch_in;
                     }
                   }
                 }
               }
               else if(i_out_x < dim_out_x - padding_x_right) {
                 /* Center-front region */
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     if (i_ker_d < 0 || i_ker_d >= dim_in_d) {
                       pulp_zero_mem(pIm2Col, ch_in * dim_kernel_x);
                     }
                     else {
                       pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_out_x * stride_x - padding_x_left) * ch_in, pIm2Col, ch_in * dim_kernel_x);
                     }
                     pIm2Col += ch_in * dim_kernel_x;
                   }
                 }
               }
               else {
                 /* Right-front region */
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                       if (i_ker_d < 0 || i_ker_d >= dim_in_d || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                         pulp_zero_mem(pIm2Col, ch_in);
                       }
                       else {
                         pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                       }
                       pIm2Col += ch_in;
                     }
                   }
                 }
               }
             }
             else {
               /* Bottom-front region */
               for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                 for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                   for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                     if (i_ker_d < 0 || i_ker_d >= dim_in_d || i_ker_y < 0 || i_ker_y >= dim_in_y || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                       pulp_zero_mem(pIm2Col, ch_in);
                     }
                     else {
                       pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                     }
                     pIm2Col += ch_in;
                   }
                 }
               }
             }
           }
           else if(i_out_d < dim_out_d - padding_d_back) {
             /* Middle depth region - original 2D logic applies */
             if(i_out_y < padding_y_top) {
               /* This part implements the im2col function */
               for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                 for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                   for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                     if (i_ker_y < 0 || i_ker_y >= dim_in_y || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                       pulp_zero_mem(pIm2Col, ch_in);
                     }
                     else {
                       pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                     }
                     pIm2Col += ch_in;
                   }
                 }
               }
             }
             else if(i_out_y < dim_out_y - padding_y_bottom) {
               if(i_out_x < padding_x_left) {
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                       if (i_ker_x < 0 || i_ker_x >= dim_in_x) {
                         pulp_zero_mem(pIm2Col, ch_in);
                       }
                       else {
                         pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                       }
                       pIm2Col += ch_in;
                     }
                   }
                 }
               }
               else if(i_out_x < dim_out_x - padding_x_right) {
                 /* Optimized case - can copy entire rows and potentially entire 2D slices */
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_out_x * stride_x - padding_x_left) * ch_in, pIm2Col, ch_in * dim_kernel_x);
                     pIm2Col += ch_in * dim_kernel_x;
                   }
                 }
               }
               else
               {
                 /* This part implements the im2col function */
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                       if (i_ker_x < 0 || i_ker_x >= dim_in_x) {
                         pulp_zero_mem(pIm2Col, ch_in);
                       }
                       else {
                         pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                       }
                       pIm2Col += ch_in;
                     }
                   }
                 }
               }
             }
             else {
               for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                 for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                   for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                     if (i_ker_y < 0 || i_ker_y >= dim_in_y || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                       pulp_zero_mem(pIm2Col, ch_in);
                     }
                     else {
                       pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                     }
                     pIm2Col += ch_in;
                   }
                 }
               }
             }
           }
           else {
             /* Back padding region - need to handle depth padding */
             if(i_out_y < padding_y_top) {
               /* Top-back region */
               for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                 for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                   for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                     if (i_ker_d < 0 || i_ker_d >= dim_in_d || i_ker_y < 0 || i_ker_y >= dim_in_y || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                       pulp_zero_mem(pIm2Col, ch_in);
                     }
                     else {
                       pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                     }
                     pIm2Col += ch_in;
                   }
                 }
               }
             }
             else if(i_out_y < dim_out_y - padding_y_bottom) {
               /* Middle-back region */
               if(i_out_x < padding_x_left) {
                 /* Left-back region */
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                       if (i_ker_d < 0 || i_ker_d >= dim_in_d || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                         pulp_zero_mem(pIm2Col, ch_in);
                       }
                       else
                       {
                         pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                       }
                       pIm2Col += ch_in;
                     }
                   }
                 }
               }
               else if(i_out_x < dim_out_x - padding_x_right) {
                 /* Center-back region */
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     if (i_ker_d < 0 || i_ker_d >= dim_in_d) {
                       pulp_zero_mem(pIm2Col, ch_in * dim_kernel_x);
                     }
                     else {
                       pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_out_x * stride_x - padding_x_left) * ch_in, pIm2Col, ch_in * dim_kernel_x);
                     }
                     pIm2Col += ch_in * dim_kernel_x;
                   }
                 }
               }
               else {
                 /* Right-back region */
                 for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                   for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                     for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                       if (i_ker_d < 0 || i_ker_d >= dim_in_d || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                         pulp_zero_mem(pIm2Col, ch_in);
                       }
                       else {
                         pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                       }
                       pIm2Col += ch_in;
                     }
                   }
                 }
               }
             }
             else {
               /* Bottom-back region */
               for (i_ker_d = i_out_d * stride_d - padding_d_front; i_ker_d < i_out_d * stride_d - padding_d_front + dim_kernel_d; i_ker_d++) {
                 for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++) {
                   for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++) {
                     if (i_ker_d < 0 || i_ker_d >= dim_in_d || i_ker_y < 0 || i_ker_y >= dim_in_y || i_ker_x < 0 || i_ker_x >= dim_in_x) {
                       pulp_zero_mem(pIm2Col, ch_in);
                     }
                     else {
                       pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_d * dim_in_y * dim_in_x + i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                     }
                     pIm2Col += ch_in;
                   }
                 }
               }
             }
             
           }
 
         if (pIm2Col == pIm2ColBase + 2 * ch_in * dim_kernel_x * dim_kernel_y * dim_kernel_d)
           {
             pOut = ((ch_out - eff_chunk) << 1) + pulp_nn_matmul(
               pW,
               pIm2ColBase,
               eff_chunk,
               ch_in * dim_kernel_x * dim_kernel_y * dim_kernel_d,
               out_shift,
               out_mult,
               k0,
               lambda0,
               bias,
               pOut,
               pOut + ch_out,
               flag_relu,
               flag_batch_norm
             );
             pIm2Col = pIm2ColBase;
           }
         }
       }
     }
 
     /* check if there is left-over for compute */
     if (pIm2Col != pIm2ColBase)
     {
       const int8_t *pA = pW;
       int i;
       for (i = start_channel; i < stop_channel; i++)
       {
         int sum = 0;
 
         if (bias != NULL)
         {
           sum = ((int)(bias[i]));
         }
 
         uint8_t *pB = pIm2ColBase;
         /* basically each time it process 4 entries */
         uint16_t  col_cnt_im2col = ch_in * dim_kernel_x * dim_kernel_y * dim_kernel_d >> 2;
 
         for (int j=0 ; j < col_cnt_im2col; j++)
         {
           v4s inA = *((v4s*) pA);
           v4u inB = *((v4u*) pB);
 
           sum = SumDotp(inB, inA, sum);
           pA+=4;
           pB+=4;
         }
         col_cnt_im2col = (ch_in * dim_kernel_d * dim_kernel_y * dim_kernel_x) & 0x3;
         while (col_cnt_im2col)
         {
           int8_t      inA1 = *pA++;
           uint8_t     inB1 = *pB++;
           asm volatile("": : :"memory");
           sum += inA1 * inB1;
 
           col_cnt_im2col--;
         }
         /* if activation layer follows batch normalization */
         if (flag_batch_norm && flag_relu)
         {
           *pOut = pulp_nn_bn_quant_u8(sum, *k0, *lambda0, out_shift);
           k0++;
           lambda0++;
           pOut++;
         }
         else
         {
           /* if there isn't batch normalization but there is activation layer */
           if(flag_relu == 1)
           {
             *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
           }
           else
           {
             *pOut = (uint8_t) clip8(sum >> out_shift);
           }
           pOut++;
         }
       }
     }
   }
   pi_cl_team_barrier(0);
}


void pulp_nn_conv3d_naive(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer,
  const int8_t *  bias,
  uint8_t *       pOutBuffer,
  const int8_t *  pWeight,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  dim_in_d,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  dim_out_d,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  dim_kernel_d,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  padding_d_front,
  const uint16_t  padding_d_back,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  const uint16_t  stride_d,
  int             flag_relu,
  int             flag_batch_norm
){
  int core_id = pi_core_id();
  if(!core_id) {
      printf("pulp_nn_conv3d_Co_parallel: dim_in_d=%d, dim_in_y=%d, dim_in_x=%d, ch_in=%d, dim_out_d=%d, dim_out_y=%d, dim_out_x=%d, ch_out=%d, dim_kernel_d=%d, dim_kernel_y=%d, dim_kernel_x=%d\n",
          dim_in_d, dim_in_y, dim_in_x, ch_in,
          dim_out_d, dim_out_y, dim_out_x, ch_out,
          dim_kernel_d, dim_kernel_y, dim_kernel_x);
      printf("pulp_nn_conv3d_Co_parallel: padding_y_top=%d, padding_y_bottom=%d, padding_x_left=%d, padding_x_right=%d, padding_d_front=%d, padding_d_back=%d\n",
          padding_y_top, padding_y_bottom,
          padding_x_left, padding_x_right,
          padding_d_front, padding_d_back);
      printf("pulp_nn_conv3d_Co_parallel: stride_y=%d, stride_x=%d, stride_d=%d\n",
          stride_y, stride_x, stride_d);
      printf("pulp_nn_conv3d_Co_parallel: flag_relu=%d, flag_batch_norm=%d\n",
          flag_relu, flag_batch_norm);
      printf("pulp_nn_conv3d_Co_parallel: out_mult=%d, out_shift=%d\n",
          out_mult, out_shift);
      printf("pulp_nn_conv3d_Co_parallel: first 4 weights [%d %d %d %d] first 4 acts [%d %d %d %d] first 4 scales [%d %d %d %d] first 4 biases [%d %d %d %d]\n",
          pWeight[0], pWeight[1], pWeight[2], pWeight[3],
          pInBuffer[0], pInBuffer[1], pInBuffer[2], pInBuffer[3],
          k[0], k[1], k[2], k[3],
          lambda[0], lambda[1], lambda[2], lambda[3]);
      // Initialize the L1 scratchpad memory
      for(int out_depth_idx=0; out_depth_idx<dim_out_d; out_depth_idx++) {
          for(int out_y_idx=0; out_y_idx<dim_out_y; out_y_idx++) {
              for(int out_x_idx=0; out_x_idx<dim_out_x; out_x_idx++) {
                  for(int out_ch_idx=0; out_ch_idx<ch_out; out_ch_idx++) {
                      int out_scale = k[out_ch_idx];
                      int out_bias = lambda[out_ch_idx];
                      int res = 0;
                      for(int kernel_d=0; kernel_d<dim_kernel_d; kernel_d++) {
                          int inp_depth_idx = out_depth_idx * stride_d - padding_d_front + kernel_d;
                          if(inp_depth_idx < 0 || inp_depth_idx >= dim_in_d) {
                              continue;
                          }
                          for(int kernel_y=0; kernel_y<dim_kernel_y; kernel_y++) {
                              int inp_y_idx = out_y_idx * stride_y - padding_y_top + kernel_y;
                              if(inp_y_idx < 0 || inp_y_idx >= dim_in_y) {
                                  continue;
                              }
                              for(int kernel_x=0; kernel_x<dim_kernel_x; kernel_x++) {
                                  int inp_x_idx = out_x_idx * stride_x - padding_x_left + kernel_x;
                                  if(inp_x_idx < 0 || inp_x_idx >= dim_in_x) {
                                      continue;
                                  }
                                  for(int in_ch_idx=0; in_ch_idx<ch_in; in_ch_idx++) {
                                      uint8_t in_val = pInBuffer[
                                          inp_depth_idx * dim_in_y * dim_in_x * ch_in +
                                          inp_y_idx * dim_in_x * ch_in + inp_x_idx * ch_in + in_ch_idx
                                      ];
                                      int8_t weight_val = pWeight[
                                          out_ch_idx * dim_kernel_d * dim_kernel_y * dim_kernel_x * ch_in +
                                          kernel_d * dim_kernel_y * dim_kernel_x * ch_in +
                                          kernel_y * dim_kernel_x * ch_in +
                                          kernel_x * ch_in + in_ch_idx
                                      ];
                                      // printf("Processing: out_depth_idx=%d, out_y_idx=%d, out_x_idx=%d, out_ch_idx=%d, inp_depth_idx=%d, inp_y_idx=%d, inp_x_idx=%d, in_ch_idx=%d, in_val=%d, weight_val=%d mult should be %d res was %d so now should be %d\n",
                                          // out_depth_idx, out_y_idx, out_x_idx, out_ch_idx,
                                          // inp_depth_idx, inp_y_idx, inp_x_idx, in_ch_idx,
                                          // in_val, weight_val, in_val * weight_val,
                                          // res, res + in_val * weight_val);
                                      res += in_val * weight_val;
                                  }
                              }
                          }
                      }
                      // printf("Final result for out_depth_idx=%d, out_y_idx=%d, out_x_idx=%d, out_ch_idx=%d is %d\n",
                          // out_depth_idx, out_y_idx, out_x_idx, out_ch_idx, res);
                      res *= out_scale;
                      // printf("After scaling by %d, result is %d\n", out_scale, res);
                      res += out_bias;
                      // printf("After adding bias %d, result is %d\n", out_bias, res);
                      res = res >> out_shift;
                      // printf("After shifting by %d, result is %d\n", out_shift, res);
                      res = res > 0 ? res : 0; // ReLU activation
                      // printf("After ReLU, result is %d\n", res);
                      int out_idx = out_depth_idx * dim_out_y * dim_out_x * ch_out +
                          out_y_idx * dim_out_x * ch_out +
                          out_x_idx * ch_out + out_ch_idx;
                      pOutBuffer[out_idx] = (uint8_t) res;
                  }
              }
          }
      }
  }
  pi_cl_team_barrier(0);
}