// tvm target: c -keys=arm_cpu,cpu -device=arm_cpu
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
static const int32_t __attribute__((section(".rodata.tvm"), aligned(16))) fused_nn_conv2d_constant[1] = {
    +0x58e57d67
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const int8_t __attribute__((section(".rodata.tvm"), aligned(16))) fused_constant[27] = {
    +0x2c, -0x76, -0x01, +0x0c, -0x51, +0x2a, +0x44, +0x17, 
    -0x0b, +0x26, -0x6a, +0x37, +0x40, +0x4c, -0x5f, +0x58, 
    -0x3d, +0x33, -0x32, +0x1a, +0x7b, -0x2e, +0x22, +0x5b, 
    +0x43, -0x0a, -0x03
};
#ifdef __cplusplus
}  // extern "C"
#endif
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_cpuonly_fused_nn_conv2d_add_nn_relu(int8_t* p0, int32_t* T_relu);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_cpuonly___tvm_main__(int8_t* input_0_buffer_var, int32_t* output_buffer_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_cpuonly_fused_nn_conv2d_add_nn_relu(int8_t* p0, int32_t* T_relu) {
  void* data_vec = TVMBackendAllocWorkspace(1, 0, (uint64_t)4800, 0, 8);
  if (data_vec == NULL) {
    return -1;
  }
  int32_t conv[4];
  for (int32_t h = 0; h < 8; ++h) {
    for (int32_t w = 0; w < 8; ++w) {
      for (int32_t ci = 0; ci < 3; ++ci) {
        for (int32_t vh = 0; vh < 5; ++vh) {
          for (int32_t vw = 0; vw < 5; ++vw) {
            int32_t cse_var_1 = (w * 4);
            ((int8_t*)data_vec)[(((((h * 600) + (w * 75)) + (ci * 25)) + (vh * 5)) + vw)] = (((1 <= ((h * 4) + vh)) && (1 <= (cse_var_1 + vw))) ? p0[((((((ci * 1024) + (h * 128)) + (vh * 32)) + cse_var_1) + vw) - 33)] : (int8_t)0);
          }
        }
      }
    }
  }
  for (int32_t ax2_outer = 0; ax2_outer < 8; ++ax2_outer) {
    for (int32_t ax3_outer = 0; ax3_outer < 8; ++ax3_outer) {
      for (int32_t vw_init = 0; vw_init < 2; ++vw_init) {
        conv[vw_init] = 0;
      }
      for (int32_t vw_init_1 = 0; vw_init_1 < 2; ++vw_init_1) {
        conv[(vw_init_1 + 2)] = 0;
      }
      for (int32_t ci_1 = 0; ci_1 < 3; ++ci_1) {
        for (int32_t vw_1 = 0; vw_1 < 2; ++vw_1) {
          conv[vw_1] = (conv[vw_1] + (((int32_t)((int8_t*)data_vec)[((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_1 * 2))]) * ((int32_t)((int8_t*)fused_constant)[(ci_1 * 9)])));
        }
        for (int32_t vw_2 = 0; vw_2 < 2; ++vw_2) {
          int32_t cse_var_2 = (vw_2 + 2);
          conv[cse_var_2] = (conv[cse_var_2] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_2 * 2)) + 10)]) * ((int32_t)((int8_t*)fused_constant)[(ci_1 * 9)])));
        }
        for (int32_t vw_3 = 0; vw_3 < 2; ++vw_3) {
          conv[vw_3] = (conv[vw_3] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_3 * 2)) + 1)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 1)])));
        }
        for (int32_t vw_4 = 0; vw_4 < 2; ++vw_4) {
          int32_t cse_var_3 = (vw_4 + 2);
          conv[cse_var_3] = (conv[cse_var_3] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_4 * 2)) + 11)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 1)])));
        }
        for (int32_t vw_5 = 0; vw_5 < 2; ++vw_5) {
          conv[vw_5] = (conv[vw_5] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_5 * 2)) + 2)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 2)])));
        }
        for (int32_t vw_6 = 0; vw_6 < 2; ++vw_6) {
          int32_t cse_var_4 = (vw_6 + 2);
          conv[cse_var_4] = (conv[cse_var_4] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_6 * 2)) + 12)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 2)])));
        }
        for (int32_t vw_7 = 0; vw_7 < 2; ++vw_7) {
          conv[vw_7] = (conv[vw_7] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_7 * 2)) + 5)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 3)])));
        }
        for (int32_t vw_8 = 0; vw_8 < 2; ++vw_8) {
          int32_t cse_var_5 = (vw_8 + 2);
          conv[cse_var_5] = (conv[cse_var_5] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_8 * 2)) + 15)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 3)])));
        }
        for (int32_t vw_9 = 0; vw_9 < 2; ++vw_9) {
          conv[vw_9] = (conv[vw_9] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_9 * 2)) + 6)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 4)])));
        }
        for (int32_t vw_10 = 0; vw_10 < 2; ++vw_10) {
          int32_t cse_var_6 = (vw_10 + 2);
          conv[cse_var_6] = (conv[cse_var_6] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_10 * 2)) + 16)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 4)])));
        }
        for (int32_t vw_11 = 0; vw_11 < 2; ++vw_11) {
          conv[vw_11] = (conv[vw_11] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_11 * 2)) + 7)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 5)])));
        }
        for (int32_t vw_12 = 0; vw_12 < 2; ++vw_12) {
          int32_t cse_var_7 = (vw_12 + 2);
          conv[cse_var_7] = (conv[cse_var_7] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_12 * 2)) + 17)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 5)])));
        }
        for (int32_t vw_13 = 0; vw_13 < 2; ++vw_13) {
          conv[vw_13] = (conv[vw_13] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_13 * 2)) + 10)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 6)])));
        }
        for (int32_t vw_14 = 0; vw_14 < 2; ++vw_14) {
          int32_t cse_var_8 = (vw_14 + 2);
          conv[cse_var_8] = (conv[cse_var_8] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_14 * 2)) + 20)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 6)])));
        }
        for (int32_t vw_15 = 0; vw_15 < 2; ++vw_15) {
          conv[vw_15] = (conv[vw_15] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_15 * 2)) + 11)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 7)])));
        }
        for (int32_t vw_16 = 0; vw_16 < 2; ++vw_16) {
          int32_t cse_var_9 = (vw_16 + 2);
          conv[cse_var_9] = (conv[cse_var_9] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_16 * 2)) + 21)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 7)])));
        }
        for (int32_t vw_17 = 0; vw_17 < 2; ++vw_17) {
          conv[vw_17] = (conv[vw_17] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_17 * 2)) + 12)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 8)])));
        }
        for (int32_t vw_18 = 0; vw_18 < 2; ++vw_18) {
          int32_t cse_var_10 = (vw_18 + 2);
          conv[cse_var_10] = (conv[cse_var_10] + (((int32_t)((int8_t*)data_vec)[(((((ax2_outer * 600) + (ax3_outer * 75)) + (ci_1 * 25)) + (vw_18 * 2)) + 22)]) * ((int32_t)((int8_t*)fused_constant)[((ci_1 * 9) + 8)])));
        }
      }
      for (int32_t ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
        int32_t v_ = conv[ax3_inner] + ((int32_t*)fused_nn_conv2d_constant)[0];
        T_relu[(((ax2_outer * 32) + (ax3_outer * 2)) + ax3_inner)] = ((v_) > (0) ? (v_) : (0));
      }
      for (int32_t ax3_inner_1 = 0; ax3_inner_1 < 2; ++ax3_inner_1) {
        int32_t v__1 = conv[(ax3_inner_1 + 2)] + ((int32_t*)fused_nn_conv2d_constant)[0];
        T_relu[((((ax2_outer * 32) + (ax3_outer * 2)) + ax3_inner_1) + 16)] = ((v__1) > (0) ? (v__1) : (0));
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, 0, data_vec) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_cpuonly___tvm_main__(int8_t* input_0_buffer_var, int32_t* output_buffer_var) {
  if (tvmgen_cpuonly_fused_nn_conv2d_add_nn_relu(input_0_buffer_var, output_buffer_var) != 0 ) return -1;
  return 0;
}

