// tvm target: c -keys=riscv_cpu,cpu -device=riscv_cpu
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(int8_t* input_0_buffer_var, int32_t* output_buffer_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_match_main_0(int8_t*, int32_t*);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(int8_t* input_0_buffer_var, int32_t* output_buffer_var) {
  if (tvmgen_default_match_main_0(input_0_buffer_var, output_buffer_var) != 0 ) return -1;
  return 0;
}

