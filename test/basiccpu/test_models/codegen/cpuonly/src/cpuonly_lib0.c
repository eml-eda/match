#include "tvm/runtime/c_runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <tvmgen_cpuonly.h>
TVM_DLL int32_t tvmgen_cpuonly___tvm_main__(void* input_0,void* output0);
int32_t tvmgen_cpuonly_run(struct tvmgen_cpuonly_inputs* inputs,struct tvmgen_cpuonly_outputs* outputs) {return tvmgen_cpuonly___tvm_main__(inputs->input_0,outputs->output);
}
#ifdef __cplusplus
}
#endif
;