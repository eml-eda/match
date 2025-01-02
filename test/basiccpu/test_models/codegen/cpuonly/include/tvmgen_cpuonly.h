#ifndef TVMGEN_CPUONLY_H_
#define TVMGEN_CPUONLY_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Input tensor input_0 size (in bytes) for TVM module "cpuonly" 
 */
#define TVMGEN_CPUONLY_INPUT_0_SIZE 3072
/*!
 * \brief Output tensor output size (in bytes) for TVM module "cpuonly" 
 */
#define TVMGEN_CPUONLY_OUTPUT_SIZE 1024
/*!
 * \brief Input tensor pointers for TVM module "cpuonly" 
 */
struct tvmgen_cpuonly_inputs {
  void* input_0;
};

/*!
 * \brief Output tensor pointers for TVM module "cpuonly" 
 */
struct tvmgen_cpuonly_outputs {
  void* output;
};

/*!
 * \brief entrypoint function for TVM module "cpuonly"
 * \param inputs Input tensors for the module 
 * \param outputs Output tensors for the module 
 */
int32_t tvmgen_cpuonly_run(
  struct tvmgen_cpuonly_inputs* inputs,
  struct tvmgen_cpuonly_outputs* outputs
);
/*!
 * \brief Workspace size for TVM module "cpuonly" 
 */
#define TVMGEN_CPUONLY_WORKSPACE_SIZE 4816

#ifdef __cplusplus
}
#endif

#endif // TVMGEN_CPUONLY_H_
