#ifndef _LLMCPULIB_H_
#define _LLMCPULIB_H_

#include <match_target_params.h>
#include <match_dimensions.h>
#include <match_tile_indexes.h>
#include <match_kernel.h>
#include <stdio.h>

void* llm_matmul(match_kernel* kernel);

#endif