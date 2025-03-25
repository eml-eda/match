#ifndef __CARUS_HELPER_H__
#define __CARUS_HELPER_H__

#include <match/ctx.h>
#include <arcane.h>
#include <stdlib.h>
#include "l1_hal.h"
#include "l1_bare.h"
#include "l1_loader.h"
#include "l1_kernels.h"
#include "l1_misc.h"

void mnist_handle_output(int* output_pt, match_runtime_ctx* runtime_ctx);

void carus_helper_init_l1_mem();

void carus_compute_wrapper(MatchCtx* ctx);

#endif 