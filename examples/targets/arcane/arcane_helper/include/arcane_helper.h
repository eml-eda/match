#ifndef __CARUS_HELPER_H__
#define __CARUS_HELPER_H__

#include <match/ctx.h>
#include <xheepsoc.h>
#include <stdlib.h>
#include "l1_hal.h"
#include "l1_bare.h"
#include "l1_loader.h"
#include "l1_kernels.h"
#include "l1_misc.h"

// void mnist_handle_output(int* output_pt, int classes, int runtime_status);

void arcane_helper_init_l1_mem();

void arcane_compute_wrapper(MatchCtx* ctx);

#endif 