#if defined(__MATCH_NODE_PARAMS_tvmgen_default_match_main_0_H__)

#ifndef __MATCH_NODE_HELPER_NN_H__
#define __MATCH_NODE_HELPER_NN_H__

#include <pulp_cluster/node_config.h>
#include <match/ctx.h>
#include <pulp_cluster/pulp_nn_kernels.h>
#include <nodes/default/main_0_params.h>

#include <pulp_cluster/pulp_rt_profiler_wrapper.h>
#include <pmsis.h>
#include <pulp_cluster/cluster_dev.h>
#include <pulp_cluster/dory_dma.h>

void pulp_nn_schedule__(void* args);
void basic_parallel_pulp_schedule(void* args);
void run_node_schedule_nn(MatchCtx* ctx);
void run_pulp_cluster_fn(void (inner_function)(unsigned int* args_inner_function),unsigned int* args);
#endif
#endif