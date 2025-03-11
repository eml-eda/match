#ifndef __PULP_NEURAL_NEURAL_ENGINE_H__
#define __PULP_NEURAL_NEURAL_ENGINE_H__

#include <match/ctx.h>
#include <NE16.h>
#include <pulp_neural/ne16.h>
#include <pulp_neural/ne16_task.h>
#include <pulp_neural/ne16_pulp_bsp.h>

#define BUFFER_DIMENSION 1

static ne16_dev_t* nnx_dev;
static ne16_task_t nnx_tasks[BUFFER_DIMENSION];

inline void match_ne16_set_nnx_dev(){
    nnx_dev = ne16_pulp_get_dev();
}

inline ne16_dev_t* match_ne16_get_nnx_dev(){
    return nnx_dev;
}

inline ne16_task_t* match_ne16_get_nnx_task(int n){
    return &nnx_tasks[n];
}

void neural_engine_compute_tile(MatchCtx* ctx);

void neural_engine_lib_init(MatchCtx* ctx);

void neural_engine_lib_close(MatchCtx* ctx);

void wait_neural_engine_compute(MatchCtx* ctx);

#endif