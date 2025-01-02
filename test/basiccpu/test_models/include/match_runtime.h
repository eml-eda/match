#ifndef __MATCH_RUNTIME_H__
#define __MATCH_RUNTIME_H__

#include <tvmgen_default.h>
#include <tvmgen_cpuonly.h>

typedef enum{
    MATCH_GEN_MODEL_DEFAULT,
}MATCH_DYN_MODELS;



typedef struct match_runtime_ctx_t{
    int status;
}match_runtime_ctx;

void match_generative_runtime(
    int8_t* input_0_pt,
    match_runtime_ctx* match_ctx
);

void match_default_runtime(
    match_runtime_ctx* match_ctx);

#endif