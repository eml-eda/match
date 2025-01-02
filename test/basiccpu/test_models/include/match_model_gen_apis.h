#ifndef __MATCH_MODEL_GEN_APIS_H__
#define __MATCH_MODEL_GEN_APIS_H__

#include <match/types.h>

void match_api_prepare_inputs_from_prev_out(
    int8_t* inp_pt_0,
    int* freed_inp_0,
    int* generation_done,
    int prev_model_idx
);

void match_api_update_sizes();

#endif