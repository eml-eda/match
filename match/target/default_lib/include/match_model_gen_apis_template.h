#ifndef __MATCH_MODEL_GEN_APIS_H__
#define __MATCH_MODEL_GEN_APIS_H__

#include <match/types.h>

void match_api_prepare_inputs_from_prev_out(
    % for idx,(inp_name,inp) in enumerate(match_inputs.items()):
    ${inp["c_type"]}* inp_pt_${idx},
    int* freed_inp_${idx},
    % endfor
    % for idx,(out_name,out) in enumerate(match_outputs.items()):
    ${out["c_type"]}* out_pt_${idx},
    % endfor
    int* generation_done,
    int prev_model_idx
);

void match_api_update_sizes();

#endif