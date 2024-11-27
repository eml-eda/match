#ifndef __MATCH_RUNTIME_H__
#define __MATCH_RUNTIME_H__

% for gen_model in generative_models.keys():
#include <tvmgen_${gen_model}.h>
% endfor

typedef enum{
    % for gen_model in generative_models.keys():
    MATCH_GEN_MODEL_${gen_model.upper()},
    % endfor
}MATCH_DYN_MODELS;


% for dim_name,dim in dynamic_dims.items():
#define DIM_${dim_name}_MAX ${dim.max}
#define DIM_${dim_name}_MIN ${dim.min}
#define DIM_${dim_name}
% endfor

typedef struct match_runtime_ctx_t{
    int status;
}match_runtime_ctx;

void match_generative_runtime(
    % for inp_name,inp in inputs.items():
    ${inp["c_type"]}* ${inp_name}_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_pt,
    % endfor
    % for dyn_dim in dynamic_dims.keys():
    int starting_dyn_dim_${dyn_dim}_size,
    % endfor
    match_runtime_ctx* match_ctx
);

void match_default_runtime(match_runtime_ctx* match_ctx);

#endif