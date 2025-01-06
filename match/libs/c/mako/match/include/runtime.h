#ifndef __MATCH_RUNTIME_H__
#define __MATCH_RUNTIME_H__
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tvm_runtime.h>  // Include TVM runtime API
#include <match/generative_model_apis.h>
#include <match/default_inputs.h>
#include <${target.name}.h>
% for gen_model in generative_models.keys():
#include <tvmgen_${gen_model}.h>
% endfor

// target specific inlcudes
% for inc_h in target.include_list:
#include <${inc_h}.h>
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

% for dim_name,dim in dynamic_dims.items():
extern int dyn_dim_${dim_name}_size;
extern int dyn_dim_${dim_name}_size_pad;
extern int dyn_dim_${dim_name}_padded_sizes[${len(generative_models)}];
% endfor

% for gen_model_name in generative_models.keys():
extern struct tvmgen_${gen_model_name}_inputs model_inps_${gen_model_name.upper()};
extern struct tvmgen_${gen_model_name}_outputs model_outs_${gen_model_name.upper()};
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

void match_basic_runtime(
    % for inp_name,inp in inputs.items():
    ${inp["c_type"]}* ${inp_name}_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_pt,
    % endfor
    match_runtime_ctx* match_ctx);

void match_default_runtime(
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_pt,
    % endfor
    match_runtime_ctx* match_ctx);

#endif