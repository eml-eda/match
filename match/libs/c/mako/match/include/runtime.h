#ifndef __MATCH_${match_model.model_name}_RUNTIME_H__
#define __MATCH_${match_model.model_name}_RUNTIME_H__
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tvm_runtime.h>  // Include TVM runtime API
// #include <${match_model.model_name}/default_inputs.h>
#include <${target.name}.h>
% for model_name in all_model_names:
% if match_model.executor=="aot":
#include <tvmgen_${model_name}.h>
% elif match_model.executor=="graph":
#include <${model_name}/graph_runtime.h>
% endif
% endfor

// target specific inlcudes
% for inc_h in target.include_list:
#include <${inc_h}.h>
% endfor

% if match_model.executor=="aot":
% for model_name in all_model_names:
extern struct tvmgen_${model_name}_inputs model_inps_${model_name.upper()};
extern struct tvmgen_${model_name}_outputs model_outs_${model_name.upper()};
% endfor
% endif

typedef struct match_runtime_ctx_t{
    int status;
}match_runtime_ctx;


#endif