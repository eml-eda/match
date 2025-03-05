#ifndef __MATCH_${match_model.model_name}_RUNTIME_H__
#define __MATCH_${match_model.model_name}_RUNTIME_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tvm_runtime.h>  // Include TVM runtime API
#include <match/ctx.h>
#include <${match_model.model_name}/default_inputs.h>
#include <${target.name}.h>
% for model_name in all_model_names:
% if executors[model_name]=="aot":
#include <tvmgen_${model_name}.h>
% elif executors[model_name]=="graph":
#include <${model_name}_graph.h>
% endif
% endfor

// target specific inlcudes
% for inc_h in target.include_list:
#include <${inc_h}.h>
% endfor

% for model_name in all_model_names:
% if executors[model_name]=="aot":
extern struct tvmgen_${model_name}_inputs model_inps_${model_name.upper()};
extern struct tvmgen_${model_name}_outputs model_outs_${model_name.upper()};
% endif
% endfor

% if match_model.benchmark_model:
% for model_name in all_model_names:
double benchmark_${model_name}_model(int iterations);
% endfor
% endif

% if match_model.golden_cpu_model:
% for model_name in [mod for mod in all_model_names if mod!=match_model.model_name+"_golden_cpu"]:
int check_${model_name}_differences_with_golden_model();
void match_golden_check_${model_name}_runtime(
    % for match_inp_name,match_inp in inputs.items():
    ${match_inp["c_type"]}* ${match_inp_name}_${model_name}_pt,
    % endfor
    % for match_inp_name,match_inp in inputs.items():
    ${match_inp["c_type"]}* ${match_inp_name}_${match_model.model_name+"_golden_cpu"}_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_${model_name}_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_${match_model.model_name+"_golden_cpu"}_pt,
    % endfor
    int benchmark_iterations,
    match_runtime_ctx* match_ctx
);
% endfor
% endif

% for model_name in all_model_names:
void match_${model_name}_runtime(
    % for inp_name,inp in inputs.items():
    ${inp["c_type"]}* ${inp_name}_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_pt,
    % endfor
    match_runtime_ctx* match_ctx);
% endfor

#endif