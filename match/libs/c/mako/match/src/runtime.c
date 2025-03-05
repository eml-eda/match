## This is a Mako template for generating a dynamic runtime in C for a static LLM model with multiple inputs and outputs.
#include <${match_model.model_name}/runtime.h>

% for model_name in all_model_names:
% if executors[model_name]=="aot":
struct tvmgen_${model_name}_inputs model_inps_${model_name.upper()};
struct tvmgen_${model_name}_outputs model_outs_${model_name.upper()};
% endif
% endfor

% if match_model.benchmark_model:
% for model_name in all_model_names:
double benchmark_${model_name}_model(int iterations){
    int status = 0;
    int fails = 0;
    % if executors[model_name]=="graph":
    #ifndef __MATCH_${model_name}_DEFAULT_INPUTS_H__
    % for inp_name,inp in inputs.items():
    ${inp["c_type"]}* ${inp_name}_bench_pt = ${target.alloc_fn}(sizeof(${inp["c_type"]}) * ${inp["prod_shape"]});
    % endfor
    #else
    % for inp_name,inp in inputs.items():
    ${inp["c_type"]}* ${inp_name}_bench_pt = ${inp_name}_default;
    % endfor
    #endif
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_bench_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % endfor
    % endif
    ${target.timestamp_type} start,end;
    start = ${target.start_get_timestamp_api}();
    for(int i=0;i<iterations;i++){
        % if executors[model_name]=="aot":
        status = tvmgen_${model_name}_run(&model_inps_${model_name.upper()},&model_outs_${model_name.upper()});
        % elif executors[model_name]=="graph":
        status = match_${model_name}_run_graph(
            % for inp_name,inp in inputs.items():
            ${inp_name}_bench_pt,
            % endfor
            % for out_idx,(out_name,out) in enumerate(outputs.items()):
            ${"" if out_idx==0 else ","} ${out_name}_bench_pt
            % endfor
        );
        % endif
        if(status) fails++;
    }
    end = ${target.end_get_timestamp_api}();

    double time_elapsed_ms = ((double)(end - start)) ${target.timestamp_to_ms};
    printf("[MATCH RUNTIME] [${model_name}_BENCH] time %fms; time per iterations %fms; fails %d\n",
        time_elapsed_ms, time_elapsed_ms/iterations, fails);
    
    % if executors[model_name]=="graph":
    // free up tensors
    #ifndef __MATCH_${model_name}_DEFAULT_INPUTS_H__
    % for inp_name,inp in inputs.items():
    ${target.free_fn}(${inp_name}_bench_pt);
    % endfor
    #endif
    % for out_name,out in outputs.items():
    ${target.free_fn}(${out_name}_bench_pt);
    % endfor
    % endif
    return time_elapsed_ms/iterations;
}
% endfor
% endif

% if match_model.golden_cpu_model:
% for model_name in [mod for mod in all_model_names if mod!=match_model.model_name+"_golden_cpu"]:
int check_${model_name}_differences_with_golden_model(){
    int diffs = 0;
    % for mod_name in [model_name, match_model.model_name+"_golden_cpu"]:
    % if executors[mod_name]=="aot":
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_${mod_name}_pt = (${out["c_type"]}*)model_outs_${mod_name.upper()}.output;
    % endfor
    tvmgen_${mod_name}_run(&model_inps_${mod_name.upper()},&model_outs_${mod_name.upper()});
    % elif executors[mod_name]=="graph":
    #ifndef __MATCH_${model_name}_DEFAULT_INPUTS_H__
    % for inp_name,inp in inputs.items():
    ${inp["c_type"]}* ${inp_name}_${mod_name}_pt = ${target.alloc_fn}(sizeof(${inp["c_type"]}) * ${inp["prod_shape"]});
    % endfor
    #else
    % for inp_name,inp in inputs.items():
    ${inp["c_type"]}* ${inp_name}_${mod_name}_pt = ${inp_name}_default;
    % endfor
    #endif
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_${mod_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % endfor
    match_${mod_name}_run_graph(
        % for inp_name,inp in inputs.items():
        ${inp_name}_${mod_name}_pt,
        % endfor
        % for out_idx,(out_name,out) in enumerate(outputs.items()):
        ${"" if out_idx==0 else ","} ${out_name}_${mod_name}_pt
        % endfor
    );
    % endif
    % endfor
    // check diffs
    % for out_name,out in outputs.items():
    for(int i=0;i<${out["prod_shape"]};i++)
        if(${out_name}_${match_model.model_name+"_golden_cpu"}_pt[i] != ${out_name}_${model_name}_pt[i]){
            #ifdef VERBOSE
            printf("[MATCH RUNTIME] golden cpu model and ${model_name} outputs DO NOT match at i %d golden cpu model: %d ${model_name}: %d diff: %d\n",
                i, ${out_name}_${match_model.model_name+"_golden_cpu"}_pt[i], ${out_name}_${model_name}_pt[i]
                , ${out_name}_${match_model.model_name+"_golden_cpu"}_pt[i]-${out_name}_${model_name}_pt[i]
            );
            #endif
            diffs++;
        }
    % endfor
    % for mod_name in [model_name, match_model.model_name+"_golden_cpu"]:
    % if executors[mod_name]=="graph":
    // free up tensors
    #ifndef __MATCH_${model_name}_DEFAULT_INPUTS_H__
    % for inp_name,inp in inputs.items():
    ${target.free_fn}(${out_name}_${mod_name}_pt);
    % endfor
    #endif
    % for out_name,out in outputs.items():
    ${target.free_fn}(${out_name}_${mod_name}_pt);
    % endfor
    % endif
    % endfor
    return diffs;
}

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
    match_runtime_ctx* match_ctx){
    % for mod_name in [model_name, match_model.model_name+"_golden_cpu"]:
    % if executors[mod_name]=="aot":
    model_inps_${mod_name.upper()} = (struct tvmgen_${mod_name}_inputs){
        % for inp_name in inputs.keys():
        .${inp_name} = ${inp_name}_${mod_name}_pt,
        % endfor
    };
    model_outs_${mod_name.upper()} = (struct tvmgen_${mod_name}_outputs){
        % for out_name in outputs.keys():
        .${out_name} = ${out_name}_${mod_name}_pt,
        % endfor
    };
    % endif
    % endfor
    int diffs = check_${model_name}_differences_with_golden_model();
    if(diffs)   printf("[MATCH RUNTIME] Golden check: check failed ❌ %d differences between golden cpu model and ${model_name}\n",diffs);
    else    printf("[MATCH RUNTIME] Golden check: check passed ✅ no differences between golden cpu model and ${model_name}\n");
    % if match_model.benchmark_model:
    double golden_cpu_model_time_per_iter = benchmark_${match_model.model_name+"_golden_cpu"}_model(benchmark_iterations);
    double ${model_name}_time_per_iter = benchmark_${model_name}_model(benchmark_iterations);
    printf("[MATCH RUNTIME] ${model_name}/golden_cpu_model ms per iteration: %f golden_cpu_model/${model_name} ms per iteration %f\n",
    ${model_name}_time_per_iter/golden_cpu_model_time_per_iter,golden_cpu_model_time_per_iter/${model_name}_time_per_iter);
    % endif
}
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
    match_runtime_ctx* match_ctx){
    % if executors[model_name]=="aot":
    model_inps_${model_name.upper()} = (struct tvmgen_${model_name}_inputs){
        % for inp_name in inputs.keys():
        .${inp_name} = ${inp_name}_pt,
        % endfor
    };
    model_outs_${model_name.upper()} = (struct tvmgen_${model_name}_outputs){
        % for out_name in outputs.keys():
        .${out_name} = ${out_name}_pt,
        % endfor
    };
    match_ctx->status = tvmgen_${model_name}_run(&model_inps_${model_name.upper()},&model_outs_${model_name.upper()});
    % elif executors[model_name]=="graph":
    match_ctx->status = match_${model_name}_run_graph(
        % for inp_name,inp in inputs.items():
        ${inp_name}_pt,
        % endfor
        % for out_idx,(out_name,out) in enumerate(outputs.items()):
        ${"" if out_idx==0 else ","} ${out_name}_pt
        % endfor
    );
    % endif
    return;
}
% endfor