## This is a Mako template for generating a dynamic runtime in C for a static LLM model with multiple inputs and outputs.
#include <${match_model.model_name}/runtime.h>

% if match_model.executor=="aot":
% for model_name in all_model_names:
struct tvmgen_${model_name}_inputs model_inps_${model_name.upper()};
struct tvmgen_${model_name}_outputs model_outs_${model_name.upper()};
% endfor
% endif

% if match_model.benchmark_model:
% for model_name in all_model_names:
double benchmark_${model_name}_model(int iterations){
    int status = 0;
    int fails = 0;
    ${target.timestamp_type} start,end;
    start = ${target.start_get_timestamp_api}();
    for(int i=0;i<iterations;i++){
        % if match_model.executor=="aot":
        status = tvmgen_${model_name}_run(&model_inps_${model_name.upper()},&model_outs_${model_name.upper()});
        % elif match_model.executor=="graph":
        status = match_${model_name}_graph_runtime();
        % endif
        if(status) fails++;
    }
    end = ${target.end_get_timestamp_api}();

    double time_elapsed_ms = ((double)(end - start)) ${target.timestamp_to_ms};
    printf("[MATCH RUNTIME] [${model_name}_BENCH] time %fms; time per iterations %fms; fails %d\n",
        time_elapsed_ms, time_elapsed_ms/iterations, fails);
    return time_elapsed_ms/iterations;
}
% endfor
% endif

% if match_model.golden_cpu_model:
% for model_name in [mod for mod in all_model_names if mod!=match_model.model_name+"_golden_cpu"]:
int check_${model_name}_differences_with_golden_model(){
    % if match_model.executor=="aot":
    tvmgen_${match_model.model_name+"_golden_cpu"}_run(&model_inps_${(match_model.model_name+"_golden_cpu").upper()},&model_outs_${(match_model.model_name+"_golden_cpu").upper()});
    tvmgen_${model_name}_run(&model_inps_${model_name.upper()},&model_outs_${model_name.upper()});
    % elif match_model.executor=="graph":
    match_${match_model.model_name+"_golden_cpu"}_graph_runtime();
    match_${model_name}_graph_runtime();
    % endif
    int diffs = 0;
    % for out_name,out in outputs.items():
    for(int i=0;i<${out["prod_shape"]};i++)
        if(((${out["c_type"]}*)model_outs_${(match_model.model_name+"_golden_cpu").upper()}.output)[i]!=((${out["c_type"]}*)model_outs_${model_name.upper()}.output)[i]){
            printf("[MATCH RUNTIME] golden cpu model and ${model_name} outputs DO NOT match at i %d golden cpu model: %d ${model_name}: %d diff: %d\n",
                i,((${out["c_type"]}*)model_outs_${(match_model.model_name+"_golden_cpu").upper()}.output)[i],((${out["c_type"]}*)model_outs_${model_name.upper()}.output)[i]
                ,((${out["c_type"]}*)model_outs_${(match_model.model_name+"_golden_cpu").upper()}.output)[i]-((${out["c_type"]}*)model_outs_${model_name.upper()}.output)[i]
            );
            diffs++;
        }
    % endfor
    return diffs;
}

void match_golden_check_${model_name}_runtime(
    % for match_inp_name,match_inp in inputs.items():
    ${out["c_type"]}* ${match_inp_name}_pt,
    % endfor
    % for match_inp_name,match_inp in inputs.items():
    ${out["c_type"]}* ${match_inp_name}_golden_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_golden_pt,
    % endfor
    int benchmark_iterations,
    match_runtime_ctx* match_ctx){
    % if match_model.executor=="aot":
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
    model_inps_${(match_model.model_name+"_golden_cpu").upper()} = (struct tvmgen_${match_model.model_name+"_golden_cpu"}_inputs){
        % for inp_name in inputs.keys():
        .${inp_name} = ${inp_name}_golden_pt,
        % endfor
    };
    model_outs_${(match_model.model_name+"_golden_cpu").upper()} = (struct tvmgen_${match_model.model_name+"_golden_cpu"}_outputs){
        % for out_name in outputs.keys():
        .${out_name} = ${out_name}_golden_pt,
        % endfor
    };
    int diffs = check_${model_name}_differences_with_golden_model();
    % elif match_model.executor=="graph":
    // TODO: fix this part here
    int diffs = check_${model_name}_differences_with_golden_model();
    % endif
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
    % if match_model.executor=="aot":
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
    % elif match_model.executor=="graph":
    match_ctx->status = match_${model_name}_graph_runtime(
        % for inp_name,inp in inputs.items():
        ${inp["c_type"]}* ${inp_name}_pt,
        % endfor
        % for out_idx,(out_name,out) in enumerate(outputs.items()):
        ${"" if out_idx==0 else ","}${out["c_type"]}* ${out_name}_pt
        % endfor
    );
    % endif
    return;
}
% endfor