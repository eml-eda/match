## This is a Mako template for generating a dynamic runtime in C for a static LLM model with multiple inputs and outputs.
#include <match/runtime.h>

% for dim_name,dim in dynamic_dims.items():
int dyn_dim_${dim_name}_size = ${dim.min};
int dyn_dim_${dim_name}_size_pad = ${dim.min};
int dyn_dim_${dim_name}_padded_sizes[${len(generative_models)}] = {${str([dyn_model.dynamic_sizes[dim_name] for dyn_model_name,dyn_model in models.items()])[1:-1]}};
% endfor

% for gen_model_name in generative_models.keys():
struct tvmgen_${gen_model_name}_inputs model_inps_${gen_model_name.upper()};
struct tvmgen_${gen_model_name}_outputs model_outs_${gen_model_name.upper()};
% endfor

% for gen_model_name,gen_model in generative_models.items():
int calc_padding_for_gen_model_${gen_model_name}(){
    int padding_required = 0;
    int size_padded = 1;
    int size_curr = 1;
    % for inp_name,inp in inputs.items():
    size_padded = 1;
    size_curr = 1;
    % for dim in inp["dims"]:
    % if type(dim)==str:
    size_curr*=dyn_dim_${dim}_size;
    size_padded*=(${"+".join([str(dim_) if type(dim_)==int else str(gen_model.dynamic_sizes[dim_]) for dim_ in dim.split(" ")[::2]])});
    % elif type(dim)==int:
    size_curr*=${dim};
    size_padded*=${dim};
    % endif
    % endfor
    padding_required+=size_padded-size_curr;
    % endfor
    return padding_required;
}

int is_${gen_model_name}_usable(){
    % for dyn_dim in dynamic_dims.keys():
    if(dyn_dim_${dyn_dim}_size>${gen_model.dynamic_sizes[dyn_dim]}) return 0;
    % endfor
    return 1;
}

% for inp_name,inp in inputs.items():
void store_inp_${inp_name}_for_${gen_model_name}(${inp["c_type"]}* buffer_inp_pt,${inp["c_type"]}* model_inp_pt){
    int padded_idx = 0;
    int model_idx_pt = 0;
    int buffer_idx_pt = 0;
    % for idx_dim,dim in enumerate(inp["dims"]):
    for(int dim_${idx_dim}=0;dim_${idx_dim}<(${dim if type(dim)==int else "+".join([str(dim_) if type(dim_)==int else str(gen_model.dynamic_sizes[dim_]) for dim_ in dim.split(" ")[::2]])});dim_${idx_dim}++){
    % if type(dim)==str:
        if(dim_${idx_dim}>= (${"+".join([str(dim_) if type(dim_)==int else "dyn_dim_"+dim_+"_size" for dim_ in dim.split(" ")[::2]])}))  padded_idx=1;
        else    padded_idx=0;
    % endif
    % if idx_dim==len(inp["dims"])-1:
        if(padded_idx) model_inp_pt[model_idx_pt] = 0;
        else    model_inp_pt[model_idx_pt] = buffer_inp_pt[buffer_idx_pt];
    % endif
    % endfor
    % for idx_dim,dim in list(enumerate(inp["dims"]))[::-1]:
        model_idx_pt +=
        % for idx_dim_inner,dim_inner in enumerate(inp["dims"][idx_dim+1:]):
            (${dim_inner if type(dim_inner)==int else "+".join([str(dim_inner_) if type(dim_inner_)==int else str(gen_model.dynamic_sizes[dim_inner_]) for dim_inner_ in dim_inner.split(" ")[::2]])})*
        % endfor
        1;
        buffer_idx_pt +=
        % for idx_dim_inner,dim_inner in enumerate(inp["dims"][idx_dim+1:]):
            ${dim_inner if type(dim_inner)==int else f"dyn_dim_"+dim_inner+"_size"}*
        % endfor
        1;
    }
    % endfor
}
% endfor
% endfor

% for inp_name in inputs.keys():
void store_inp_${inp_name}_for_gen_model(${inp["c_type"]}* buffer_inp_pt,${inp["c_type"]}* model_inp_pt,int model_idx){
    switch(model_idx){
        % for gen_model in set(generative_models.keys())-{"default"}:
        case MATCH_GEN_MODEL_${gen_model.upper()}:
            store_inp_${inp_name}_for_${gen_model}(buffer_inp_pt,model_inp_pt);
            break;
        % endfor
        default:
            store_inp_${inp_name}_for_default(buffer_inp_pt,model_inp_pt);
            break;
    }
}
% endfor

int get_closest_model_idx(){
    int min_padding = -1;
    int padding = -1;
    int model = -1;
    % for gen_model in generative_models.keys():
    if(is_${gen_model}_usable()){
        padding = calc_padding_for_gen_model_${gen_model}();
        if(padding<min_padding || min_padding<0){
            min_padding = padding;
            model = MATCH_GEN_MODEL_${gen_model.upper()};
        }
    }
    % endfor
    return model;
}

// Main runtime logic
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
) {
    //setting input data
    #ifdef MATCH_DEBUG_LOG
    FILE *debug_file=fopen("match_debug.log", "w");
    fprintf(debug_file, "[MATCH RUNTIME] startup...\n");
    #endif
    int prev_model_idx = -1;
    // Generation process (e.g., LLM token generation)
    int generation_done = 0;
    % for inp_name,inp in inputs.items():
    int freed_${inp_name} = 1;
    ${inp["c_type"]}* ${inp_name}_prev_pt = ${inp_name}_pt;
    int prev_size_${inp_name} = 1;
    % for dim in inp["dims"]:
    % if type(dim)==str:
    prev_size_${inp_name} *= (${"+".join([str(dim_) if type(dim_)==int else "starting_dyn_dim_"+dim_+"_size" for dim_ in dim.split(" ")[::2]])});
    % elif type(dim)==int:
    prev_size_${inp_name} *= ${dim};
    % endif
    % endfor
    ${inp["c_type"]}* ${inp_name}_inp;
    % endfor

    % for dyn_dim in dynamic_dims.keys():
    dyn_dim_${dyn_dim}_size_pad = starting_dyn_dim_${dyn_dim}_size;
    dyn_dim_${dyn_dim}_size = starting_dyn_dim_${dyn_dim}_size;
    % endfor

    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_out;
    ${out["c_type"]}* ${out_name}_prev_pt;
    int prev_size_${out_name} = 1;
    % endfor

    while (!generation_done) {

        #ifdef MATCH_DEBUG_LOG
        fprintf(debug_file, "[MATCH RUNTIME] generating...\n");
        #endif

        // Pad each input to match the nearest static model size
        int model_idx = get_closest_model_idx();
        #ifdef MATCH_DEBUG_LOG
        fprintf(debug_file, "[MATCH RUNTIME] model #%d\n",model_idx);
        #endif
        // Pad inputs to match the nearest model size
        % for inp_name, inp in inputs.items():
        #ifdef MATCH_DEBUG_LOG
        fprintf(debug_file, "[MATCH RUNTIME] setting inp ${inp_name}\n");
        #endif
        if(freed_${inp_name}){
            int size_${inp_name} = 1;
            % for dim in inp["dims"]:
            % if type(dim)==str:
            size_${inp_name} *= (${"+".join([str(dim_) if type(dim_)==int else "dyn_dim_"+dim_+"_padded_sizes[model_idx]" for dim_ in dim.split(" ")[::2]])});
            % elif type(dim)==int:
            size_${inp_name} *= ${dim};
            % endif
            % endfor
            if(size_${inp_name} != prev_size_${inp_name})   ${inp_name}_inp = malloc(sizeof(${inp["c_type"]}) * size_${inp_name});
            prev_size_${inp_name} = size_${inp_name};
            #ifdef MATCH_DEBUG_LOG
            fprintf(debug_file, "[MATCH RUNTIME] storing inp ${inp_name}\n");
            #endif
            store_inp_${inp_name}_for_gen_model(${inp_name}_prev_pt,${inp_name}_inp,model_idx);
            if(prev_model_idx!=-1)  free(${inp_name}_prev_pt); 
        }
        % endfor

        #ifdef MATCH_DEBUG_LOG
        fprintf(debug_file, "[MATCH RUNTIME] setting outputs\n");
        #endif
        if(model_idx != prev_model_idx){
            % for out_name,out in outputs.items():
            int size_${out_name} = 1;
            % for dim in out["dims"]:
            % if type(dim)==str:
            size_${out_name} *= (${"+".join([str(dim_) if type(dim_)==int else "dyn_dim_"+dim_+"_padded_sizes[model_idx]" for dim_ in dim.split(" ")[::2]])});
            % elif type(dim)==int:
            size_${out_name} *= ${dim};
            % endif
            % endfor
            if(size_${out_name} != prev_size_${out_name}){
                if(prev_model_idx!=-1)  free(${out_name}_prev_pt);
                ${out_name}_out = malloc(sizeof(${out["c_type"]}) * size_${out_name});
            }
            % endfor
        }
        switch(model_idx){
        % for idx_model,(gen_model_name,gen_model) in enumerate(generative_models.items()):
            % if gen_model_name!="default":
            case ${idx_model}:
                model_inps_${gen_model_name.upper()} = (struct tvmgen_${gen_model_name}_inputs){
                    % for inp_name in inputs.keys():
                    .${inp_name} = ${inp_name}_inp,
                    % endfor
                };
                model_outs_${gen_model_name.upper()} = (struct tvmgen_${gen_model_name}_outputs){
                    % for out_name in outputs.keys():
                    .${out_name} = ${out_name}_out,
                    % endfor
                };
                match_ctx->status = tvmgen_${gen_model_name}_run(&model_inps_${gen_model_name.upper()},&model_outs_${gen_model_name.upper()});
                break;
            % endif
        % endfor
            default:
                model_inps_DEFAULT = (struct tvmgen_default_inputs){
                    % for inp_name in inputs.keys():
                    .${inp_name} = ${inp_name}_inp,
                    % endfor
                };
                model_outs_DEFAULT = (struct tvmgen_default_outputs){
                    % for out_name in outputs.keys():
                    .${out_name} = ${out_name}_out,
                    % endfor
                };
                match_ctx->status = tvmgen_default_run(&model_inps_DEFAULT,&model_outs_DEFAULT);
                break;
        }
        
        match_api_update_sizes();
        % for dyn_dim in dynamic_dims.keys():
        dyn_dim_${dyn_dim}_size_pad = dyn_dim_${dyn_dim}_padded_sizes[model_idx];
        % endfor
        % for inp_name in inputs.keys():
        ${inp_name}_prev_pt = ${inp_name}_inp;
        % endfor
        prev_model_idx = model_idx;

        % for out_name,out in outputs.items():
        ${out_name}_prev_pt=${out_name}_out;
        % endfor

        match_api_prepare_inputs_from_prev_out(
            % for inp_name in inputs.keys():
            ${inp_name}_prev_pt,
            &freed_${inp_name},
            % endfor
            % for out_name in outputs.keys():
            ${out_name}_prev_pt,
            % endfor
            &generation_done,
            prev_model_idx
        );
        #ifdef MATCH_DEBUG_LOG
        fprintf(debug_file, "[MATCH RUNTIME] finished generation with generation status %d\n",generation_done);
        #endif
    }
    #ifdef MATCH_DEBUG_LOG
    fprintf(debug_file, "[MATCH RUNTIME] finished generation\n");
    fclose(debug_file);
    #endif
}

% if golden_cpu_model:
% for model_name in set(generative_models.keys())-{"golden_cpu_model"}:
int check_${model_name}_differences_with_golden_model(){
    tvmgen_golden_cpu_model_run(&model_inps_GOLDEN_CPU_MODEL,&model_outs_GOLDEN_CPU_MODEL);
    tvmgen_${model_name}_run(&model_inps_${model_name.upper()},&model_outs_${model_name.upper()});
    int diffs = 0;
    % for out_name,out in outputs.items():
    for(int i=0;i<${out["prod_shape"]};i++)
        if(((${out["c_type"]}*)model_outs_GOLDEN_CPU_MODEL.output)[i]!=((${out["c_type"]}*)model_outs_${model_name.upper()}.output)[i]){
            printf("[MATCH RUNTIME] golden cpu model and ${model_name} outputs DO NOT match at i %d golden cpu model: %d ${model_name}: %d diff: %d\n",
                i,((${out["c_type"]}*)model_outs_GOLDEN_CPU_MODEL.output)[i],((${out["c_type"]}*)model_outs_${model_name.upper()}.output)[i]
                ,((${out["c_type"]}*)model_outs_GOLDEN_CPU_MODEL.output)[i]-((${out["c_type"]}*)model_outs_${model_name.upper()}.output)[i]
            );
            diffs++;
        }
    % endfor
    return diffs;
}
% endfor
% endif
% if benchmarking:
% for model_name in generative_models.keys():
double benchmark_${model_name}_model(int iterations){
    int status = 0;
    int fails = 0;
    clock_t start, end;
    start = clock();
    for(int i=0;i<iterations;i++){
        status=tvmgen_${model_name}_run(&model_inps_${model_name.upper()},&model_outs_${model_name.upper()});
        if(status) fails++;
    }
    end = clock();

    double time_elapsed_ms = ((double)(end - start))/CLOCKS_PER_SEC * 1000;
    printf("[MATCH RUNTIME] [${model_name}_BENCH] time %fms; time per iterations %fms; fails %d\n",
        time_elapsed_ms, time_elapsed_ms/iterations, fails);
    return time_elapsed_ms/iterations;
}
% endfor
% endif

% if golden_cpu_model:
% for model_name in set(generative_models.keys())-{"golden_cpu_model"}:
void match_golden_check_${model_name}_runtime(
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_golden_pt,
    % endfor
    int benchmark_iterations,
    match_runtime_ctx* match_ctx){
    model_inps_${model_name.upper()} = (struct tvmgen_${model_name}_inputs){
        % for inp_name in inputs.keys():
        .${inp_name} = ${inp_name}_default,
        % endfor
    };
    model_outs_${model_name.upper()} = (struct tvmgen_${model_name}_outputs){
        % for out_name in outputs.keys():
        .${out_name} = ${out_name}_pt,
        % endfor
    };
    model_inps_GOLDEN_CPU_MODEL = (struct tvmgen_golden_cpu_model_inputs){
        % for inp_name in inputs.keys():
        .${inp_name} = ${inp_name}_default,
        % endfor
    };
    model_outs_GOLDEN_CPU_MODEL = (struct tvmgen_golden_cpu_model_outputs){
        % for out_name in outputs.keys():
        .${out_name} = ${out_name}_pt,
        % endfor
    };
    int diffs = check_${model_name}_differences_with_golden_model();
    if(diffs)   printf("[MATCH RUNTIME] Golden check: check failed ❌ %d differences between golden cpu model and ${model_name}\n",diffs);
    else    printf("[MATCH RUNTIME] Golden check: check passed ✅ no differences between golden cpu model and ${model_name}\n");
    % if benchmarking:
    double golden_cpu_model_time_per_iter = benchmark_golden_cpu_model_model(benchmark_iterations);
    double ${model_name}_time_per_iter = benchmark_${model_name}_model(benchmark_iterations);
    printf("[MATCH RUNTIME] ${model_name}/golden_cpu_model ms per iteration: %f golden_cpu_model/${model_name} ms per iteration %f\n",
    ${model_name}_time_per_iter/golden_cpu_model_time_per_iter,golden_cpu_model_time_per_iter/${model_name}_time_per_iter);
    % endif
}
% endfor
% endif

void match_basic_runtime(
    % for inp_name,inp in inputs.items():
    ${inp["c_type"]}* ${inp_name}_pt,
    % endfor
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_pt,
    % endfor
    match_runtime_ctx* match_ctx){
    model_inps_DEFAULT = (struct tvmgen_default_inputs){
        % for inp_name in inputs.keys():
        .${inp_name} = ${inp_name}_pt,
        % endfor
    };
    model_outs_DEFAULT = (struct tvmgen_default_outputs){
        % for out_name in outputs.keys():
        .${out_name} = ${out_name}_pt,
        % endfor
    };
    match_ctx->status = tvmgen_default_run(&model_inps_DEFAULT,&model_outs_DEFAULT);
}

void match_default_runtime(
    % for out_name,out in outputs.items():
    ${out["c_type"]}* ${out_name}_pt,
    % endfor
    match_runtime_ctx* match_ctx){
    model_inps_DEFAULT = (struct tvmgen_default_inputs){
        % for inp_name in inputs.keys():
        .${inp_name} = ${inp_name}_default,
        % endfor
    };
    model_outs_DEFAULT = (struct tvmgen_default_outputs){
        % for out_name in outputs.keys():
        .${out_name} = ${out_name}_pt,
        % endfor
    };
    match_ctx->status = tvmgen_default_run(&model_inps_DEFAULT,&model_outs_DEFAULT);
}