#include <${default_model}/default_inputs.h>
#include <${default_model}/runtime.h>

// target specific inlcudes
% for inc_h in target.include_list:
#include <${inc_h}.h>
% endfor
% if golden_cpu_model:
#define GOLDEN_CHECK_BENCH_ITERATIONS ${bench_iterations}
% endif

int main(int argc,char** argv){
    // target specific inits
    % for init_func in target.init_funcs:
    ${init_func}();
    % endfor
    
    match_runtime_ctx match_ctx;

    #if defined(__${default_model}_GRAPH_INPUTS_OUTPUTS_EXT_MEM__) && __${default_model}_GRAPH_INPUTS_OUTPUTS_EXT_MEM__>0
    void* match_ext_mem = ${target.allocate_ext_mem}(__${default_model}_GRAPH_INPUTS_OUTPUTS_EXT_MEM__*${int(golden_cpu_model)+1});
    int ext_mem_offset = 0;
    #endif
    % for inp_name,inp in match_inputs.items():
    #if !defined(__${default_model}_GRAPH_${inp_name}_FROM_EXTERNAL_MEM__) || !__${default_model}_GRAPH_${inp_name}_FROM_EXTERNAL_MEM__
    ${inp["c_type"]}* ${inp_name}_pt = ${inp_name}_default;
    #else
    void* ${inp_name}_pt = match_ext_mem+ext_mem_offset;
    ext_mem_offset += ${inp["bytes"]};
    ${target.load_file_to_ext_mem_fn}("${default_model}_${inp_name}_data.hex", ${inp_name}_pt, ${inp["bytes"]});
    #endif
    % endfor
    % for out_idx,(out_name,out) in enumerate(match_outputs.items()):
    #if !defined(__${default_model}_GRAPH_${default_model}_out_${out_idx}_FROM_EXTERNAL_MEM__) || !__${default_model}_GRAPH_${default_model}_out_${out_idx}_FROM_EXTERNAL_MEM__
    ${out["c_type"]}* ${out_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % if golden_cpu_model:
    ${out["c_type"]}* golden_check_${out_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % endif
    #else
    void* ${out_name}_pt = match_ext_mem+ext_mem_offset;
    ext_mem_offset += ${out["bytes"]};
    % if golden_cpu_model:
    void* golden_check_${out_name}_pt = match_ext_mem+ext_mem_offset;
    ext_mem_offset += ${out["bytes"]};
    % endif
    #endif
    % endfor

    match_${"golden_check_" if golden_cpu_model else ""}${default_model}_runtime(
        % for inp_name in match_inputs.keys():
        ${inp_name}_pt,
        % endfor
        % if golden_cpu_model:
        % for inp_name in match_inputs.keys():
        ${inp_name}_pt,
        % endfor
        % endif
        % for out_name in match_outputs.keys():
        ${out_name}_pt,
        % endfor
        % if golden_cpu_model:
        % for out_name in match_outputs.keys():
        golden_check_${out_name}_pt,
        % endfor
        GOLDEN_CHECK_BENCH_ITERATIONS,
        % endif
        &match_ctx
    );
    
    % if handle_out_fn!="":
    ${handle_out_fn}(
        % for out_name in match_outputs.keys():
        ${out_name}_pt,
        ${match_outputs[out_name]["prod_shape"]},
        % endfor
        match_ctx.status
    );
    % endif
    
    % for out_idx,out_name in enumerate(match_outputs.keys()):
    #if !defined(__${default_model}_GRAPH_${default_model}_out_${out_idx}_FROM_EXTERNAL_MEM__) || !__${default_model}_GRAPH_${default_model}_out_${out_idx}_FROM_EXTERNAL_MEM__
    % if golden_cpu_model:
    ${target.free_fn}(golden_check_${out_name}_pt);
    % endif
    ${target.free_fn}(${out_name}_pt);
    #endif
    % endfor

    #if defined(__${default_model}_GRAPH_INPUTS_OUTPUTS_EXT_MEM__) && __${default_model}_GRAPH_INPUTS_OUTPUTS_EXT_MEM__>0
    ${target.free_external_mem}(match_ext_mem, __${default_model}_GRAPH_INPUTS_OUTPUTS_EXT_MEM__*${int(golden_cpu_model)+1});
    #endif
    // target specific cleaning functions
    % for clean_func in target.clean_funcs:
    ${clean_func}();
    % endfor
    return 0;
}