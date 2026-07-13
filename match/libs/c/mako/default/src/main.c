#include <${default_model}/default_inputs.h>
#include <${default_model}/runtime.h>

// target specific inlcudes
% for inc_h in target.include_list:
#include <${inc_h}.h>
% endfor

% for inc_h_q in target.include_list_quotes:
#include "${inc_h_q}.h"
% endfor

% for exec_module in target.exec_modules:
% if exec_module.separate_build:
#include "${default_model}_${exec_module.name}_runtime_payload.h"
% endif
% endfor

% if golden_cpu_model:
#define GOLDEN_CHECK_BENCH_ITERATIONS ${bench_iterations}
% endif

int main(int argc, char** argv){
    // target specific inits
    % for init_func in target.init_funcs:
    ${init_func}();
    % endfor
    
    match_runtime_ctx match_ctx;

    ## Offload exec_module runtime if needed
    % for exec_module in target.exec_modules:
    % if exec_module.separate_build:
    // Offloading ${exec_module.name} runtime
    asm volatile("":::"memory");
    load_binary();
    asm volatile("":::"memory");
    *(volatile uint32_t*)(${exec_module.shared_memory_extern_addr}) = __MATCH_INVALID_TASK_ID__;
    asm volatile("":::"memory");
    ${exec_module.match_platform_apis().init_platform}(${exec_module.name_boot_addr});
    *(volatile uint32_t*)(${exec_module.shared_memory_extern_addr}) = __MATCH_INVALID_TASK_ID__;
    asm volatile("":::"memory");
    % endif
    % endfor

    // setting inputs pointers
    % for inp_name,inp in match_inputs.items():
    ${inp["c_type"]}* ${inp_name}_pt = ${inp_name}_default;
    % endfor
    // setting outputs pointers
    % for out_idx,(out_name,out) in enumerate(match_outputs.items()):
    % if out["is_copy_of"]:
    void* ${out_name}_pt = ${out["is_copy_of"]}_pt;
    % elif out["associated_input"]!="":
    void* ${out_name}_pt = ${out["associated_input"]}_pt;
    % else:
    % if target.alloc_fn != "" or target.free_fn != "":
    ${out["c_type"]}* ${out_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % else:
    ${out["c_type"]} ${out_name}_pt_[${out["prod_shape"]}];
    ${out["c_type"]}* ${out_name}_pt = ${out_name}_pt_;
    % endif
    % if golden_cpu_model:
    % if target.alloc_fn != "" or target.free_fn != "":
    ${out["c_type"]}* golden_check_${out_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % else:
    ${out["c_type"]} golden_check_${out_name}_pt_[${out["prod_shape"]}];
    ${out["c_type"]}* golden_check_${out_name}_pt = golden_check_${out_name}_pt_;
    % endif
    % endif
    % endif
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
    
    // target specific cleaning functions
    % for clean_func in target.clean_funcs:
    ${clean_func}();
    % endfor
    return 0;
}