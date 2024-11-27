#include <match_default_inputs.h>
#include <match_runtime.h>

// target specific inlcudes
% for inc_h in target.include_list:
#include <${inc_h}.h>
% endfor

int main(int argc,char** argv){
    // target specific inits
    % for init_func in target.init_funcs:
    ${init_func}();
    % endfor
    
    match_runtime_ctx match_ctx;

    % if runtime=="generative":
    % for inp_name,inp in match_inputs.items():
    ${inp["c_type"]} ${inp_name} = 1;
    % endfor
    
    % for out_name,out in match_outputs.items():
    ${out["c_type"]} ${out_name} = 0;
    % endfor
    
    match_${runtime}_runtime(
        % for inp_name in match_inputs.keys():
        &${inp_name},
        % endfor
        % for out_name in match_outputs.keys():
        &${out_name},
        % endfor
        % for dyn_dim in dynamic_dims.keys():
        1,
        % endfor
        &match_ctx
    );

    % else:
    match_default_runtime(&match_ctx);
    % endif

    // target specific cleaning functions
    % for clean_func in target.clean_funcs:
    ${clean_func}();
    % endfor
    return 0;
}