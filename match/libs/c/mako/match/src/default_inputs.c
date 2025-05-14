% if len(match_inputs)>0:
#include <${default_model}/default_inputs.h>

% for inp_name,inp in match_inputs.items():
#if !defined(__${default_model}_GRAPH_${inp_name}_FROM_EXTERNAL_MEM__) || !__${default_model}_GRAPH_${inp_name}_FROM_EXTERNAL_MEM__
const ${inp["c_type"]} ${target.input_macros} ${inp["name"]}_default[${inp["c_arr_size"]}] = ${inp["c_arr_values"]};
#endif
% endfor
% endif