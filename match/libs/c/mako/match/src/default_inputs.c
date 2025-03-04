% if len(match_inputs)>0:
#include <${default_model}/default_inputs.h>
% for include in target.include_list:
#include <${include}.h>
% endfor

% for inp_name,inp in match_inputs.items():
const ${inp["c_type"]} ${target.input_macros} ${inp["name"]}_default[${inp["c_arr_size"]}] = ${inp["c_arr_values"]};
% endfor
% endif