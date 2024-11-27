#ifndef __MATCH_DEFAULT_INPUTS_H__
#define __MATCH_DEFAULT_INPUTS_H__

% if runtime=="default":
% for include in target.include_list:
#include <${include}.h>
% endfor

% for inp in match_inputs:
${inp["c_type"]} ${target.input_macros} ${inp["name"]}_default[${inp["c_arr_size"]}] = ${inp["c_arr_values"]};
% endfor
% endif

#endif