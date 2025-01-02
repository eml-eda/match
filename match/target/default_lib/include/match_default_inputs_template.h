#ifndef __MATCH_DEFAULT_INPUTS_H__
#define __MATCH_DEFAULT_INPUTS_H__

#include <match/types.h>
% if runtime=="default":
% for include in target.include_list:
#include <${include}.h>
% endfor

% for inp_name,inp in match_inputs.items():
extern const ${inp["c_type"]} ${target.input_macros} ${inp["name"]}_default[${inp["c_arr_size"]}];
% endfor
% endif

#endif