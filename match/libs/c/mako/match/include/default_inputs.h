#ifndef __MATCH_${default_model}_DEFAULT_INPUTS_H__
#define __MATCH_${default_model}_DEFAULT_INPUTS_H__

#include <match/types.h>
% if len(match_inputs)>0:
% for include in target.include_list:
#include <${include}.h>
% endfor

% for inp_name,inp in match_inputs.items():
#if !defined(__${default_model}_GRAPH_${inp_name}_FROM_EXTERNAL_MEM__) || !__${default_model}_GRAPH_${inp_name}_FROM_EXTERNAL_MEM__
extern const ${inp["c_type"]} ${target.input_macros} ${inp["name"]}_default[${inp["c_arr_size"]}];
#endif
% endfor
% endif

#endif