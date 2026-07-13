#ifndef __MATCH_${model_name}_PARAMS_DATA_H__
#define __MATCH_${model_name}_PARAMS_DATA_H__

#include <match/types.h>
% for node in nodes:
% if not node.fallback:
#include <nodes/${model_name}/${node.node_name}_data.h>
% endif
% endfor

% if target.params_data_on_chip_flag!="":
#define __MATCH_${model_name}_PARAM_ON_CHIP__ ${target.params_data_on_chip_flag}
% endif
% if target.params_data_off_chip_flag!="":
#define __MATCH_${model_name}_PARAM_OFF_CHIP__ ${target.params_data_off_chip_flag}
% endif

% for mem_tensor in mem_tensors:
% if mem_tensor.is_constant:
% if mem_tensor.stored_in_external_memory and not target.params_off_chip_file:
extern ${"__MATCH_"+model_name+"_PARAM_OFF_CHIP__" if target.params_data_off_chip_flag != "" else ""} ${mem_tensor.c_type} ${mem_tensor.name}_data_[${mem_tensor.prod_shape}];
% elif not mem_tensor.stored_in_external_memory:
extern ${"__MATCH_"+model_name+"_PARAM_ON_CHIP__" if target.params_data_on_chip_flag != "" else ""} ${mem_tensor.c_type} ${mem_tensor.name}_data_[${mem_tensor.prod_shape}];
% endif
% endif
% endfor

#endif