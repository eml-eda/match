#ifndef __MATCH_${model_name}_PARAMS_DATA_H__
#define __MATCH_${model_name}_PARAMS_DATA_H__

#include <match/types.h>

% for mem_tensor in mem_tensors:
% if mem_tensor.is_constant and not mem_tensor.stored_in_external_memory:
extern const ${mem_tensor.c_type} ${mem_tensor.name}_data_[${mem_tensor.prod_shape}];

% endif
% endfor

#endif