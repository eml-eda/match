#include <${model_name}_params_data.h>

% for mem_tensor in mem_tensors:
% if mem_tensor.is_constant and not mem_tensor.stored_in_external_memory:
const ${mem_tensor.c_type} ${mem_tensor.name}_data_[${mem_tensor.prod_shape}] = ${mem_tensor.c_value};

% endif
% endfor