#include <${model_name}_params_data.h>

% for mem_tensor in mem_tensors:
    % if mem_tensor.is_constant:
        % if mem_tensor.stored_in_external_memory and not target.params_off_chip_file:
            ${"__MATCH_"+model_name+"_PARAM_OFF_CHIP__" if target.params_data_off_chip_flag != "" else ""} ${mem_tensor.c_type} ${mem_tensor.name}_data_[${mem_tensor.prod_shape}] = ${mem_tensor.c_value};
        % elif not mem_tensor.stored_in_external_memory:
            ${"__MATCH_"+model_name+"_PARAM_ON_CHIP__" if target.params_data_on_chip_flag != "" else ""} ${mem_tensor.c_type} ${mem_tensor.name}_data_[${mem_tensor.prod_shape}] = ${mem_tensor.c_value};
        % endif
    % endif
% endfor