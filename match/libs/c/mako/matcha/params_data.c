#include <${model_name}_params_data.h>

% for tensor in tensors:
    % if tensor.is_constant:
        % if tensor.static_in_ext_mem:
            // External memory constant tensor
            const ${tensor.c_type} ${tensor.name}_data_[${tensor.size}] __attribute__((section("${target.ext_mem_linker_section}"))) = ${tensor.c_value};
        % elif tensor.static_in_soc_mem:
            const ${tensor.c_type} ${tensor.name}_data_[${tensor.size}] = ${tensor.c_value};
        % endif
    % endif
% endfor