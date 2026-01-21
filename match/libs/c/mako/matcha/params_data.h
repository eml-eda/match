#ifndef __MATCH_${model_name}_PARAMS_DATA_H__
#define __MATCH_${model_name}_PARAMS_DATA_H__

#include <match/types.h>
% for node in nodes:
    % if not node.fallback:
        #include <nodes/${model_name}/${node.node_name}_data.h>
    % endif
% endfor

% for tensor in tensors:
    % if tensor.is_constant and (tensor.static_in_ext_mem or tensor.static_in_soc_mem):
        extern const ${tensor.c_type} ${tensor.name}_data_[${tensor.size}];
    % endif
% endfor

#endif