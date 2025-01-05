#include <nodes/${model_name}/${name}_params.h>

% for idx,const_ in enumerate(match_node.const_tensors.values()):
const ${c_dtype(const_.dtype)} ${const_.name}_data[${const_.data.size}] = ${c_np_array(const_.data)};
% endfor