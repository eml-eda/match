#include <nodes/${model_name}/${name}_params.h>

% for idx,const_ in enumerate(match_node.const_tensors.values()):
const ${c_dtype(const_.dtype)} ${name}_${const_.name}_const_data[${const_.data.size}] = ${c_np_array(const_.data)};
% endfor