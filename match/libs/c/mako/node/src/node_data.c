#include <nodes/${model_name}/${name}_params.h>

% for idx,const_ in enumerate(match_node.const_tensors.values()):
% if executor=="aot":
const ${c_dtype(const_.dtype)} ${name}_${const_.name}_data[${const_.data.size}] = ${c_np_array(const_.data)};
% else:
${c_dtype(const_.dtype)}* ${name}_${const_.name}_data = NULL;
% endif
% endfor