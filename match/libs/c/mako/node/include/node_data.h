#ifndef __MATCH_NODE_DATA_${node_fullname}_H__
#define __MATCH_NODE_DATA_${node_fullname}_H__

% for idx,const_ in enumerate(match_node.const_tensors.values()):
extern const ${c_dtype(const_.dtype)} ${name}_${const_.name}_const_data[${const_.data.size}];
% endfor

#endif