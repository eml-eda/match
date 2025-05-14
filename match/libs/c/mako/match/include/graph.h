#ifndef __MATCH_${model_name}_RUN_GRAPH_H__
#define __MATCH_${model_name}_RUN_GRAPH_H__

% for include in target.include_list:
#include <${include}.h>
% endfor
#include <tvm/runtime/c_runtime_api.h>
#include <${model_name}_params_data.h>

// TVM signature
// void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle
// MATCH signature
// type* inp_A, ..., type* inp_Z, type* out_A, ..., type* out_N
% for mem_tensor in mem_tensors:
% if mem_tensor.is_input or mem_tensor.is_output:
#define __${model_name}_GRAPH_${mem_tensor.name}_FROM_EXTERNAL_MEM__ ${int(mem_tensor.stored_in_external_memory)}
% endif
% endfor
#define __${model_name}_GRAPH_INPUTS_OUTPUTS_EXT_MEM__ ${sum([mem_tensor.num_bytes for mem_tensor in mem_tensors if (mem_tensor.is_input or mem_tensor.is_output) and mem_tensor.stored_in_external_memory])}
// profiling flags
#define __${model_name}_GRAPH_PROFILE__ ${int(profile)}
#define __${model_name}_FALLBACK_GRAPH_PROFILE__ ${int(profile_fallback)}
// debugging flags
#define __${model_name}_GRAPH_DEBUG__ ${int(debug)}
#define __${model_name}_FALLBACK_GRAPH_DEBUG__ ${int(debug_fallback)}
#if __${model_name}_GRAPH_DEBUG__
% for activation_name, activation_checksum in checksums.items():
% if map_names[activation_name][2] in nodes_map:
% if nodes_map[map_names[activation_name][2]].fallback:
#if __${model_name}_FALLBACK_GRAPH_DEBUG__
% endif
#define __${model_name}_GRAPH_${map_names[activation_name][0]}_CHECKSUM__ ${activation_checksum}
#define __${model_name}_GRAPH_${map_names[activation_name][0]}_BYTES__ ${tensor_map[map_names[activation_name][1]].elems * tensor_map[map_names[activation_name][1]].dtype.itemsize}
% if nodes_map[map_names[activation_name][2]].fallback:
#endif
% endif
% endif
% endfor
#endif

% for node in nodes:
#ifndef __MATCH_${model_name}_RUN_GRAPH_${node.fn_name}__
% if node.fallback:
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t ${node.fn_name}(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
% else:
${node.fn_name}(
    % for inp_idx,node_in in enumerate([inp__ for inp__ in node.inputs if not inp__.is_constant]):
    ${"" if inp_idx==0 else ", "}${node_in.c_type}* ${node_in.name}_pt
    % endfor
    % for tens_out in node.outputs:
    , ${tens_out.c_type}* ${tens_out.name}_pt
    % endfor
);
% endif
#endif
% endfor

int match_${model_name}_run_graph(
    % for rt_i in rt_inputs:
    ${rt_i.c_type}* ${rt_i.name}_${"ext_" if rt_i.stored_in_external_memory else ""}pt,
    % endfor
    % for rt_o_idx,rt_o in enumerate(rt_outputs):
    ${"" if rt_o_idx==0 else ", "}${rt_o.c_type}* ${rt_o.name}_${"ext_" if rt_o.stored_in_external_memory else ""}pt
    % endfor
);
#endif