#include <${model_name}_graph.h>

// DLTensor declarations
% for mem_tensor in mem_tensors:
% if mem_tensor.used_by_tvm:
DLTensor ${mem_tensor.name}_dltensor;
% endif
% endfor
// params of nodes
// void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle
% for node in nodes:
% if node.fallback:
// node ${node.name}
TVMValue ${node.name}_args_[${len(node.inputs)+len(node.outputs)}];
int* ${node.name}_arg_type_ids_;
int ${node.name}_num_args_ = ${len(node.inputs)+len(node.outputs)};
void* ${node.name}_out_ret_value_;
int* ${node.name}_out_ret_tcode_;
void* ${node.name}_resource_handle_;
% endif
% endfor

int match_${model_name}_run_graph(
    % for rt_i in rt_inputs:
    ${rt_i.c_type}* ${rt_i.name}_${"ext_" if rt_i.stored_in_external_memory else ""}pt,
    % endfor
    % for rt_o_idx,rt_o in enumerate(rt_outputs):
    ${"" if rt_o_idx==0 else ", "}${rt_o.c_type}* ${rt_o.name}_${"ext_" if rt_o.stored_in_external_memory else ""}pt
    % endfor
){
    #if __${model_name}_GRAPH_PROFILE__ || __${model_name}_FALLBACK_GRAPH_PROFILE__
    ${target.timestamp_type} start,end;
    double time_elapsed_ms = 0.0f;
    #endif
    % if ext_mem_needed_bytes>0:
    void* match_ext_mem = ${target.allocate_ext_mem}(${ext_mem_needed_bytes});
    int ext_mem_offset = 0;
    % endif
    % if mem_needed_bytes>0:
    void* match_mem = ${target.alloc_fn}(${mem_needed_bytes});
    % endif
    % for mem_tensor in mem_tensors:
    % if (mem_tensor.is_input or mem_tensor.is_output) and (len(mem_tensor.move_temp_to_ext_mem)>0 or len(mem_tensor.load_from_ext_mem_at)>0):
    void* ${mem_tensor.name}_pt = match_mem+${mem_tensor.mem_offset};
    % endif
    % if mem_tensor.is_intermediate or (mem_tensor.is_constant and mem_tensor.stored_in_external_memory):
    void* ${mem_tensor.name}_pt = match_mem+${mem_tensor.mem_offset};
    % if len(mem_tensor.load_from_ext_mem_at)>0:
    void* ${mem_tensor.name}_ext_pt = match_ext_mem+ext_mem_offset;
    % endif
    % if mem_tensor.is_constant and mem_tensor.stored_in_external_memory:
    ${target.load_file_to_ext_mem_fn}("${model_name}_${mem_tensor.name}_data.hex", ${mem_tensor.name}_ext_pt, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
    % endif
    % if len(mem_tensor.load_from_ext_mem_at)>0:
    ext_mem_offset += ${mem_tensor.elems * mem_tensor.dtype.itemsize};
    % endif
    % elif mem_tensor.is_constant and not mem_tensor.stored_in_external_memory:
    void* ${mem_tensor.name}_pt = ${mem_tensor.name}_data_;
    % endif
    % endfor
    % for node in nodes:
    #if __${model_name}_GRAPH_DEBUG__
    % if node.fallback:
    #if __${model_name}_FALLBACK_GRAPH_DEBUG__
    % endif
    printf("[${model_name} GRAPH] Running ${'TVM' if node.fallback else 'MATCH'} node ${node.name}\n");
    % if node.fallback:
    #endif
    % endif
    #endif
    % for mem_tensor in mem_tensors:
    % if node.node_id in mem_tensor.move_temp_to_ext_mem:
    ${target.load_to_ext_mem_fn}(${mem_tensor.name}_pt, ${mem_tensor.name}_ext_pt,${mem_tensor.elems * mem_tensor.dtype.itemsize});
    % endif
    % if node.node_id in mem_tensor.load_from_ext_mem_at:
    % if mem_tensor.mem_offset_at[node.node_id]!=mem_tensor.mem_offset:
    // update mem pt of tensor in soc memory
    <% mem_tensor.mem_offset = mem_tensor.mem_offset_at[node.node_id] %>
    ${mem_tensor.name}_pt = match_mem+${mem_tensor.mem_offset};
    % endif
    // load tensor from external memory
    ${target.load_from_ext_mem_fn}(${mem_tensor.name}_pt, ${mem_tensor.name}_ext_pt,${mem_tensor.elems * mem_tensor.dtype.itemsize});
    % endif
    % endfor
    ## NODES in TVM Graph Runtime are called with
    ## void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle
    % if node.fallback:
    ## SET V_HANDLE OF TENSORS
    // set correct pointers for node
    % for inp_idx,node_in in enumerate(node.inputs):
    ${node.name}_args_[${inp_idx}].v_handle = (void*)(&${node_in.name}_dltensor);
    ${node_in.name}_dltensor.data = ${node_in.name}_pt;
    % endfor
    % for out_idx,node_out in enumerate(node.outputs):
    ${node.name}_args_[${len(node.inputs)+out_idx}].v_handle = (void*)(&${node_out.name}_dltensor);
    ${node_out.name}_dltensor.data = ${node_out.name}_pt;
    % endfor
    #if __${model_name}_FALLBACK_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    if( ${node.fn_name}(${node.name}_args_, ${node.name}_arg_type_ids_, ${node.name}_num_args_,
                        ${node.name}_out_ret_value_, ${node.name}_out_ret_tcode_, ${node.name}_resource_handle_)) return -1;
    #if __${model_name}_FALLBACK_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    time_elapsed_ms = ((double)(end - start)) ${target.timestamp_to_ms};
    printf("[${model_name} GRAPH] TVM node ${node.name} done, took %fms\n", time_elapsed_ms);
    #endif
    % else:
    % for node_in in [inp__ for inp__ in node.inputs if inp__.is_constant]:
    ${node_in.name}_data = ${node_in.name}_pt;
    % endfor
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    if( ${node.fn_name}(
            % for inp_idx,node_in in enumerate([inp__ for inp__ in node.inputs if not inp__.is_constant]):
            ${"" if inp_idx==0 else ","}${node_in.name}_pt
            % endfor
            % for tens_out in node.outputs:
            ,${tens_out.name}_pt
            % endfor
        )
    ) return -1;
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    time_elapsed_ms = ((double)(end - start)) ${target.timestamp_to_ms};
    printf("[${model_name} GRAPH] MATCH node ${node.name} done, took %fms\n", time_elapsed_ms);
    #endif
    % endif
    #if __${model_name}_GRAPH_DEBUG__
    % if node.fallback:
    #if __${model_name}_FALLBACK_GRAPH_DEBUG__
    % endif
    printf("[${model_name} GRAPH] ${'TVM' if node.fallback else 'MATCH'} node ${node.name} done, output differs from checksum by %d\n", match_byte_checksum_check(${node.outputs[0].name}_pt, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
    % if node.fallback:
    #endif
    % endif
    #endif
    % endfor
    % for mem_tensor in [m_t__ for m_t__ in mem_tensors if -1 in m_t__.move_temp_to_ext_mem]:
    ${target.load_to_ext_mem_fn}(${mem_tensor.name}_pt, ${mem_tensor.name}_ext_pt,${mem_tensor.elems * mem_tensor.dtype.itemsize});
    % endfor
    // final cleanup
    % if mem_needed_bytes>0:
    ${target.free_fn}(match_mem);
    % endif
    % if ext_mem_needed_bytes>0:
    ${target.free_external_mem}(match_ext_mem, ${ext_mem_needed_bytes});
    % endif
    return 0;
}