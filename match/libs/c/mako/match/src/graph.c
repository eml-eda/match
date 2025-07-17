#include <${model_name}_graph.h>

// DLTensor declarations
% for tens_idx in range(max([len(node.inputs)+len(node.outputs) for node in nodes if node.fallback])):
DLTensor tvm_fallback_dltensor_${tens_idx};
% endfor
// params of nodes
// void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handl
TVMValue tvm_fallback_args_[${max([len(node.inputs)+len(node.outputs) for node in nodes if node.fallback])}];
int* tvm_fallback_arg_type_ids_;
void* tvm_fallback_out_ret_value_;
int* tvm_fallback_out_ret_tcode_;
void* tvm_fallback_resource_handle_;

// Perf counters kernels
#if __${model_name}_FALLBACK_GRAPH_PROFILE__
int sum_kernel_comp_cyc = 0;
% for  node in nodes:
int ${node.name}_perf_cnt;
% endfor
// Perf Counters Mem Transfers
int sum_mem_transfer_cyc = 0;
% for mem_tensor in mem_tensors:
% for  node in nodes:
% if node.node_id in mem_tensor.move_temp_to_ext_mem:
int ${mem_tensor.name}_cp_to_ext_mem_cyc;
% endif
% if node.node_id in mem_tensor.load_from_ext_mem_at:
int ${mem_tensor.name}_cp_from_ext_mem_cyc;
% endif
% endfor
% endfor
% for mem_tensor in [m_t__ for m_t__ in mem_tensors if -1 in m_t__.move_temp_to_ext_mem]:
int ${mem_tensor.name}_cp_to_ext_mem_cyc;
% endfor
#endif

void match_${model_name}_graph_load_files(void* match_mem, void* match_ext_mem){
    <% ext_mem_offset = 0 %>
    //read all files into ext mem <% for mem_tensor in mem_tensors: ext_mem_offset = mem_tensor.get_new_mem_offset(ext_mem_offset) %>
    % for mem_tensor in mem_tensors:
    % if mem_tensor.is_constant and mem_tensor.stored_in_external_memory:
    ${target.load_file_to_ext_mem_fn}("${model_name}_${mem_tensor.name}_data.hex", ${mem_tensor.get_ext_pt}, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
    % endif
    % endfor
    return;
}

#if __${model_name}_FALLBACK_GRAPH_PROFILE__
void match_${model_name}_graph_profile_summary(void){
    printf("Node\tCycle\n");
    % for node in nodes:
    printf("[${node.fn_name}]\t%d\n", ${node.name}_perf_cnt );
    % endfor
    printf("Total node cycles\t%d\n", sum_kernel_comp_cyc );
    printf("Total memory cycles\t%d\n", sum_mem_transfer_cyc );

    printf("\nProfiling Mem Transfers Performance\n");

    % for  node in nodes:
    % for mem_tensor in mem_tensors:
    % if node.node_id in mem_tensor.move_temp_to_ext_mem:
    printf("[${node.fn_name} ${mem_tensor.name} STORE]\tBytes:\t${mem_tensor.elems * mem_tensor.dtype.itemsize}\tCycles:\t%d\n",${mem_tensor.name}_cp_to_ext_mem_cyc );
    % endif
    % if node.node_id in mem_tensor.load_from_ext_mem_at:
    printf("[${node.fn_name} ${mem_tensor.name} LOAD]\tBytes:\t${mem_tensor.elems * mem_tensor.dtype.itemsize}\tCycles:\t%d\n", ${mem_tensor.name}_cp_from_ext_mem_cyc );
    % endif
    % endfor
    % endfor
    % for mem_tensor in [m_t__ for m_t__ in mem_tensors if -1 in m_t__.move_temp_to_ext_mem]:
    printf("[\t${mem_tensor.name} STORE]\tBytes:\t${mem_tensor.elems * mem_tensor.dtype.itemsize}\tCycles:\t%d\n", ${mem_tensor.name}_cp_to_ext_mem_cyc );
    % endfor
}
#endif

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
    % else:
    void* match_ext_mem = NULL;
    % endif
    % if mem_needed_bytes>0:
    void* match_mem = ${target.alloc_fn}(${mem_needed_bytes});
    if (!match_mem) {
        printf("Error: match_mem allocation failed\n");
        return -1;
    }
    % else:
    void* match_mem = NULL;
    % endif
    match_set_match_mem_pt(match_mem);

    match_${model_name}_graph_load_files(match_mem, match_ext_mem);

    % for node in nodes:
    % for (free_buffer_off, free_buffer_size) in node.free_buffers:
    match_alloc_workspace(${free_buffer_off}, ${free_buffer_size});
    % endfor
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
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    ${target.load_to_ext_mem_fn}(${mem_tensor.get_pt}, ${mem_tensor.get_ext_pt},${mem_tensor.elems * mem_tensor.dtype.itemsize});
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${mem_tensor.name}_cp_to_ext_mem_cyc = (int)(end - start);
    sum_mem_transfer_cyc += ${mem_tensor.name}_cp_to_ext_mem_cyc;
    #endif
    % endif
    % if node.node_id in mem_tensor.load_from_ext_mem_at:
    % if mem_tensor.mem_offset_at[node.node_id]!=mem_tensor.mem_offset:
    ## update mem pt of tensor in soc memory
    <% mem_tensor.mem_offset = mem_tensor.mem_offset_at[node.node_id] %>
    % endif
    // load tensor from external memory
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    ${target.load_from_ext_mem_fn}(${mem_tensor.get_pt}, ${mem_tensor.get_ext_pt},${mem_tensor.elems * mem_tensor.dtype.itemsize});
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${mem_tensor.name}_cp_from_ext_mem_cyc = (int)(end - start);
    sum_mem_transfer_cyc += ${mem_tensor.name}_cp_from_ext_mem_cyc;
    #endif
    % endif
    % endfor
    ## NODES in TVM Graph Runtime are called with
    ## void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle
    % if node.fallback:
    ## SET V_HANDLE OF TENSORS
    // set correct pointers for node
    % for inp_idx,node_in in enumerate(node.inputs):
    tvm_fallback_dltensor_${inp_idx}.data = ${node_in.get_pt};
    tvm_fallback_args_[${inp_idx}].v_handle = (void*)(&tvm_fallback_dltensor_${inp_idx});
    % endfor
    % for out_idx,node_out in enumerate(node.outputs):
    tvm_fallback_dltensor_${len(node.inputs)+out_idx}.data = ${node_out.get_pt};
    tvm_fallback_args_[${len(node.inputs)+out_idx}].v_handle = (void*)(&tvm_fallback_dltensor_${len(node.inputs)+out_idx});
    % endfor
    #if __${model_name}_FALLBACK_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    if( ${node.fn_name}(tvm_fallback_args_, tvm_fallback_arg_type_ids_, ${len(node.inputs)+len(node.outputs)},
                        tvm_fallback_out_ret_value_, tvm_fallback_out_ret_tcode_, tvm_fallback_resource_handle_)) return -1;
    #if __${model_name}_FALLBACK_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    time_elapsed_ms = ((double)(end - start)) ${target.timestamp_to_ms};
    ${node.name}_perf_cnt = (int)(end - start);
    sum_kernel_comp_cyc += ${node.name}_perf_cnt;
    printf("[${model_name} GRAPH] TVM node ${node.name} done, took %fms\n", time_elapsed_ms);
    #endif
    % else:
    % for node_in in [inp__ for inp__ in node.inputs if inp__.is_constant]:
    ${node_in.name}_data = ${node_in.get_pt};
    % endfor
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    if( ${node.fn_name}(
            % for inp_idx,node_in in enumerate([inp__ for inp__ in node.inputs if not inp__.is_constant]):
            ${"" if inp_idx==0 else ","}${node_in.get_pt}
            % endfor
            % for tens_out in node.outputs:
            ,${tens_out.get_pt}
            % endfor
        )
    ) return -1;
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    time_elapsed_ms = ((double)(end - start)) ${target.timestamp_to_ms};
    ${node.name}_perf_cnt = (int)(end - start);
    sum_kernel_comp_cyc += ${node.name}_perf_cnt;
    printf("[${model_name} GRAPH] MATCH node ${node.name} done, took %fms\n", time_elapsed_ms);
    #endif
    % endif
    #if __${model_name}_GRAPH_DEBUG__
    % if node.fallback:
    #if __${model_name}_FALLBACK_GRAPH_DEBUG__
    % endif
    % if node.dtype_output_node=="float32":
    printf("[${model_name} GRAPH] ${'TVM' if node.fallback else 'MATCH'} node ${node.name} done, relative error between output and checksum by %.4f\n", match_float_checksum_check(${node.outputs[0].get_pt}, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
    % else:
    printf("[${model_name} GRAPH] ${'TVM' if node.fallback else 'MATCH'} node ${node.name} done, output differs from checksum by %d\n", match_byte_checksum_check(${node.outputs[0].get_pt}, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
    % endif
    % if node.fallback:
    #endif
    % endif
    #endif
    match_free_workspace();
    % endfor

    % for mem_tensor in [m_t__ for m_t__ in mem_tensors if -1 in m_t__.move_temp_to_ext_mem]:
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    ${target.load_to_ext_mem_fn}(${mem_tensor.get_pt}, ${mem_tensor.get_ext_pt}, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${mem_tensor.name}_cp_to_ext_mem_cyc = (int)(end - start);
    sum_mem_transfer_cyc += ${mem_tensor.name}_cp_to_ext_mem_cyc;
    #endif
    % endfor

    #if __${model_name}_FALLBACK_GRAPH_PROFILE__
    match_${model_name}_graph_profile_summary();
    #endif
    // final cleanup
    % if mem_needed_bytes>0:
    ${target.free_fn}(match_mem);
    % endif
    % if ext_mem_needed_bytes>0:
    ${target.free_external_mem}(match_ext_mem, ${ext_mem_needed_bytes});
    % endif
    return 0;
}