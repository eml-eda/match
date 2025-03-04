#include <${model_name}/graph_runtime.h>

int match_${model_name}_graph_runtime(
    % for rt_i in rt_inputs:
    ${rt_i.c_type}* ${rt_i.name}_pt,
    % endfor
    % for rt_o_idx,rt_o in enumerate(rt_outputs):
    ${"" if rt_o_idx==0 else ","}${rt_o.c_type}* ${rt_o.name}_pt
    % endfor
){
    % if ext_mem_needed_bytes>0:
    void* match_ext_mem = ${target.allocate_ext_mem}(${ext_mem_needed_bytes});
    int ext_mem_offset = 0;
    % endif
    void* match_mem = ${target.alloc_fn}(${soc_mem_needed_bytes});
    % for mem_tensor in mem_tensors:
    % if mem_tensor.is_intermediate:
    void* ${mem_tensor.name}_pt = match_mem+${mem_tensor.soc_memory_offset};
    % elif mem_tensor.is_constant:
    % if mem_tensor.stored_in_external_memory:
    void* ${mem_tensor.name}_pt = match_mem+${mem_tensor.soc_memory_offset};
    void* ${mem_tensor.name}_ext_pt = match_ext_mem+ext_mem_offset;
    ${target.load_to_ext_mem_fn}("${model_name}_params_${mem_tensor.name}_data.hex", ${mem_tensor.name}_ext_pt, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
    ext_mem_offset += ${mem_tensor.elems * mem_tensor.dtype.itemsize};
    % else:
    void* ${mem_tensor.name}_pt = ${mem_tensor.name}_data_;
    % endif
    % endif
    % endfor
    % for node in nodes:
    ## NODES in TVM Graph Runtime are called with
    ## void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle
    % if node.fallback:
    ## SET V_HANDLE OF TENSORS
    // set correct pointers for node
    % for inp_idx,node_in in enumerate(node.inputs):
    % if node_in.is_constant and node.node_id in node_in.load_from_ext_mem_at:
    // load constant from external memory
    ${target.load_from_ext_mem_fn}(${node_in.name}_pt, ${node_in.name}_ext_pt,${node_in.elems * node_in.dtype.itemsize});
    % endif
    ${node.name}_args_[${inp_idx}].v_handle = ${node_in.name}_pt;
    % endfor
    % for out_idx,node_out in enumerate(node.outputs):
    ${node.name}_args_[${len(node.inputs)+out_idx}].v_handle = ${node_out.name}_pt;
    % endfor
    if( ${node.fn_name}(${node.name}_args_, ${node.name}_arg_type_ids_, ${node.name}_num_args_,
                        ${node.name}_out_ret_value_, ${node.name}_out_ret_tcode, ${node.name}_resource_handle_)) return -1;
    % else:
    % for inp_idx,node_in in enumerate(node.inputs):
    % if node_in.is_constant and node.node_id in node_in.load_from_ext_mem_at:
    // load constant from external memory
    ${target.load_from_ext_mem_fn}(${node_in.name}_pt, ${node_in.name}_ext_pt,${node_in.elems * node_in.dtype.itemsize});
    % endif
    % endfor
    if( ${node.fn_name}(
            % for inp_idx,node_in in enumerate(node.inputs):
            ${"" if inp_idx==0 else ","}${node_in.name}_pt
            % endfor
            % for tens_out in node.outputs:
            ,${tens_out.name}_pt
            % endfor
        )
    ) return -1;
    % endif
    % endfor
    // final cleanup
    ${target.free_fn}(match_mem);
    % if ext_mem_needed_bytes>0:
    ${target.free_external_mem}(match_ext_mem);
    % endif
    return 0;
}