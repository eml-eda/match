#include <${model_name}_graph.h>

// DLTensor declarations
% for tens_idx in range(max([len(node.inputs)+len(node.outputs) for node in nodes if node.fallback]+[0])):
DLTensor tvm_fallback_dltensor_${tens_idx};
% endfor
// params of nodes
TVMValue tvm_fallback_args_[${max([len(node.inputs)+len(node.outputs) for node in nodes if node.fallback]+[0])}];
int* tvm_fallback_arg_type_ids_;
void* tvm_fallback_out_ret_value_;
int* tvm_fallback_out_ret_tcode_;
void* tvm_fallback_resource_handle_;

// Perf counters kernels
#if __${model_name}_FALLBACK_GRAPH_PROFILE__
int constants_loading_cycles = 0;
% for node in nodes:
int ${node.name}_perf_cnt;
% endfor
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

% if mem_needed_bytes>0 and (target.alloc_fn=="" or target.free_fn==""):
// static memory allocation if no alloc/free functions are provided
uint8_t match_static_malloc_mem[__MATCH_MEM_SIZE__];
% endif

<% ext_mem_offset = 0 %>
% for mem_tensor in mem_tensors:
<% ext_mem_offset = mem_tensor.get_new_mem_offset(ext_mem_offset) %>
% endfor

static inline void wait_async_off_chip_transfer(int transfer_id){
% if target.async_off_chip_to_on_chip_dm and getattr(target, "wait_async_off_chip_transfer_fn", ""):
    if (transfer_id >= 0) {
        ${target.wait_async_off_chip_transfer_fn}(transfer_id);
    }
% elif getattr(target, "wait_async_off_chip_transfer_fn", ""):
    (void)transfer_id;
    ${target.wait_async_off_chip_transfer_fn}();
% else:
    (void)transfer_id;
    return;
% endif
}

% if target.async_off_chip_to_on_chip_dm:
% for mem_tensor in [m_t__ for m_t__ in mem_tensors if len(m_t__.load_from_ext_mem_at) > 0]:
static int ${mem_tensor.name}_dma_transfer_id = -1;
static int ${mem_tensor.name}_dma_issue_ord = -1;
% endfor
static int ${model_name}_dma_issue_ord_counter = 0;
% endif

// GPIO variables
#ifdef USE_GPIO 
    pi_gpio_e gpio_test_0, gpio_test_1, gpio_test_2;
    #define TEST_GPIO_0 PI_GPIO_A89
    #define TEST_GPIO_1 PI_GPIO_A68
    #define TEST_GPIO_2 PI_GPIO_A52
    #define WRITE_GPIO(gpio_pin_x, x) {hal_compiler_barrier(); pi_gpio_pin_write(gpio_pin_x, x); hal_compiler_barrier();}
    #define SWITCH_GPIO(gpio_pin_x) {hal_compiler_barrier(); pi_gpio_pin_toggle(gpio_pin_x); hal_compiler_barrier();}
#endif

void match_${model_name}_graph_load_files(void* match_mem, void* match_ext_mem){
    <% ext_mem_offset = 0 %>
    % for mem_tensor in mem_tensors:
    % if mem_tensor.is_constant and mem_tensor.stored_in_external_memory:
    ${target.load_file_to_ext_mem_fn}("${model_name}_${mem_tensor.name}_data.hex", ${mem_tensor.get_ext_pt}, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
    % endif
    % endfor
    return;
}

#if __${model_name}_FALLBACK_GRAPH_PROFILE__
void match_${model_name}_graph_profile_summary(void){
    ${target.print_fn}("Node\tCycle\n\r");
    % for node in nodes:
    ${target.print_fn}("[${node.fn_name}]\t%d\n\r", ${node.name}_perf_cnt );
    % endfor

    ${target.print_fn}("\nProfiling Mem Transfers Performance\n\r");
    ${target.print_fn}("[file_constants_off_chip_loading LOAD]\tBytes:\t${sum([m_t.num_bytes for m_t in mem_tensors if m_t.is_constant and m_t.stored_in_external_memory])}\tCycles:\t%d\n\r", constants_loading_cycles);
    % for  node in nodes:
    % for mem_tensor in mem_tensors:
    % if node.node_id in mem_tensor.move_temp_to_ext_mem:
    ${target.print_fn}("[${node.fn_name} ${mem_tensor.name} STORE]\tBytes:\t${mem_tensor.elems * mem_tensor.dtype.itemsize}\tCycles:\t%d\n\r",${mem_tensor.name}_cp_to_ext_mem_cyc );
    % endif
    % if node.node_id in mem_tensor.load_from_ext_mem_at:
    ${target.print_fn}("[${node.fn_name} ${mem_tensor.name} LOAD]\tBytes:\t${mem_tensor.elems * mem_tensor.dtype.itemsize}\tCycles:\t%d\n\r", ${mem_tensor.name}_cp_from_ext_mem_cyc );
    % endif
    % endfor
    % endfor
    % for mem_tensor in [m_t__ for m_t__ in mem_tensors if -1 in m_t__.move_temp_to_ext_mem]:
    ${target.print_fn}("[\t${mem_tensor.name} STORE]\tBytes:\t${mem_tensor.elems * mem_tensor.dtype.itemsize}\tCycles:\t%d\n\r", ${mem_tensor.name}_cp_to_ext_mem_cyc );
    % endfor
}
#endif

% for node in nodes:
static int match_${model_name}_run_node_${node.node_id}(
    void* match_mem, void* match_ext_mem
% for rt_i in rt_inputs:
    , ${rt_i.c_type}* ${rt_i.name}_${"ext_" if rt_i.stored_in_external_memory else ""}pt
% endfor
% for rt_o in rt_outputs:
    , ${rt_o.c_type}* ${rt_o.name}_${"ext_" if rt_o.stored_in_external_memory else ""}pt
% endfor
) {
#if __${model_name}_GRAPH_RUN_ALL_NODES__ || __${model_name}_GRAPH_RUN_ONLY_NODE_ID__==${node.node_id}
#if __${model_name}_GRAPH_PROFILE__ || __${model_name}_FALLBACK_GRAPH_PROFILE__
    ${target.timestamp_type} start, end;
#endif

    % for (free_buffer_off, free_buffer_size, free_buffer_name) in node.free_buffers:
    match_alloc_workspace(${free_buffer_off}, ${free_buffer_size});
    % endfor
    #if __${model_name}_GRAPH_DEBUG__
    % if node.fallback:
        #if __${model_name}_FALLBACK_GRAPH_DEBUG__
    % endif
    ${target.print_fn}("[${model_name} GRAPH] Running ${'TVM' if node.fallback else 'MATCH'} node ${node.name}: '${node.fn_name}'\n\r");
    % if node.fallback:
        #endif
    % endif
    #endif

    % for mem_tensor in mem_tensors:
    % if node.node_id in mem_tensor.move_temp_to_ext_mem:
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_2, 0);
    #endif
    ${target.load_to_ext_mem_fn}(${mem_tensor.get_pt}, ${mem_tensor.get_ext_pt},${mem_tensor.elems * mem_tensor.dtype.itemsize});
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_2, 1);
    #endif
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${mem_tensor.name}_cp_to_ext_mem_cyc = (int)((end - start) ${target.timestamp_to_ms});;
    #endif
    % endif
    % endfor

    // 1) Issue loads required by this node first.
    % for mem_tensor in mem_tensors:
    % if node.node_id in mem_tensor.load_from_ext_mem_at and node.node_id in mem_tensor.used_at:
    % if mem_tensor.mem_offset_at[node.node_id]!=mem_tensor.mem_offset:
    <% mem_tensor.mem_offset = mem_tensor.mem_offset_at[node.node_id] %>
    % endif
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_2, 0);
    #endif
    % if target.async_off_chip_to_on_chip_dm:
    ${mem_tensor.name}_dma_transfer_id = ${target.offload_dma_fn}(${mem_tensor.get_ext_pt}, ${mem_tensor.get_pt}, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
    ${mem_tensor.name}_dma_issue_ord = ++${model_name}_dma_issue_ord_counter;
    % else:
    ${target.load_from_ext_mem_fn}(${mem_tensor.get_pt}, ${mem_tensor.get_ext_pt},${mem_tensor.elems * mem_tensor.dtype.itemsize});
    % endif
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_2, 1);
    #endif
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${mem_tensor.name}_cp_from_ext_mem_cyc = (int)((end - start) ${target.timestamp_to_ms});;
    #endif
    % endif
    % endfor

    // 2) Wait only for loads required by this node (including prefetched earlier).
    % if target.async_off_chip_to_on_chip_dm:
    // DMA transfers are issued in order, so waiting on the latest required
    // transfer is sufficient for all earlier required transfers.
    int node_last_needed_dma_transfer_id = -1;
    int node_last_needed_dma_issue_ord = -1;
    % for mem_tensor in [m_t__ for m_t__ in mem_tensors if len(m_t__.load_from_ext_mem_at) > 0]:
    % if node.node_id in mem_tensor.used_at:
    if (${mem_tensor.name}_dma_transfer_id >= 0 && ${mem_tensor.name}_dma_issue_ord >= node_last_needed_dma_issue_ord) {
        node_last_needed_dma_issue_ord = ${mem_tensor.name}_dma_issue_ord;
        node_last_needed_dma_transfer_id = ${mem_tensor.name}_dma_transfer_id;
    }
    ${mem_tensor.name}_dma_transfer_id = -1;
    ${mem_tensor.name}_dma_issue_ord = -1;
    % endif
    % endfor
    wait_async_off_chip_transfer(node_last_needed_dma_transfer_id);
    % endif

    // 3) Issue pure prefetches for future nodes without waiting here.
    % for mem_tensor in mem_tensors:
    % if node.node_id in mem_tensor.load_from_ext_mem_at and node.node_id not in mem_tensor.used_at:
    % if mem_tensor.mem_offset_at[node.node_id]!=mem_tensor.mem_offset:
    <% mem_tensor.mem_offset = mem_tensor.mem_offset_at[node.node_id] %>
    % endif
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_2, 0);
    #endif
    % if target.async_off_chip_to_on_chip_dm:
    ${mem_tensor.name}_dma_transfer_id = ${target.offload_dma_fn}(${mem_tensor.get_ext_pt}, ${mem_tensor.get_pt}, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
    ${mem_tensor.name}_dma_issue_ord = ++${model_name}_dma_issue_ord_counter;
    % else:
    ${target.load_from_ext_mem_fn}(${mem_tensor.get_pt}, ${mem_tensor.get_ext_pt},${mem_tensor.elems * mem_tensor.dtype.itemsize});
    % endif
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_2, 1);
    #endif
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${mem_tensor.name}_cp_from_ext_mem_cyc = (int)((end - start) ${target.timestamp_to_ms});;
    #endif
    % endif
    % endfor

    % if node.fallback:
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
    #ifdef USE_GPIO
    SWITCH_GPIO(gpio_test_1);
    #endif
    #if __${model_name}_FALLBACK_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${node.name}_perf_cnt = (int)((end - start) ${target.timestamp_to_ms});
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
    #ifdef USE_GPIO
    SWITCH_GPIO(gpio_test_1);
    #endif
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${node.name}_perf_cnt = (int)((end - start) ${target.timestamp_to_ms});;
    #endif
    % endif

    #if __${model_name}_GRAPH_DEBUG__
    % if node.fallback:
        #if __${model_name}_FALLBACK_GRAPH_DEBUG__
    % endif
    % if node.dtype_output_node=="float32":
    ${target.print_fn}("[${model_name} GRAPH] ${'TVM' if node.fallback else 'MATCH'} node ${node.name} done, relative error between output and checksum by %f\n\r", match_float_checksum_check(${node.outputs[0].get_pt}, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
    % else:
    ${target.print_fn}("[${model_name} GRAPH] ${'TVM' if node.fallback else 'MATCH'} node ${node.name} done, output differs from checksum by %d\n\r", match_byte_checksum_check(${node.outputs[0].get_pt}, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
    % endif
    % if node.fallback:
        #endif
    % endif
    #endif
    % if len(node.free_buffers)>0:
    match_free_workspace();
    % endif
#endif
    return 0;
}
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
#endif
#ifdef USE_GPIO
    gpio_test_0 = TEST_GPIO_0;
    pi_pad_function_set(PI_PAD_089, 1);
    pi_gpio_pin_configure(gpio_test_0, PI_GPIO_OUTPUT);
    WRITE_GPIO(gpio_test_0, 1);

    gpio_test_1 = TEST_GPIO_1;
    pi_pad_function_set(PI_PAD_068, 1);
    pi_gpio_pin_configure(gpio_test_1, PI_GPIO_OUTPUT);
    WRITE_GPIO(gpio_test_1, 1);

    gpio_test_2 = TEST_GPIO_2;
    pi_pad_function_set(PI_PAD_052, 1);
    pi_gpio_pin_configure(gpio_test_2, PI_GPIO_OUTPUT);
    WRITE_GPIO(gpio_test_2, 1);
#endif
% if ext_mem_needed_bytes>0:
    void* match_ext_mem = ${target.allocate_ext_mem}(${ext_mem_needed_bytes});
    % else:
    void* match_ext_mem = NULL;
    % endif
    % if mem_needed_bytes>0:
    % if target.alloc_fn!="" and target.free_fn!="":
    void* match_mem = ${target.alloc_fn}(__MATCH_MEM_SIZE__);
    % else:
    void* match_mem = match_static_malloc_mem;
    % endif
    if (!match_mem) {
        ${target.print_fn}("Error: match_mem allocation failed\n\r");
        return -1;
    }
    % else:
    void* match_mem = NULL;
    % endif
    match_set_match_mem_pt(match_mem);

    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    match_${model_name}_graph_load_files(match_mem, match_ext_mem);
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    constants_loading_cycles = (int)((end - start) ${target.timestamp_to_ms});;
    #endif
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_0, 0);
    #endif
    % for node in nodes:
    if (match_${model_name}_run_node_${node.node_id}(match_mem, match_ext_mem
    % for rt_i in rt_inputs:
        , ${rt_i.name}_${"ext_" if rt_i.stored_in_external_memory else ""}pt
    % endfor
    % for rt_o in rt_outputs:
        , ${rt_o.name}_${"ext_" if rt_o.stored_in_external_memory else ""}pt
    % endfor
    )) return -1;
    % endfor

    % for mem_tensor in [m_t__ for m_t__ in mem_tensors if -1 in m_t__.move_temp_to_ext_mem]:
    #if __${model_name}_GRAPH_PROFILE__
    start = ${target.start_get_timestamp_api}();
    #endif
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_2, 0);
    #endif
    ${target.load_to_ext_mem_fn}(${mem_tensor.get_pt}, ${mem_tensor.get_ext_pt}, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_2, 1);
    #endif
    #if __${model_name}_GRAPH_PROFILE__
    end = ${target.end_get_timestamp_api}();
    ${mem_tensor.name}_cp_to_ext_mem_cyc = (int)((end - start) ${target.timestamp_to_ms});;
    #endif
    % endfor
    #ifdef USE_GPIO
    WRITE_GPIO(gpio_test_0, 1);
    #endif

    #if __${model_name}_FALLBACK_GRAPH_PROFILE__
    match_${model_name}_graph_profile_summary();
    #endif
    % if mem_needed_bytes>0 and target.free_fn != "" and target.alloc_fn != "":
    ${target.free_fn}(match_mem);
% endif

% if ext_mem_needed_bytes > 0:
    ${target.free_external_mem}(match_ext_mem, ${ext_mem_needed_bytes});
% endif

return 0;
}
