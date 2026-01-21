#include <${model_name}_graph.h>

% for include in target.include_list:
    #include <${include}.h>
% endfor


// DLTensor declarations
% for mem_tensor in mem_tensors:
    % if mem_tensor.used_by_tvm:
        static DLTensor ${mem_tensor.name}_dltensor;
    % endif
% endfor

// TVM node calls arguments
% for node in nodes:
    %if node.fallback:
        static TVMValue ${node.name}_args_[${len(node.inputs)+len(node.outputs)}];
        static int* ${node.name}_arg_type_ids_;
        static int ${node.name}_num_args_ = ${len(node.inputs)+len(node.outputs)};
        static void* ${node.name}_out_ret_value_;
        static int* ${node.name}_out_ret_tcode_;
        static void* ${node.name}_resource_handle_;
    % endif
% endfor


// Declare MATCH tensors pointers
% for mem_tensor in mem_tensors:
    % if (mem_tensor.is_input or mem_tensor.is_output):
        % if (len(mem_tensor.move_temp_to_ext_mem) > 0 or len(mem_tensor.load_from_ext_mem_at) > 0):
        static void* ${mem_tensor.name}_pt;
        static void* ${mem_tensor.name}_ext_pt;
        % else:
        static void* ${mem_tensor.name}_pt;
        %endif
    % endif
    % if mem_tensor.is_intermediate or (mem_tensor.is_constant and mem_tensor.stored_in_external_memory):
        static void* ${mem_tensor.name}_pt;
        % if len(mem_tensor.load_from_ext_mem_at) > 0:
            static void* ${mem_tensor.name}_ext_pt;
        % endif
    % elif mem_tensor.is_constant and not mem_tensor.stored_in_external_memory:
        static void* ${mem_tensor.name}_pt;
    % endif
% endfor

// Profiling things

static volatile uint64_t node_start_time[${len(nodes)}];
static volatile uint64_t node_end_time[${len(nodes)}];

// Additional node info for async exec

% for node in nodes:
    static int ${node.name}_children[] = {${", ".join(str(child) for child in node.children)}};
% endfor
static int* ${model_name}_node_children[] = {${", ".join(f"{node.name}_children" for node in nodes)}};

static int ${model_name}_num_node_children[] = {${", ".join(str(len(node.children)) for node in nodes)}};
static int ${model_name}_node_device_id[] = {${", ".join(str(node.device_id) for node in nodes)}};

// Static node start calls, TVM is blocking, MATCH is async
% for node_id, node in enumerate(nodes):

    /*
     * ${"TVM" if node.fallback else "MATCH"} NODE: ${node.name}
     */

    static void match_${model_name}${"_run_tvm" if node.fallback else "_start"}_node_${node_id}() {
    #if __${model_name}_GRAPH_DEBUG__ ${f"&& __{model_name}_FALLBACK_GRAPH_DEBUG__" if node.fallback else ""}
    ${target.print_fn}("\r\n[${model_name} ASYNC] Started ${"TVM" if node.fallback else "MATCH"} node ${node_id} -> ${node.name} -> ${node.fn_name}\r\n");
    #endif

    % for mem_tensor in mem_tensors:
        % if node.node_id in mem_tensor.move_temp_to_ext_mem:
            ${target.load_to_ext_mem_fn}(${mem_tensor.name}_pt, ${mem_tensor.name}_ext_pt, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
        % endif
        % if node.node_id in mem_tensor.load_from_ext_mem_at:
            % if mem_tensor.mem_offset_at[node.node_id]!=mem_tensor.mem_offset:
                // update mem pt of tensor in soc memory
                <% mem_tensor.mem_offset = mem_tensor.mem_offset_at[node.node_id] %>
                ${mem_tensor.name}_pt = match_mem+${mem_tensor.mem_offset};
            % endif
            // load tensor from external memory
            ${target.load_from_ext_mem_fn}(${mem_tensor.name}_pt, ${mem_tensor.name}_ext_pt, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
        % endif
    % endfor

    // Flag device as busy
    int device_id = ${model_name}_node_device_id[${node_id}];
    match_${model_name}_device_is_busy[device_id] = 1;

    // Flag node as executed
    match_${model_name}_num_remaining_parents[${node_id}] = -1;

    % if node.fallback:
        ## TVM NODE - blocking for host, interruptible

        // TVM Node function call parameters
        // tvm_node(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle)
        // This is actually a blocking call because it is run in the host

        // Set v_handle for TVM tensors
        % for inp_idx,node_in in enumerate(node.inputs):
            ${node.name}_args_[${inp_idx}].v_handle = (void*)(&${node_in.name}_dltensor);
            ${node_in.name}_dltensor.data = ${node_in.name}_pt;
        % endfor
        % for out_idx,node_out in enumerate(node.outputs):
            ${node.name}_args_[${len(node.inputs)+out_idx}].v_handle = (void*)(&${node_out.name}_dltensor);
            ${node_out.name}_dltensor.data = ${node_out.name}_pt;
        % endfor

        #if __${model_name}_FALLBACK_GRAPH_PROFILE__
        node_start_time[${node_id}] = ${target.start_get_timestamp_api}();
        #endif

        ${target.print_fn}("Start time is %d.\r\n", node_start_time[${node_id}]);

        ${node.fn_name}(
            ${node.name}_args_, 
            ${node.name}_arg_type_ids_, 
            ${node.name}_num_args_,
            ${node.name}_out_ret_value_, 
            ${node.name}_out_ret_tcode_, 
            ${node.name}_resource_handle_
        );

        #if __${model_name}_FALLBACK_GRAPH_PROFILE__
        node_end_time[${node_id}] = ${target.end_get_timestamp_api}();
        ${target.print_fn}("[${model_name} GRAPH] TVM node ${node.name} done, start: %d, end: %d, diff: %d, time: %fms\r\n",
            node_start_time[${node_id}], node_end_time[${node_id}], (node_end_time[${node_id}] - node_start_time[${node_id}]),
            ((double)(int)(node_end_time[${node_id}] - node_start_time[${node_id}])) ${target.timestamp_to_ms});
        #endif

        // Check debug checksum
        #if __${model_name}_GRAPH_DEBUG__ && __${model_name}_FALLBACK_GRAPH_DEBUG__
        ${target.print_fn}("[${model_name} GRAPH] TVM node ${node.name} done, output differs from checksum by %d\r\n", match_byte_checksum_check(${node.outputs[0].name}_pt, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
        #endif

    % else:
        ## MATCH NODE - async, just start

        % for node_in in [inp__ for inp__ in node.inputs if inp__.is_constant]:
            ${node_in.name}_data = ${node_in.name}_pt;
        % endfor

        #if __${model_name}_GRAPH_PROFILE__
        node_start_time[${node_id}] = ${target.start_get_timestamp_api}();
        #endif

        ${node.fn_name}_async(
            % for inp_idx,node_in in enumerate([inp__ for inp__ in node.inputs if not inp__.is_constant]):
            ${"" if inp_idx==0 else ","}${node_in.name}_pt
            % endfor
            % for tens_out in node.outputs:
            , ${tens_out.name}_pt
            % endfor
        );

    % endif
    }

    % if not node.fallback:
        static void match_${model_name}_finish_node_${node_id}() {
            ${node.fn_name}_finish();

            #if __${model_name}_FALLBACK_GRAPH_PROFILE__
            node_end_time[${node_id}] = ${target.end_get_timestamp_api}();
            ${target.print_fn}("[${model_name} GRAPH] MATCH node ${node.name} done, start: %d, end: %d, diff: %d, time: %fms\r\n",
                node_start_time[${node_id}], node_end_time[${node_id}], (node_end_time[${node_id}] - node_start_time[${node_id}]),
                ((double)(int)(node_end_time[${node_id}] - node_start_time[${node_id}])) ${target.timestamp_to_ms});
            #endif

            // Check debug checksum
            #if __${model_name}_GRAPH_DEBUG__
            ${target.print_fn}("[${model_name} GRAPH] MATCH node ${node.name} done, output differs from checksum by %d\r\n", match_byte_checksum_check(${node.outputs[0].name}_pt, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
            #endif
        }
    % endif


% endfor



/*
 * EOC Callback
 */


static int match_${model_name}_node_match2seq_id(int node_match_id) {
    // Convert TVM node id to sequence id
    switch (node_match_id) {
        % for node_id, node in enumerate(nodes):
            % if not node.fallback:
                case ${node.fn_name.split("_")[-1]}: return ${node_id};
            %endif
        % endfor
        default: return -1; 
    }
}

const int match_${model_name}_num_nodes = ${len(nodes)};

int match_${model_name}_num_remaining_parents[] = {${", ".join(str(node.num_parents) for node in nodes)}};

int match_${model_name}_device_is_busy[${target.num_devices}] = {0};

static void (*match_${model_name}_start_node_fn[])(void) = {${", ".join(f"match_{model_name}_start_node_{i}" if not node.fallback else f"match_{model_name}_run_tvm_node_{i}" for i, node in enumerate(nodes))}};
static void (*match_${model_name}_finish_node_fn[])(void) = {${", ".join(f"match_{model_name}_finish_node_{i}" if not node.fallback else f"NULL" for i, node in enumerate(nodes))}};

static int next_tvm_node_id = -1;

/* Warning - Begin possibly interrupt region */
static match_${model_name}_schedule_next_node() {
    next_tvm_node_id = -1;
    for(int i = 0; i < match_${model_name}_num_nodes; i++) {
        int req_dev_id = ${model_name}_node_device_id[i];
        if(match_${model_name}_num_remaining_parents[i] == 0 && !match_${model_name}_device_is_busy[req_dev_id]) {
            match_${model_name}_device_is_busy[${model_name}_node_device_id[i]] = 1;

            if (req_dev_id == 0 && next_tvm_node_id < 0) {
                // This is a TVM node, set next_tvm_node_id
                next_tvm_node_id = i;
            } else {
                // This is a MATCH node, start it async
                match_${model_name}_start_node_fn[i]();
            }
        }
    }
}
void match_${model_name}_runtime_eoc_callback(int node_match_id) {
    int node_id = match_${model_name}_node_match2seq_id(node_match_id);

    // Perform final activities after node execution
    match_${model_name}_finish_node_fn[node_id]();

    ${target.print_fn}("[${model_name} ASYNC] Device EOC callback for node %d (%d)\r\n", node_id, node_match_id);

    // Decrease the number of remaining parents for the node children
    for(int i = 0; i < ${model_name}_num_node_children[node_id]; i++) {
        int child_id = ${model_name}_node_children[node_id][i];
        match_${model_name}_num_remaining_parents[child_id]--;
    }

    // Free device 
    int device_id = ${model_name}_node_device_id[node_id];
    match_${model_name}_device_is_busy[device_id] = 0;

    // Check if last node - TODO improve way to check graph execution finished
    if (node_id == match_${model_name}_num_nodes - 1) {
        match_${model_name}_graph_execution_finished = 1;
    }

    // Schedule next node
    match_${model_name}_schedule_next_node();
}
void match_${model_name}_runtime_eoc_host_callback(int node_id) {
    ${target.print_fn}("[${model_name} ASYNC] Host EOC callback for node %d\r\n", node_id);

    // Decrease the number of remaining parents for the node children
    for(int i = 0; i < ${model_name}_num_node_children[node_id]; i++) {
        int child_id = ${model_name}_node_children[node_id][i];
        match_${model_name}_num_remaining_parents[child_id]--;
    }

    // Free device 
    int device_id = ${model_name}_node_device_id[node_id];
    match_${model_name}_device_is_busy[device_id] = 0;

    // Check if last node - TODO improve way to check graph execution finished
    if (node_id == match_${model_name}_num_nodes - 1) {
        match_${model_name}_graph_execution_finished = 1;
    }

    // Schedule next node
    match_${model_name}_schedule_next_node();
}
/* Warning - End possibly interrupt region */



/*
 * ASYNC GRAPH RUNTIME
 */

volatile int match_${model_name}_graph_execution_finished = 0;

int match_${model_name}_run_graph_async(
% for rt_i in rt_inputs:
    ${rt_i.c_type}* ${rt_i.name}_${"ext_" if rt_i.stored_in_external_memory else ""}pt_,
% endfor
% for rt_o_idx,rt_o in enumerate(rt_outputs):
    ${"" if rt_o_idx==0 else ", "}${rt_o.c_type}* ${rt_o.name}_${"ext_" if rt_o.stored_in_external_memory else ""}pt_
% endfor
){

% if ext_mem_needed_bytes > 0:
    // L3 memory pool allocation
    void* match_ext_mem = ${target.allocate_ext_mem}(${ext_mem_needed_bytes});
    int ext_mem_offset = 0;
% endif

% if mem_needed_bytes > 0:
    // L2 memory pool allocation
    % if target.alloc_fn != "":
        void* match_mem = ${target.alloc_fn}(${mem_needed_bytes});
    % else:
        uint8_t match_mem_[${mem_needed_bytes}];
        volatile void* match_mem = (void*) match_mem_;
    % endif
% endif

// Inizialize input and output tensor pointers
% for rt_i in rt_inputs:
${rt_i.name}_${"ext_" if rt_i.stored_in_external_memory else ""}pt = ${rt_i.name}_${"ext_" if rt_i.stored_in_external_memory else ""}pt_;
% endfor
% for rt_o_idx,rt_o in enumerate(rt_outputs):
${rt_o.name}_${"ext_" if rt_o.stored_in_external_memory else ""}pt = ${rt_o.name}_${"ext_" if rt_o.stored_in_external_memory else ""}pt_;
% endfor

// Inizialize other tensor pointers in MATCH memory as planned
% for mem_tensor in mem_tensors:
    % if (mem_tensor.is_input or mem_tensor.is_output):
        % if (len(mem_tensor.move_temp_to_ext_mem) > 0 or len(mem_tensor.load_from_ext_mem_at) > 0):
        ${mem_tensor.name}_pt = match_mem+${mem_tensor.mem_offset};
        %endif
    % endif
    % if mem_tensor.is_intermediate or (mem_tensor.is_constant and mem_tensor.stored_in_external_memory):
        ${mem_tensor.name}_pt = match_mem+${mem_tensor.mem_offset};
        % if len(mem_tensor.load_from_ext_mem_at) > 0:
            ${mem_tensor.name}_ext_pt = match_ext_mem+ext_mem_offset;
        % endif
        % if mem_tensor.is_constant and mem_tensor.stored_in_external_memory:
            ${target.load_file_to_ext_mem_fn}("${model_name}_${mem_tensor.name}_data.hex", ${mem_tensor.name}_ext_pt, ${mem_tensor.elems * mem_tensor.dtype.itemsize});
        % endif
        % if len(mem_tensor.load_from_ext_mem_at) > 0:
            ext_mem_offset += ${mem_tensor.elems * mem_tensor.dtype.itemsize};
        % endif
    % elif mem_tensor.is_constant and not mem_tensor.stored_in_external_memory:
        ${mem_tensor.name}_pt = ${mem_tensor.name}_data_;
    % endif
% endfor

// Reset graph state
match_${model_name}_graph_execution_finished = 0;

// Reset device state
for (int i = 0; i < ${target.num_devices}; i++) {
    match_${model_name}_device_is_busy[i] = 0;
}

// Reset remaining parents
// TODO 

// Start DAG front nodes, ready nodes allocated in host start at the end.
match_${model_name}_schedule_next_node();

// Run TVM nodes when possible
while (!match_${model_name}_graph_execution_finished) 
{
    int running_tvm_node_id = next_tvm_node_id;
    switch (running_tvm_node_id) {
    % for node_id, node in enumerate(nodes):
        %if node.fallback:
            case ${node_id}:
                match_${model_name}_run_tvm_node_${node_id}();
                break;
        %endif
    % endfor
        case -1:
            // No TVM node to execute, wait MATCH nodes
            ${target.print_fn}("[${model_name} GRAPH] Host idle waiting for device EOC...\r\n");
            ${target.wait_eoc}();
            break;
        default:
            // Error
            ${target.print_fn}("[${model_name} GRAPH] Error: next_tvm_node_id %d is not a valid TVM node ID\r\n", next_tvm_node_id);
            return -1;
    }

    if (next_tvm_node_id >= 0) {
        // Reset next_tvm_node_id and free device (host) after execution
        next_tvm_node_id = -1;
        // Call EOC callback after TVM host node execution
        match_${model_name}_runtime_eoc_host_callback( running_tvm_node_id );
    }
}


% for mem_tensor in [m_t__ for m_t__ in mem_tensors if -1 in m_t__.move_temp_to_ext_mem]:
    ${target.load_to_ext_mem_fn}(${mem_tensor.name}_pt, ${mem_tensor.name}_ext_pt,${mem_tensor.elems * mem_tensor.dtype.itemsize});
% endfor

% if mem_needed_bytes>0 and target.free_fn != "":
    // Free MATCH memory pool
    ${target.free_fn}(match_mem);
% endif

% if ext_mem_needed_bytes>0:
    // Free external memory pool
    ${target.free_external_mem}(match_ext_mem, ${ext_mem_needed_bytes});
% endif

// Print stats
#if __${model_name}_GRAPH_PROFILE__
    ${target.print_fn}("[${model_name} GRAPH] Graph execution finished\r\n");
    % for node_id, node in enumerate(nodes):
        ${target.print_fn}("[${model_name} GRAPH] Node %d (${node.name}) start: %d - end: %d - interval: %f - device: %d\r\n", ${node_id},
            node_start_time[${node_id}],
            node_end_time[${node_id}],
            ((double)(int)(node_end_time[${node_id}] - node_start_time[${node_id}])) ${target.timestamp_to_ms},
            ${model_name}_node_device_id[${node_id}]
            );
    % endfor
#endif

return 0;
}