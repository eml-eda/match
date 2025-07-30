#include <${model_name}_graph.h>

% for include in target.include_list:
    #include <${include}.h>
% endfor


// DLTensor declarations
% for tensor in tensors:
    % if tensor.used_by_tvm:
        static DLTensor ${tensor.name}_dltensor;
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


// Declare L2 tensors pointers
% for tensor in tensors:
    % for i in range(len(tensor.soc_mem_offsets)):
        static ${"const"*tensor.is_constant} void* ${tensor.name}_pt_${i};
    % endfor
    % if tensor.static_in_soc_mem:
        static ${"const"*tensor.is_constant} void* ${tensor.name}_pt_0;
    % endif
% endfor

// Declare L3 tensors pointers
% for tensor in tensors:
    % for i in range(len(tensor.ext_mem_offsets)):
        static void* ${tensor.name}_ext_pt_${i};
    % endfor
    % if tensor.static_in_ext_mem:
        static void* ${tensor.name}_ext_pt_0;
    % endif
% endfor

// Profiling things

static volatile uint64_t node_start_time[${len(nodes)}];
static volatile uint64_t node_end_time[${len(nodes)}];

// Additional node info for async exec

% for node in nodes:
    static int ${node.name}_children[] = {${", ".join(str(child) for child in node.children_nids)}};
% endfor
static int* ${model_name}_node_children[] = {${", ".join(f"{node.name}_children" for node in nodes)}};

static int ${model_name}_num_node_children[] = {${", ".join(str(len(node.children_nids)) for node in nodes)}};
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
        % for inp_idx,inp_tensor in enumerate(node.inputs):
            ${node.name}_args_[${inp_idx}].v_handle = (void*)(&${inp_tensor.name}_dltensor);
            ${inp_tensor.name}_dltensor.data = ${inp_tensor.name}_pt_${node.tensor_soc_segments_ids[inp_tensor.id]};
        % endfor
        % for out_idx,out_tensor in enumerate(node.outputs):
            ${node.name}_args_[${len(node.inputs)+out_idx}].v_handle = (void*)(&${out_tensor.name}_dltensor);
            ${out_tensor.name}_dltensor.data = ${out_tensor.name}_pt_${node.tensor_soc_segments_ids[out_tensor.id]};
        % endfor

        #if __${model_name}_FALLBACK_GRAPH_PROFILE__
        node_start_time[${node_id}] = ${target.start_get_timestamp_api}();
        #endif

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
            ${target.print_fn}("[${model_name} GRAPH] TVM node ${node.name} done, output differs from checksum by %d\r\n", 
            match_byte_checksum_check(${node.outputs[0].name}_pt_${node.tensor_soc_segments_ids[node.outputs[0].id]}, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
        #endif

    % else:
        ## MATCH NODE - async, just start

        % for inp_tensor in [inp__ for inp__ in node.inputs if inp__.is_constant]:
            ${inp_tensor.name}_data = ${inp_tensor.name}_pt_${node.tensor_soc_segments_ids[inp_tensor.id]};
        % endfor

        #if __${model_name}_GRAPH_PROFILE__
        node_start_time[${node_id}] = ${target.start_get_timestamp_api}();
        #endif

        ${node.fn_name}_async(
            % for inp_idx,inp_tensor in enumerate([inp__ for inp__ in node.inputs if not inp__.is_constant]):
            ${"" if inp_idx==0 else ","}${inp_tensor.name}_pt_${node.tensor_soc_segments_ids[inp_tensor.id]}
            % endfor
            % for out_tensor in node.outputs:
            , ${out_tensor.name}_pt_${node.tensor_soc_segments_ids[out_tensor.id]}
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
            ${target.print_fn}("[${model_name} GRAPH] MATCH node ${node.name} done, output differs from checksum by %d\r\n", 
            match_byte_checksum_check(${node.outputs[0].name}_pt_${node.tensor_soc_segments_ids[node.outputs[0].id]}, __${model_name}_GRAPH_${node.name}_BYTES__, __${model_name}_GRAPH_${node.name}_CHECKSUM__));
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

const int match_${model_name}_num_parents[] = {${", ".join(str(node.num_parents) for node in nodes)}};
int match_${model_name}_num_remaining_parents[${len(nodes)}];

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
    ${rt_i.c_type}* ${rt_i.name}_${"ext_" if rt_i.static_in_ext_mem else ""}pt,
% endfor
% for rt_o_idx,rt_o in enumerate(rt_outputs):
    ${"" if rt_o_idx==0 else ", "}${rt_o.c_type}* ${rt_o.name}_${"ext_" if rt_o.static_in_ext_mem else ""}pt
% endfor
){

% if ext_mem_pool_size > 0:
    // L3 memory pool allocation
    % if target.allocate_ext_mem != "":
        void* match_ext_mem_pool = ${target.allocate_ext_mem}(${ext_mem_pool_size});
    % elif target.ext_mem_linker_section != "":
        static uint8_t match_ext_mem_pool_[${ext_mem_pool_size}] __attribute__((section("${target.ext_mem_linker_section}")));
        volatile void* match_ext_mem_pool = (void*) match_ext_mem_pool_;
    % endif
% endif

% if soc_mem_pool_size > 0:
    // L2 memory pool allocation
    % if target.alloc_fn != "":
        void* match_soc_mem_pool = ${target.alloc_fn}(${soc_mem_pool_size});
    % else:
        static uint8_t match_soc_mem_pool_[${soc_mem_pool_size}];
        volatile void* match_soc_mem_pool = (void*) match_soc_mem_pool_;
    % endif
% endif

// Inizialize other tensor pointers in MATCH memory as planned

// Tensors L2 Instances Addresses
% for tensor in tensors:
    % if tensor.is_input or tensor.is_output:
        ${tensor.name}_pt_0 = ${tensor.name}_pt; // ${"Input" if tensor.is_input else "Output"}
    % else:
        % for i in range(len(tensor.soc_mem_offsets)):
            ${tensor.name}_pt_${i} = match_soc_mem_pool+${tensor.soc_mem_offsets[i]}; // ${"Const"*tensor.is_constant} ${"Intermediate"*tensor.is_intermediate}
        % endfor
        % if tensor.static_in_soc_mem:
            ${tensor.name}_pt_0 = ${tensor.name}_data_; // ${"Const"*tensor.is_constant} Static
        % endif
    % endif
% endfor

// Tensors L3 Instances Addresses
<% l3_tensor_count = 0 %>
% for tensor in tensors:
    % for i in range(len(tensor.ext_mem_offsets)):
        ${tensor.name}_ext_pt_${i} = match_ext_mem_pool+${tensor.ext_mem_offsets[i]};
        <% l3_tensor_count += 1 %>
    % endfor
    % if tensor.static_in_ext_mem:
        ${tensor.name}_ext_pt_0 = ${tensor.name}_data_;
        <% l3_tensor_count += 1 %>
    % endif
% endfor
% if not l3_tensor_count:
    // None
% endif

// Reset graph state
match_${model_name}_graph_execution_finished = 0;

// Reset device state
for (int i = 0; i < ${target.num_devices}; i++) {
    match_${model_name}_device_is_busy[i] = 0;
}

// Reset remaining parents
for (int i = 0; i < match_${model_name}_num_nodes; i++) {
    match_${model_name}_num_remaining_parents[i] = match_${model_name}_num_parents[i];
}

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

% if soc_mem_pool_size>0 and target.free_fn != "":
    // Free MATCH memory pool
    ${target.free_fn}(match_soc_mem_pool);
% endif

% if ext_mem_pool_size>0:
    // Free external memory pool
    ${target.free_external_mem}(match_ext_mem_pool, ${ext_mem_pool_size});
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