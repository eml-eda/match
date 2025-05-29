/*
 * Mohamed Amine Hamdi <mohamed.hamdi@polito.it>
 *
 * Copyright (C) 2024 Politecnico Di Torino
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
*/

<%namespace name="template_blocks" file="match_computation_block.c" />
<%namespace name="template_blocks" file="node_inner.c" />   

% if not exec_module.separate_build:

    // include params file
    #include "nodes/${model_name}/${name}_params.h"
    #ifdef __MATCH_TEST_NODE_WITH_HELPER__
    #include <${target.name}/node_helper_nn.h>
    #endif

    % for block_idx,block in enumerate(schedule.blocks):
        % if block.backend == "MATCH":
            ${template_blocks.match_computation_block(block_idx,block)}
        % endif
    % endfor

    <%template_blocks:node_inner/>

    % if platform_apis.init_platform != "":
        int __attribute__ ((noinline)) ${node_fullname}(
            % for var in match_node.var_tensors.values():
            void* var_${var.name}_pt,
            % endfor
            % for idx,out in enumerate(match_node.output_tensors.values()):
            ${", " if idx>0 else ""}void* out_${out.name}_pt
            % endfor
        ){
            unsigned int* args[${len(match_node.var_tensors)+len(match_node.output_tensors)}];
            % for tensor_idx,tensor in enumerate({**match_node.var_tensors,**match_node.output_tensors}.values()):
            args[${tensor_idx}] = ${"var_" if tensor.tensor_type=="var" else "out_"}${tensor.name}_pt;
            % endfor	
            ${platform_apis.init_platform}(${name}_ctx, ${node_fullname}_inner, args);
            return 0;
        }
    % endif
    
% else:


## In this case we need to distinguish the two compilations: host and device
#ifdef __${exec_module.name}__
    ## Node exec_module code

    #include <nodes/${model_name}/${name}_params.h>
    #ifdef __MATCH_TEST_NODE_WITH_HELPER__
    #include <${target.name}/node_helper_nn.h>
    #endif

    % for block_idx,block in enumerate(schedule.blocks):
        % if block.backend == "MATCH":
            ${template_blocks.match_computation_block(block_idx,block)}
        % endif
    % endfor

    <%template_blocks:node_inner/>
    
#else
    ## Node host code

    #include "match/ctx.h"
    #include "match/utils.h"
    #include "${target.name}.h"
    % for inc_h in target.include_list:
        #include "${inc_h}.h"
    % endfor
    
    #include "nodes/${model_name}/${name}_params.h"

    int __attribute__ ((noinline)) ${node_fullname}(
        % for var in match_node.var_tensors.values():
            void* var_${var.name}_pt,
        % endfor
        % for idx,out in enumerate(match_node.output_tensors.values()):
            ${", " if idx>0 else ""}void* out_${out.name}_pt
        % endfor
    ){
        

        // Write args (input and output tensor addresses) in shared memory
        volatile uint32_t* args = (volatile uint32_t*)${exec_module.shared_memory_extern_addr};
        for (int i = 0; i < 16; i++) args[i] = 0;

        <% tensor_cnt = 1 %> 
        % for tensor in {**match_node.var_tensors,**match_node.output_tensors}.values():
            args[${tensor_cnt}] = (volatile uint32_t)${"var_" if tensor.tensor_type=="var" else "out_"}${tensor.name}_pt;\
            <% tensor_cnt += 1 %>
        % endfor	
        % for const_tensor in schedule.tensors.values():
            % if const_tensor.tensor_type=="const":
                args[${tensor_cnt}] = (volatile uint32_t)${name}_${const_tensor.name}_data;\
                <% tensor_cnt += 1 %>
            % endif
        % endfor

        // Set start signal - TODO send interrupt
        ${target.print_fn}("[HOST] Writing node_id (${node_idx} + 1) in %p. Now waiting...\r\n", args);

        % if target.timer_start_fn != "":
            ${target.timer_start_fn}();
        % endif

        ${exec_module.host_send_task_fn}(args, ${node_idx});

        // Wait completion - TODO this need to be reconsidered for parallel node execution
        int error = ${exec_module.host_wait_end_of_task_fn}(args, ${node_idx});
        if (error) {
            ${target.print_fn}("[HOST] Error in offloaded node execution.\r\n");
            return -1;
        }

        % if target.timer_stop_fn != "":
            ${name}_stats.total_cycles = ${target.timer_stop_fn}();
        % endif
        % if exec_module.timer_stop_fn != "":
            ${name}_stats.compute_cycles = args[9 + 0];
            ${name}_stats.load_cycles = args[9 + 1];
            ${name}_stats.store_cycles = args[9 + 2];
            ${name}_stats.load_bytes = args[9 + 3];
            ${name}_stats.store_bytes = args[9 + 4];
        % endif

        ${target.print_fn}("[HOST] Offload device finished.\r\n");

        ${target.print_fn}("[HOST] Stats:\r\n");
        ${target.print_fn}("       Total Cycles: %d\r\n", ${name}_stats.total_cycles);
        ${target.print_fn}("       Compute Cycles: %d\r\n", ${name}_stats.compute_cycles);
        ${target.print_fn}("       Load Cycles: %d\r\n", ${name}_stats.load_cycles);
        ${target.print_fn}("       Store Cycles: %d\r\n", ${name}_stats.store_cycles);
        ${target.print_fn}("       Load Bytes: %d\r\n", ${name}_stats.load_bytes);
        ${target.print_fn}("       Store Bytes: %d\r\n", ${name}_stats.store_bytes);

        return 0;
    }
#endif


% endif