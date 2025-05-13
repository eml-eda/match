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

% if not exec_module.separate_build:

    <%namespace name="template_blocks" file="match_computation_block.c" />
    <%namespace name="template_blocks" file="node_inner.c" />

    // include params file
    #include <nodes/${model_name}/${name}_params.h>
    #ifdef __MATCH_TEST_NODE_WITH_HELPER__
    #include <${target.name}/node_helper_nn.h>
    #endif

    % for block_idx,block in enumerate(schedule.blocks):
        % if block.backend == "MATCH":
            ${template_blocks.match_computation_block(block_idx,block)}
        % endif
    % endfor

    ${template_blocks.node_inner()}

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

    #include <match/ctx.h>
    #include <match/utils.h>
    #include <${target.name}.h>
    % for inc_h in target.include_list:
        #include <${inc_h}.h>
    % endfor

    #include "nodes/${model_name}/${name}_payload.h"
    #include <nodes/${model_name}/${name}_params.h>


    int __attribute__ ((noinline)) ${node_fullname}(
        % for var in match_node.var_tensors.values():
            void* var_${var.name}_pt,
        % endfor
        % for idx,out in enumerate(match_node.output_tensors.values()):
            ${", " if idx>0 else ""}void* out_${out.name}_pt
        % endfor
    ){
        // Write args (input and output tensor addresses) in shared memory
        volatile uint32_t* args = (volatile uint32_t*)${node_fullname}_args_addr; \
        <% tensor_cnt = 0 %>
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

        // DMA the binary
        for (int i = 0; ${node_fullname}_binary_sections[i].size != 0; i++) {
            ${target.offload_dma_fn}(
                ${node_fullname}_binary_sections[i].src,
                ${node_fullname}_binary_sections[i].dst,
                ${node_fullname}_binary_sections[i].size
            );
        }

        // Start device
        if (${node_fullname}_boot_addr != NULL) {
            ${platform_apis.init_platform}(${node_fullname}_boot_addr);
        }
        
        return 0;
    }

% endif