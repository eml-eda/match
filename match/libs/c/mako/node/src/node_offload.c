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
// include params file

% if exec_module.separate_build:

#include <nodes/${model_name}/${name}_params.h>
#ifdef __MATCH_TEST_NODE_WITH_HELPER__
#include <${target.name}/node_helper_nn.h>
#endif

<%namespace name="template_blocks" file="match_computation_block.c" />
<%namespace name="template_blocks" file="node_inner.c" />

% for block_idx,block in enumerate(schedule.blocks):
    % if block.backend == "MATCH":
        ${template_blocks.match_computation_block(block_idx,block)}
    % endif
% endfor

${template_blocks.node_inner()}

int main(int argc, char** argv){
    volatile uint32_t* args = ${mem_apis.shared_memory_extern_addr};
    ${node_fullname}_inner(args);
    return 0;
}

% else:

int main() {
    ;
}

% endif