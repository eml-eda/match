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
#include <nodes/${model_name}/${name}_params.h>
#ifdef __MATCH_TEST_NODE_WITH_HELPER__
#include <${target.name}/node_helper_nn.h>
#endif

% for block_idx,block in enumerate(schedule.blocks):
% if block.backend == "MATCH":
<% brackets_cnt = 0 %>

void match_backend_block_${block_idx}_computation(MatchCtx* ctx){
    // get task ID and num tasks, may be useful for future multithreading approaches...
    ctx->task_id = ${platform_apis.get_task_id}(ctx);
    ctx->num_tasks = ${block.num_tasks};
    % for instr in block.init_instrs:
    ${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
    % endfor
    % for loop_idx,lp in enumerate(block.loops[block.loop_idx_end_sw_controlled_loads:]):
    % if not exec_module.backend_constraints_check(match_node,schedule,block,lp,loop_idx):
    <% continue %>
    % else:
    <% brackets_cnt += 1 %>
    % endif
    ${c_ident(brackets_cnt-1)}for(block_${block_idx}_loop_${lp.name}_set();
        ${c_ident(brackets_cnt-1)}block_${block_idx}_loop_${lp.name}_end();
        ${c_ident(brackets_cnt-1)}block_${block_idx}_loop_${lp.name}_update()){
        % for instr in lp.init_instrs:
        ${c_ident(brackets_cnt-1)}${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
        % endfor
    % endfor
    ## close braces and save output
    % for loop_idx_ in range(loop_idx,-1,-1):
        % for instr in schedule.blocks[block_idx].loops[loop_idx_].instrs:
        ${c_ident(brackets_cnt-1)}${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
        % endfor
    ${c_ident(brackets_cnt-1)}}
    <% brackets_cnt -= 1 %>
    % if brackets_cnt<=0:
    <% break %>
    % endif
    % endfor
    % for instr in block.instrs:
    ${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
    % endfor

}
% endif
% endfor

${"void" if platform_apis.init_platform!="" else "int"} __attribute__ ((noinline)) ${node_fullname}${"_inner" if platform_apis.init_platform!="" else ""}(
    % if platform_apis.init_platform!="":
    void* args
    % else:
    % for var in match_node.var_tensors.values():
    void* var_${var.name}_pt,
    % endfor
    % for idx,out in enumerate(match_node.output_tensors.values()):
    ${", " if idx>0 else ""}void* out_${out.name}_pt
    % endfor
    % endif
)
{
    unsigned int *real_args = (unsigned int *) args;
    % for tensor_idx,tensor in enumerate({**match_node.var_tensors,**match_node.output_tensors}.values()):
    void* ${"var_" if tensor.tensor_type=="var" else "out_"}${tensor.name}_pt = (void*) real_args[${tensor_idx}];
    % endfor
    MatchCtx* ctx = ${name}_ctx;

    % if platform_apis.init_module!="":
    ${platform_apis.init_module}(ctx);
    % endif

    % for var in match_node.var_tensors.values():
    ${var.name}->base_pt = var_${var.name}_pt;
    ${var.name}->pts[${memory_hierarchy["var"][-1].name}] = var_${var.name}_pt;
    % endfor
    % for out in match_node.output_tensors.values():
    ${out.name}->base_pt = out_${out.name}_pt;
    ${out.name}->pts[${memory_hierarchy["out"][-1].name}] = out_${out.name}_pt;
    % endfor
    
    % for const_tensor in match_node.const_tensors.values():
    ${const_tensor.name}->base_pt = ${const_tensor.name}_data;
    ${const_tensor.name}->pts[${memory_hierarchy["const"][-1].name}] = ${const_tensor.name}_data;
    % endfor

    #ifndef __MATCH_TEST_NODE_WITH_HELPER__
    % for intermediate_tensor in schedule.tensors.values():
    % if intermediate_tensor.tensor_type=="intermediate":
    ${intermediate_tensor.name}->base_pt = ${target.alloc_fn}(${intermediate_tensor.prod_shape}*sizeof(${c_dtype(intermediate_tensor.dtype)}));
    ${intermediate_tensor.name}->pts[${memory_hierarchy["inter"][-1].name}] = ${intermediate_tensor.name}->base_pt;
    % endif
    % endfor

    % for mem_level in set([mem_ for k,v in memory_hierarchy.items() for mem_ in v]):
    % if mem_level.sw_controlled and mem_level.name!=exec_module.top_memory:
    void* ${mem_level.name}_base_pt = ${mem_apis.init_memory[mem_level.name]}(ctx);
    int ${mem_level.name}_curr_pt_offset = 0;
    % endif
    % endfor
    % for instr in schedule.init_instrs:
    ${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
    % endfor
    % for block_idx,block in enumerate(schedule.blocks):
    // block ${block_idx}
    % if block.num_buffers_for_computation!=1:
    // block ${block_idx} does bufferized computation
    int block_${block_idx}_buffer_for_computation_idx = 0;
    int BLOCK_${block_idx}_NUM_BUFFERS_FOR_COMPUTATION = ${block.num_buffers_for_computation};
    % endif
    % for loop_idx,lp in enumerate(block.loops):
    % for mem_transfer in lp.mem_transfers:
    ${c_ident(loop_idx)}// compute the offset from the top level memory to obtain the correct tile for the transfer
    % if tensor.is_fused and tensor.unsupported_layout:
    ${c_ident(loop_idx)}int ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size = ${mem_apis.get_size_of_fused_tensor}(ctx,${mem_transfer.tensor.name});
    ${c_ident(loop_idx)}void* ${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt = ${mem_apis.get_pt_of_fused_tensor}(ctx,${mem_transfer.tensor.name});
    % else:
    ${c_ident(loop_idx)}int ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size = ${mem_transfer.tensor.c_offset_expr_size_sw_mem(mem_transfer.mem)};
    ${c_ident(loop_idx)}void* ${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt = ${mem_transfer.tensor.name}->pts[${mem_transfer.top_mem}]+${mem_transfer.tensor.c_offset_expr_sw_mem(mem_transfer.mem)};
    % endif
    ${c_ident(loop_idx)}${mem_transfer.tensor.name}->pts[${mem_transfer.mem}] = ${mem_transfer.mem}_base_pt + ${mem_transfer.mem}_curr_pt_offset;
    ${c_ident(loop_idx)}${mem_transfer.mem}_curr_pt_offset += ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size;
    % if mem_transfer.tensor.tensor_type != "output":
    ${c_ident(loop_idx)}// call API for ${exec_module.name}-specific memory transfer handling
    ${c_ident(loop_idx)}${mem_apis.mem_transfer}(
        ${c_ident(loop_idx)}ctx,${mem_transfer.tensor.name},${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt,
        ${c_ident(loop_idx)}${mem_transfer.tensor.name}->pts[${mem_transfer.mem}],
        ${c_ident(loop_idx)}MATCH_SW_LOAD_TENSOR,MATCH_${"CONST" if mem_transfer.tensor.tensor_type=="const" else "VAR"}_TENSOR,
        ${c_ident(loop_idx)}${mem_transfer.top_mem},${mem_transfer.mem}
    ${c_ident(loop_idx)});
    % if sync_apis.must_sync_after_load:
    ${c_ident(loop_idx)}// sync after each single load as the SW transfer require...
    ${c_ident(loop_idx)}${sync_apis.wait_load}(ctx);
    % endif
    % endif
    % endfor
    ## finished sw controlled loads and stores
    % if exec_module.backend_constraints_check(match_node,schedule,block,lp,loop_idx) and block.loop_idx_end_sw_controlled_loads>=loop_idx:
    <% break %>
    % endif 
    ${c_ident(loop_idx)}for(block_${block_idx}_loop_${lp.name}_set();
        ${c_ident(loop_idx)}block_${block_idx}_loop_${lp.name}_end();
        ${c_ident(loop_idx)}block_${block_idx}_loop_${lp.name}_update()){
    % endfor
    
    % if not sync_apis.must_sync_after_load and sync_apis.wait_load!="":
    ${c_ident(loop_idx)}// sync with the SW controlled transfers
    ${c_ident(loop_idx)}${sync_apis.wait_load}(ctx);
    % endif

    % if block.backend == "MATCH":
    % if block.parallel_execution:
    ${c_ident(loop_idx)}${platform_apis.parallelize_task}(match_backend_block_${block_idx}_computation,${block.num_tasks},ctx);
    % else:
    ${c_ident(loop_idx)}match_backend_block_${block_idx}_computation(ctx);
    % endif
    % else:
    ${c_ident(loop_idx)}${comp_apis.compute_tile}(ctx);
    % endif

    % if sync_apis.must_sync_after_computation:
    % if block.backend=="MATCH" and block.parallel_execution and sync_apis.wait_parallel_tasks:
    ${c_ident(loop_idx)}${sync_apis.wait_parallel_tasks}(ctx);
    % elif sync_apis.wait_tile_computation!="":
    ${c_ident(loop_idx)}${sync_apis.wait_tile_computation}(ctx);
    % endif
    % elif block.num_buffers_for_computation!=1:
    ${c_ident(loop_idx)}buffer_for_computation_idx++;
    % if sync_apis.wait_tile_computation!="":
    ${c_ident(loop_idx)}// the buffer is full, wait before the next iteration...
    ${c_ident(loop_idx)}if(block_${block_idx}_buffer_for_computation_idx>=BLOCK_${block_idx}_NUM_BUFFERS_FOR_COMPUTATION){
        % if block.backend=="MATCH" and block.parallel_execution and sync_apis.wait_buffer_parallel_tasks!="":
        ${c_ident(loop_idx)}${sync_apis.wait_buffer_parallel_tasks}(ctx);
        % elif sync_apis.wait_buffer_tile_computation!="":
        ${c_ident(loop_idx)}${sync_apis.wait_buffer_tile_computation}(ctx);
        % endif
    ${c_ident(loop_idx)}}
    % endif
    % endif
        
    ## close braces and save output
    % for loop_idx_ in range(loop_idx,-1,-1):
    % if not exec_module.backend_constraints_check(match_node,schedule,block,block.loops[loop_idx_],loop_idx_) and block.loop_idx_end_sw_controlled_loads>=loop_idx_:
    ${c_ident(loop_idx_)}}
    % endif
    % for mem_transfer in block.loops[loop_idx_].mem_transfers:
    % if mem_transfer.tensor.tensor_type == "output":
    ${c_ident(loop_idx_)}// call API for ${exec_module.name}-specific memory transfer handling
    ${c_ident(loop_idx_)}${mem_apis.mem_transfer}(
        ${c_ident(loop_idx_)}ctx,${mem_transfer.tensor.name},${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt,
        ${c_ident(loop_idx_)}${mem_transfer.tensor.name}->pts[${mem_transfer.mem}],
        ${c_ident(loop_idx_)}MATCH_SW_STORE_TENSOR,MATCH_OUT_TENSOR,
        ${c_ident(loop_idx_)}${mem_transfer.top_mem},${mem_transfer.mem}
    ${c_ident(loop_idx_)});
    % if sync_apis.must_sync_after_store:
    ${c_ident(loop_idx_)}// sync after each single store as the SW transfer require...
    ${c_ident(loop_idx_)}${sync_apis.wait_store}(ctx);
    % endif
    % endif
    % if block.num_buffers_for_computation==1 or mem_transfer.mem!=memory_hierarchy[mem_transfer.tensor.tensor_type][0].name:
    ${c_ident(loop_idx_)}${mem_transfer.mem}_curr_pt_offset -= ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size;
    % elif block.num_buffers_for_computation>1 and mem_transfer.mem==memory_hierarchy[mem_transfer.tensor.tensor_type][0].name:
    ${c_ident(loop_idx_)}if(block_${block_idx}_buffer_for_computation_idx>=BLOCK_${block_idx}_NUM_BUFFERS_FOR_COMPUTATION)
        ${c_ident(loop_idx_)}${mem_transfer.mem}_curr_pt_offset -= ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size;
    % endif
    % endfor
    % endfor
    % endfor
    
    % for instr in schedule.instrs:
    ${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
    % endfor
    % for mem_level in set([mem_ for k,v in memory_hierarchy.items() for mem_ in v]):
    % if mem_level.sw_controlled and mem_level.name!=exec_module.top_memory:
    ${mem_apis.free_memory[mem_level.name]}(ctx,${mem_level.name}_base_pt);
    % endif
    % endfor
    % for intermediate_tensor in schedule.tensors.values():
    % if intermediate_tensor.tensor_type=="intermediate":
    ${target.free_fn}(${intermediate_tensor.name}->base_pt);
    % endif
    % endfor
    % if platform_apis.free_module!="":
    ${platform_apis.free_module}(ctx);
    % endif

    #endif
    #ifdef __MATCH_TEST_NODE_WITH_HELPER__
    run_node_schedule_nn(ctx);
    #endif
    % if platform_apis.init_platform=="":
    return 0;
    % endif
}

% if platform_apis.init_platform!="":
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
    ${platform_apis.init_platform}(${node_fullname}_inner,args);
    return 0;
}
% endif