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
    ${c_ident(brackets_cnt-1)}for(${name}_block_${block_idx}_loop_${lp.name}_set();
        ${c_ident(brackets_cnt-1)}${name}_block_${block_idx}_loop_${lp.name}_end();
        ${c_ident(brackets_cnt-1)}${name}_block_${block_idx}_loop_${lp.name}_update()){
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
    % if platform_apis.init_platform!="":
    unsigned int *real_args = (unsigned int *) args;
    % for tensor_idx,tensor in enumerate({**match_node.var_tensors,**match_node.output_tensors}.values()):
    void* ${"var_" if tensor.tensor_type=="var" else "out_"}${tensor.name}_pt = (void*) real_args[${tensor_idx}];
    % endfor
    % endif
    MatchCtx* ctx = ${name}_ctx;

    % if platform_apis.init_module!="":
    ${platform_apis.init_module}(ctx);
    % endif

    % for var in match_node.var_tensors.values():
    ${name}_${var.name}->base_pt = var_${var.name}_pt;
    ${name}_${var.name}->pt = var_${var.name}_pt;
    ${name}_${var.name}->pts[${memory_hierarchy["var"][-1].name}] = var_${var.name}_pt;
    % endfor
    % for out in match_node.output_tensors.values():
    ${name}_${out.name}->base_pt = out_${out.name}_pt;
    ${name}_${out.name}->pt = out_${out.name}_pt;
    ${name}_${out.name}->pts[${memory_hierarchy["output"][-1].name}] = out_${out.name}_pt;
    % endfor
    
    % for const_tensor in schedule.tensors.values():
    % if const_tensor.tensor_type=="const":
    ${name}_${const_tensor.name}->base_pt = ${name}_${const_tensor.name}_data;
    ${name}_${const_tensor.name}->pt = ${name}_${const_tensor.name}_data;
    ${name}_${const_tensor.name}->pts[${memory_hierarchy["const"][-1].name}] = ${name}_${const_tensor.name}_data;
    % endif
    % endfor

    #ifndef __MATCH_TEST_NODE_WITH_HELPER__
    % for intermediate_tensor in schedule.tensors.values():
    % if intermediate_tensor.tensor_type=="intermediate":
    ${name}_${intermediate_tensor.name}->base_pt = ${target.alloc_fn}(${intermediate_tensor.prod_shape}*sizeof(${c_dtype(intermediate_tensor.dtype)}));
    ${name}_${intermediate_tensor.name}->pt = ${name}_${intermediate_tensor.name}->base_pt;
    ${name}_${intermediate_tensor.name}->pts[${memory_hierarchy["intermediate"][-1].name}] = ${intermediate_tensor.name}->base_pt;
    % endif
    % endfor

    % for mem_level in set([mem_ for k,v in memory_hierarchy.items() for mem_ in v]):
    % if mem_level.sw_controlled and mem_level.name!=target.host_memory:
    void* ${mem_level.name}_base_pt = ${mem_apis.init_memory[mem_level.name]}(ctx);
    int ${mem_level.name}_curr_pt_offset = 0;
    ## alloc buffers on sw controlled memory
    % for buffer_idx,buffer in enumerate(schedule.buffers):
    % if mem_apis.alloc_buffer!="" and buffer.mem_name==mem_level.name:
    ${mem_apis.alloc_buffer}(
        "${buffer.name}",
        ${buffer.mem_name}_base_pt + ${buffer.mem_name}_curr_pt_offset,
        ${buffer.num_bytes}, ${buffer.mem_name}, ${buffer_idx}
    );
    ${buffer.mem_name}_curr_pt_offset += ${buffer.num_bytes};
    % endif
    % endfor
    % endif
    % endfor
    int tile_mem_offset = 0;
    % for instr in schedule.init_instrs:
    ${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
    % endfor
    % for block_idx,block in enumerate(schedule.blocks):
    // block ${block_idx}
    % if not sync_apis.must_sync_after_load:
    int block_${block_idx}_loads = 0;
    % endif
    % if block.num_buffers_for_computation!=1:
    // block ${block_idx} does bufferized computation
    int block_${block_idx}_buffer_for_computation_idx = 0;
    int BLOCK_${block_idx}_NUM_BUFFERS_FOR_COMPUTATION = ${block.num_buffers_for_computation};
    % endif
    % for loop_idx,lp in enumerate(block.loops):
    % for mem_transfer in lp.mem_transfers:
    ${c_ident(loop_idx)}// compute the offset from the top level memory to obtain the correct tile for the transfer
    % for t_dim_idx, t_dim in enumerate(mem_transfer.tensor.dims):
    % if t_dim in match_node.dependent_dims and len(set([schedule.tensor_tiles[mem_transfer.tensor.name][idx].tiled_dims[t_dim_idx].size for idx in range(len(schedule.tensor_tiles[mem_transfer.tensor.name]))]))!=1:
    ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].size = ${name}_${t_dim.name}->curr_size; // this dim is not independent
    ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].max_size = ${name}_${t_dim.name}->curr_max_size; // this dim is not independent
    ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].start_idx = ${name}_${t_dim.name}->global_idx;
    ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].curr_idx = ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].start_idx;
    % endif
    % if any([lp.dim==t_dim for lp in block.loops[last_transfer_of_tensor_block[(mem_transfer.tensor.name, block_idx)][0]:loop_idx]]):
    ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].start_idx = ${name}_${t_dim.name}->global_idx;
    ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].curr_idx = ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].start_idx;
    % endif
    % endfor
    % if (mem_transfer.tensor.is_fused or mem_transfer.tensor.unsupported_layout) and mem_apis.get_size_of_fused_tensor!="" and mem_apis.get_pt_of_fused_tensor!="":
    ${c_ident(loop_idx)}int ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)} = ${mem_apis.get_size_of_fused_tensor}(ctx,${name}_${mem_transfer.tensor.name});
    ${c_ident(loop_idx)}tile_mem_offset = ${mem_apis.get_pt_of_fused_tensor}(ctx,${name}_${mem_transfer.tensor.name});
    ${c_ident(loop_idx)}void* ${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt${c_unique_num_tile(mem_transfer.tensor.name)} = ${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.top_mem}] + (tile_mem_offset>0?tile_mem_offset:0);
    % else:
    ${c_ident(loop_idx)}int ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)} = ${mem_transfer.tensor.c_offset_expr_size_sw_mem(mem_transfer.mem, name)};
    ${c_ident(loop_idx)}tile_mem_offset = ${mem_transfer.tensor.c_offset_expr_sw_mem(mem_transfer.top_mem, schedule, block_idx, loop_idx, name)};
    ${c_ident(loop_idx)}void* ${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt${c_unique_num_tile(mem_transfer.tensor.name)} = ${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.top_mem}] + (tile_mem_offset>0?tile_mem_offset:0);
    % endif
    ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.mem}] = ${mem_transfer.mem}_base_pt + ${mem_transfer.mem}_curr_pt_offset;
    ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}->pt = ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.mem}];
    ${c_ident(loop_idx)}${mem_transfer.mem}_curr_pt_offset += ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)};
    % if mem_transfer.tensor.tensor_type != "output":
    ${c_ident(loop_idx)}// call API for ${exec_module.name}-specific memory transfer handling
    ${c_ident(loop_idx)}${mem_apis.mem_transfer}(
        ${c_ident(loop_idx)}ctx,${name}_${mem_transfer.tensor.name},${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt${c_unique_num_tile(mem_transfer.tensor.name)},
        ${c_ident(loop_idx)}${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.mem}],
        ${c_ident(loop_idx)}MATCH_SW_LOAD_TENSOR,MATCH_${"CONST" if mem_transfer.tensor.tensor_type=="const" else "VAR"}_TENSOR,
        ${c_ident(loop_idx)}${mem_transfer.top_mem},${mem_transfer.mem}
    ${c_ident(loop_idx)});
    % if sync_apis.must_sync_after_load:
    ${c_ident(loop_idx)}// sync after each single load as the SW transfer require...
    ${c_ident(loop_idx)}${sync_apis.wait_load}(ctx);
    % else:
    ${c_ident(loop_idx)}block_${block_idx}_loads++;
    % endif
    % endif
    <% add_tile_to_tensor_at_block_and_loop(mem_transfer.tensor.name, block_idx, loop_idx, mem_transfer.mem)%>
    % endfor
    ## finished sw controlled loads and stores
    % if exec_module.backend_constraints_check(match_node,schedule,block,lp,loop_idx) and block.loop_idx_end_sw_controlled_loads>=loop_idx:
    <% break %>
    % endif 
    ${c_ident(loop_idx)}for(${name}_block_${block_idx}_loop_${lp.name}_set();
        ${c_ident(loop_idx)}${name}_block_${block_idx}_loop_${lp.name}_end();
        ${c_ident(loop_idx)}${name}_block_${block_idx}_loop_${lp.name}_update()){
    % endfor
    
    % if not sync_apis.must_sync_after_load and sync_apis.wait_load!="":
    ${c_ident(loop_idx)}// sync with the SW controlled transfers
    ${c_ident(loop_idx)}if(block_${block_idx}_loads)    ${sync_apis.wait_load}(ctx);
    ${c_ident(loop_idx)}block_${block_idx}_loads = 0;
    % endif
    
    ## fix start idxs and curr pts of other tensors not involved in mem transfers
    % for tensor in [tens for tens in schedule.tensors.values() if last_transfer_of_tensor_block[(tens.name, block_idx)][0]!=loop_idx]:
    <% tensor_need_update = False %>
    % for t_dim_idx, t_dim in enumerate(tensor.dims):
    % if t_dim in match_node.dependent_dims and [lp.dim in t_dim.dim_dependency.dependencies for lp in block.loops[last_transfer_of_tensor_block[(tensor.name, block_idx)][0]:loop_idx]] or any([lp.dim==t_dim for lp in block.loops[last_transfer_of_tensor_block[(tensor.name, block_idx)][0]:loop_idx]]):
    <% tensor_need_update = True %>
    <% break %>
    % endif
    % endfor
    % if tensor_need_update:
    % if (tensor.is_fused or tensor.unsupported_layout) and mem_apis.get_pt_of_fused_tensor!="":
    ${c_ident(loop_idx)}tile_mem_offset = ${mem_apis.get_pt_of_fused_tensor}(ctx,${name}_${tensor.name});
    % else:
    ${c_ident(loop_idx)}tile_mem_offset = ${tensor.c_offset_expr_sw_mem(last_transfer_of_tensor_block[(tensor.name, block_idx)][1], schedule, block_idx, loop_idx, name)};
    % endif
    ${c_ident(loop_idx)}${name}_${tensor.name}->pt = ${name}_${tensor.name}->pts[${last_transfer_of_tensor_block[(tensor.name, block_idx)][1]}] + (tile_mem_offset>0?tile_mem_offset:0);
    % for t_dim_idx, t_dim in enumerate(tensor.dims):
    % if t_dim in match_node.dependent_dims and [lp.dim in t_dim.dim_dependency.dependencies for lp in block.loops[last_transfer_of_tensor_block[(tensor.name, block_idx)][0]:loop_idx]] or any([lp.dim==t_dim for lp in block.loops[last_transfer_of_tensor_block[(tensor.name, block_idx)][0]:loop_idx]]):
    ${c_ident(loop_idx)}${name}_${tensor.name}_tiles_[${last_transfer_of_tensor_block[(tensor.name, block_idx)][1]}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].curr_idx = ${name}_${t_dim.name}->global_idx;
    % endif
    % endfor
    % endif
    % endfor

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
    <% free_transfer_unique_tile(mem_transfer.tensor.name) %>
    % if mem_transfer.tensor.tensor_type == "output":
    ${c_ident(loop_idx_)}// call API for ${exec_module.name}-specific memory transfer handling
    ${c_ident(loop_idx_)}${mem_apis.mem_transfer}(
        ${c_ident(loop_idx_)}ctx,${name}_${mem_transfer.tensor.name},${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt${c_unique_num_tile(mem_transfer.tensor.name)},
        ${c_ident(loop_idx_)}${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.mem}],
        ${c_ident(loop_idx_)}MATCH_SW_STORE_TENSOR,MATCH_OUT_TENSOR,
        ${c_ident(loop_idx_)}${mem_transfer.top_mem},${mem_transfer.mem}
    ${c_ident(loop_idx_)});
    % if sync_apis.must_sync_after_store:
    ${c_ident(loop_idx_)}// sync after each single store as the SW transfer require...
    ${c_ident(loop_idx_)}${sync_apis.wait_store}(ctx);
    % endif
    % endif
    % if block.num_buffers_for_computation==1 or mem_transfer.mem!=memory_hierarchy[mem_transfer.tensor.tensor_type][0].name:
    ${c_ident(loop_idx_)}${mem_transfer.mem}_curr_pt_offset -= ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)};
    % elif block.num_buffers_for_computation>1 and mem_transfer.mem==memory_hierarchy[mem_transfer.tensor.tensor_type][0].name:
    ${c_ident(loop_idx_)}if(block_${block_idx}_buffer_for_computation_idx>=BLOCK_${block_idx}_NUM_BUFFERS_FOR_COMPUTATION)
        ${c_ident(loop_idx_)}${mem_transfer.mem}_curr_pt_offset -= ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)};
    % endif
    % endfor
    % endfor
    % endfor
    
    % for instr in schedule.instrs:
    ${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
    % endfor
    % for mem_level in set([mem_ for k,v in memory_hierarchy.items() for mem_ in v]):
    % if mem_level.sw_controlled and mem_level.name!=target.host_memory and mem_level.name in mem_apis.free_memory:
    ${mem_apis.free_memory[mem_level.name]}(ctx,${mem_level.name}_base_pt);
    % endif
    % endfor
    % for intermediate_tensor in schedule.tensors.values():
    % if intermediate_tensor.tensor_type=="intermediate":
    ${target.free_fn}(${name}_${intermediate_tensor.name}->base_pt);
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
    ${platform_apis.init_platform}(${name}_ctx, ${node_fullname}_inner, args);
    return 0;
}
% endif