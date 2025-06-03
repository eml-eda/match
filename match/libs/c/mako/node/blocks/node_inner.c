<%! smp_region_counter = 0 %> 
 
<%def name="smp_configured_core_guard()">
    % if platform_apis.smp_configured_core_guard != "":
        if (${platform_apis.smp_configured_core_guard}() == 0) return;
    % endif
</%def>

<%def name="smp_barrier()">
    % if exec_module.is_smp and sync_apis.smp_barrier != "":
        ${sync_apis.smp_barrier}(ctx);
    % endif
</%def>

<%def name="smp_primary_core_region_begin()">
    <% global smp_region_counter %>
    % if exec_module.is_smp and platform_apis.smp_primary_core_guard != "":
        if (!${platform_apis.smp_primary_core_guard}(ctx)) goto smp_skip_${smp_region_counter};
    % endif
</%def>

<%def name="smp_primary_core_region_end()">
    <% global smp_region_counter %>
    % if exec_module.is_smp and platform_apis.smp_primary_core_guard != "":
        smp_skip_${smp_region_counter}: ;
    % endif
    <% smp_region_counter += 1 %>
</%def>

<%def name="profile_region_begin()">
    %if exec_module.timer_start_fn != "" and platform_apis.init_platform != "":
        ${exec_module.timer_start_fn}();
    % endif
</%def>

<%def name="profile_region_end(label)">
    %if exec_module.timer_start_fn != "" and platform_apis.init_platform != "":
        real_args[8 + ${label}] += ${exec_module.timer_stop_fn}();
    % endif
</%def>

<%def name="profile_var(label)">
    %if exec_module.timer_start_fn != "" and platform_apis.init_platform != "":
        real_args[8 + ${label}] +=
    % endif
</%def>


<%def name="node_inner()">

    ${"void" if platform_apis.init_platform!="" else "int"} ${node_fullname}${"_inner" if platform_apis.init_platform!="" else ""}(
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
    
    ## In Symmetric Multi-Core Execution check if the current core should run
    <%self:smp_configured_core_guard/>

    ## In offloaded execution retrieve the arguments from the args pointer
    % if platform_apis.init_platform!="":
        // Rerieve tensor pointers from host trough known args location
        volatile unsigned *real_args = (unsigned int *) args;
        <% tensor_cnt = 0 if exec_module.separate_build else 0 %>
        % for tensor in {**match_node.var_tensors,**match_node.output_tensors}.values():
            void* ${"var_" if tensor.tensor_type=="var" else "out_"}${tensor.name}_pt = (void*) real_args[${tensor_cnt}];\
            <% tensor_cnt += 1 %>
        % endfor
        % for const_tensor in schedule.tensors.values():
            % if const_tensor.tensor_type=="const":
                % if exec_module.separate_build:
                void* ${name}_${const_tensor.name}_pt = (void*) real_args[${tensor_cnt}];\
                <% tensor_cnt += 1 %>
                % else:
                void* ${name}_${const_tensor.name}_pt = ${name}_${const_tensor.name}_data;\
                % endif
            % endif
        % endfor
    % endif

    MatchCtx* ctx = ${name}_ctx;

    <%self:smp_primary_core_region_begin/>

        % if platform_apis.init_module != "":
            ${platform_apis.init_module}(ctx);
        % endif

        // Variable Tensors
        % for var in match_node.var_tensors.values():
            ${name}_${var.name}->base_pt = var_${var.name}_pt;
            ${name}_${var.name}->pt = var_${var.name}_pt;
            ${name}_${var.name}->pts[${memory_hierarchy["var"][-1].name}] = var_${var.name}_pt;
        % endfor
        // Output Tensors
        % for out in match_node.output_tensors.values():
            ${name}_${out.name}->base_pt = out_${out.name}_pt;
            ${name}_${out.name}->pt = out_${out.name}_pt;
            ${name}_${out.name}->pts[${memory_hierarchy["output"][-1].name}] = out_${out.name}_pt;
        % endfor
        // Constant Tensors
        % for const_tensor in schedule.tensors.values():
            % if const_tensor.tensor_type=="const":
                ${name}_${const_tensor.name}->base_pt = ${name}_${const_tensor.name}_pt;
                ${name}_${const_tensor.name}->pt = ${name}_${const_tensor.name}_pt;
                ${name}_${const_tensor.name}->pts[${memory_hierarchy["const"][-1].name}] = ${name}_${const_tensor.name}_pt;
            % endif
        % endfor

    <%self:smp_primary_core_region_end/>

    #ifndef __MATCH_TEST_NODE_WITH_HELPER__

    <%self:smp_primary_core_region_begin/>

        // Intermediate Tensors
        % for intermediate_tensor in schedule.tensors.values():
            % if intermediate_tensor.tensor_type=="intermediate":
                ${name}_${intermediate_tensor.name}->base_pt = ${target.alloc_fn}(${intermediate_tensor.prod_shape}*sizeof(${c_dtype(intermediate_tensor.dtype)}));
                ${name}_${intermediate_tensor.name}->pt = ${name}_${intermediate_tensor.name}->base_pt;
                ${name}_${intermediate_tensor.name}->pts[${memory_hierarchy["intermediate"][-1].name}] = ${intermediate_tensor.name}->base_pt;
            % endif
        % endfor

        % for mem_level in set([mem_ for k,v in memory_hierarchy.items() for mem_ in v]):
            % if mem_level.sw_controlled and mem_level.name!=target.host_memory:
                volatile void* ${mem_level.name}_base_pt = ${mem_apis.init_memory[mem_level.name]}(ctx);
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

    <%self:smp_primary_core_region_end/>

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
            <%self:smp_primary_core_region_begin/>
                <%self:profile_region_begin/>
                % for mem_transfer in lp.mem_transfers:
                    // compute the offset from the top level memory to obtain the correct tile for the transfer
                    % for t_dim_idx, t_dim in enumerate(mem_transfer.tensor.dims):
                        % if t_dim in match_node.dependent_dims and len(set([schedule.tensor_tiles[mem_transfer.tensor.name][idx].tiled_dims[t_dim_idx].size for idx in range(len(schedule.tensor_tiles[mem_transfer.tensor.name]))]))!=1:
                            ${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].size = ${name}_${t_dim.name}->curr_size; // this dim is not independent
                            ${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].max_size = ${name}_${t_dim.name}->curr_max_size; // this dim is not independent
                            ${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].start_idx = ${name}_${t_dim.name}->global_idx;
                            ${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].curr_idx = ${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].start_idx;
                        % endif
                        % if any([lp.dim==t_dim for lp in block.loops[last_transfer_of_tensor_block[(mem_transfer.tensor.name, block_idx)][0]:loop_idx]]):
                            ${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].start_idx = ${name}_${t_dim.name}->global_idx;
                            ${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].curr_idx = ${name}_${mem_transfer.tensor.name}_tiles_[${mem_transfer.mem}*${mem_transfer.tensor.num_dims}+${t_dim_idx}].start_idx;
                        % endif
                    % endfor
                    % if (mem_transfer.tensor.is_fused or mem_transfer.tensor.unsupported_layout) and mem_apis.get_size_of_fused_tensor!="" and mem_apis.get_pt_of_fused_tensor!="":
                        int ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)} = ${mem_apis.get_size_of_fused_tensor}(ctx,${name}_${mem_transfer.tensor.name});
                        tile_mem_offset = ${mem_apis.get_pt_of_fused_tensor}(ctx,${name}_${mem_transfer.tensor.name});
                        void* ${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt${c_unique_num_tile(mem_transfer.tensor.name)} = ${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.top_mem}] + (tile_mem_offset>0?tile_mem_offset:0);
                    % else:
                        int ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)} = ${mem_transfer.tensor.c_offset_expr_size_sw_mem(mem_transfer.mem, name)};
                        tile_mem_offset = ${mem_transfer.tensor.c_offset_expr_sw_mem(mem_transfer.top_mem, schedule, block_idx, loop_idx, name)};
                        void* ${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt${c_unique_num_tile(mem_transfer.tensor.name)} = ${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.top_mem}] + (tile_mem_offset>0?tile_mem_offset:0);
                    % endif
                    ${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.mem}] = ${mem_transfer.mem}_base_pt + ${mem_transfer.mem}_curr_pt_offset;
                    ${name}_${mem_transfer.tensor.name}->pt = ${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.mem}];
                    ${mem_transfer.mem}_curr_pt_offset += ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)};
                    % if mem_transfer.tensor.tensor_type != "output":
                        // call API for ${exec_module.name}-specific memory transfer handling
                        <%self:profile_var label="3"/> ${mem_apis.mem_transfer}(
                            ctx,${name}_${mem_transfer.tensor.name},${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt${c_unique_num_tile(mem_transfer.tensor.name)},
                            ${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.mem}],
                            MATCH_SW_LOAD_TENSOR,MATCH_${"CONST" if mem_transfer.tensor.tensor_type=="const" else "VAR"}_TENSOR,
                            ${mem_transfer.top_mem},${mem_transfer.mem}
                        );
                        % if sync_apis.must_sync_after_load:
                            // sync after each single load as the SW transfer require...
                            ${sync_apis.wait_load}(ctx);
                        % else:
                            block_${block_idx}_loads++;
                        % endif
                    % endif
                    <% add_tile_to_tensor_at_block_and_loop(mem_transfer.tensor.name, block_idx, loop_idx, mem_transfer.mem)%>
                % endfor
            <%self:profile_region_end label="1"/>
            <%self:smp_primary_core_region_end/>
            ## finished sw controlled loads and stores
            % if exec_module.backend_constraints_check(match_node,schedule,block,lp,loop_idx) and block.loop_idx_end_sw_controlled_loads>=loop_idx:
                <% break %>
            % endif 

            /*
                for(${name}_block_${block_idx}_loop_${lp.name}_set();
                    ${name}_block_${block_idx}_loop_${lp.name}_end();
                    ${name}_block_${block_idx}_loop_${lp.name}_update()){
            */
            <%self:smp_primary_core_region_begin/>
                ${name}_block_${block_idx}_loop_${lp.name}_set();
            <%self:smp_primary_core_region_end/>
            <%self:smp_barrier/>
            while(${name}_block_${block_idx}_loop_${lp.name}_end()) {
        % endfor

        <%self:smp_primary_core_region_begin/>
            <%self:profile_region_begin/>
                % if not sync_apis.must_sync_after_load and sync_apis.wait_load!="":
                    // sync with the SW controlled transfers
                    if(block_${block_idx}_loads) ${sync_apis.wait_load}(ctx);
                    block_${block_idx}_loads = 0;
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
                            tile_mem_offset = ${mem_apis.get_pt_of_fused_tensor}(ctx,${name}_${tensor.name});
                        % else:
                            tile_mem_offset = ${tensor.c_offset_expr_sw_mem(last_transfer_of_tensor_block[(tensor.name, block_idx)][1], schedule, block_idx, loop_idx, name)};
                        % endif
                        ${name}_${tensor.name}->pt = ${name}_${tensor.name}->pts[${last_transfer_of_tensor_block[(tensor.name, block_idx)][1]}] + (tile_mem_offset>0?tile_mem_offset:0);
                        % for t_dim_idx, t_dim in enumerate(tensor.dims):
                            % if t_dim in match_node.dependent_dims and [lp.dim in t_dim.dim_dependency.dependencies for lp in block.loops[last_transfer_of_tensor_block[(tensor.name, block_idx)][0]:loop_idx]] or any([lp.dim==t_dim for lp in block.loops[last_transfer_of_tensor_block[(tensor.name, block_idx)][0]:loop_idx]]):
                                ${name}_${tensor.name}_tiles_[${last_transfer_of_tensor_block[(tensor.name, block_idx)][1]}*${tensor.num_dims}+${t_dim_idx}].curr_idx = ${name}_${t_dim.name}->global_idx;
                            % endif
                        % endfor
                    % endif
                % endfor
            <%self:profile_region_end label="1"/>
        <%self:smp_primary_core_region_end/>
        
        <%self:smp_barrier/>

        <%self:smp_primary_core_region_begin/>
            <%self:profile_region_begin/>
        <%self:smp_primary_core_region_end/>
            % if block.backend == "MATCH":
                % if block.parallel_execution:
                    ${platform_apis.parallelize_task}(match_backend_block_${block_idx}_computation,${block.num_tasks},ctx);
                % else:
                    match_backend_block_${block_idx}_computation(ctx);
                % endif
            % else:
                ${comp_apis.compute_tile}(ctx);
            % endif

            % if sync_apis.must_sync_after_computation:
                % if block.backend=="MATCH" and block.parallel_execution and sync_apis.wait_parallel_tasks:
                    ${sync_apis.wait_parallel_tasks}(ctx);
                % elif sync_apis.wait_tile_computation!="":
                    ${sync_apis.wait_tile_computation}(ctx);
                % endif
                % elif block.num_buffers_for_computation!=1:
                    buffer_for_computation_idx++;
                % if sync_apis.wait_tile_computation!="":
                    // the buffer is full, wait before the next iteration...
                    if(block_${block_idx}_buffer_for_computation_idx>=BLOCK_${block_idx}_NUM_BUFFERS_FOR_COMPUTATION){
                        % if block.backend=="MATCH" and block.parallel_execution and sync_apis.wait_buffer_parallel_tasks!="":
                        ${sync_apis.wait_buffer_parallel_tasks}(ctx);
                        % elif sync_apis.wait_buffer_tile_computation!="":
                        ${sync_apis.wait_buffer_tile_computation}(ctx);
                        % endif
                    }
                % endif
            % endif

        <%self:smp_barrier/>
        
        <%self:smp_primary_core_region_begin/>
            <%self:profile_region_end label="0"/>
        <%self:smp_primary_core_region_end/>
        
        ## close braces and save output
        % for loop_idx_ in range(loop_idx,-1,-1):
            <% lp = block.loops[loop_idx_] %>
            % if not exec_module.backend_constraints_check(match_node,schedule,block,block.loops[loop_idx_],loop_idx_) and block.loop_idx_end_sw_controlled_loads>=loop_idx_:
                <%self:smp_primary_core_region_begin/>
                ${name}_block_${block_idx}_loop_${lp.name}_update();
                <%self:smp_primary_core_region_end/>
                <%self:smp_barrier/>
                }
                <%self:smp_primary_core_region_begin/>
                ${name}_block_${block_idx}_loop_${lp.name}_reset();
                <%self:smp_primary_core_region_end/>
            % endif

            <%self:smp_primary_core_region_begin/>
                <%self:profile_region_begin/>
                    % for mem_transfer in block.loops[loop_idx_].mem_transfers:
                        <% free_transfer_unique_tile(mem_transfer.tensor.name) %>
                        % if mem_transfer.tensor.tensor_type == "output":
                            // call API for ${exec_module.name}-specific memory transfer handling
                            <%self:profile_var label="4"/> ${mem_apis.mem_transfer}(
                                ctx,${name}_${mem_transfer.tensor.name},${mem_transfer.tensor.name}_${mem_transfer.top_mem}_tile_pt${c_unique_num_tile(mem_transfer.tensor.name)},
                                ${name}_${mem_transfer.tensor.name}->pts[${mem_transfer.mem}],
                                MATCH_SW_STORE_TENSOR,MATCH_OUT_TENSOR,
                                ${mem_transfer.top_mem},${mem_transfer.mem}
                            );
                            % if sync_apis.must_sync_after_store:
                                // sync after each single store as the SW transfer require...
                                ${sync_apis.wait_store}(ctx);
                            % endif
                        % endif
                        % if block.num_buffers_for_computation==1 or mem_transfer.mem!=memory_hierarchy[mem_transfer.tensor.tensor_type][0].name:
                            ${mem_transfer.mem}_curr_pt_offset -= ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)};
                        % elif block.num_buffers_for_computation>1 and mem_transfer.mem==memory_hierarchy[mem_transfer.tensor.tensor_type][0].name:
                            if(block_${block_idx}_buffer_for_computation_idx>=BLOCK_${block_idx}_NUM_BUFFERS_FOR_COMPUTATION)
                                ${mem_transfer.mem}_curr_pt_offset -= ${mem_transfer.tensor.name}_${mem_transfer.mem}_tile_size${c_unique_num_tile(mem_transfer.tensor.name)};
                        % endif
                    % endfor
                <%self:profile_region_end label="2"/>
            <%self:smp_primary_core_region_end/>
        % endfor

    % endfor

    <%self:smp_primary_core_region_begin/>
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
    <%self:smp_primary_core_region_end/>
    #endif
    #ifdef __MATCH_TEST_NODE_WITH_HELPER__
    run_node_schedule_nn(ctx);
    #endif

    <%self:smp_barrier/>

    % if platform_apis.init_platform=="":
        return;
    % endif
    }

</%def>