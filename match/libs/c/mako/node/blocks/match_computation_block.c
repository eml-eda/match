<%def name='match_computation_block(block_idx,block)'>

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

</%def>


