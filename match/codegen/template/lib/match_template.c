% for inc_name in include_list:
#include <${inc_name}>
% endfor

static ${types.mem_data_macro_and_type} padding_array ${padding_c_array["shape"]} = ${padding_c_array["value"]};
static ${types.mem_data_macro_and_type} strides_array ${strides_c_array["shape"]} = ${strides_c_array["value"]};
% if weights_and_constants["len"]>0:
## define weights for layers containing weights
static ${types.mem_data_macro_and_type} weights_and_constants ${weights_and_constants["shape"]} = ${weights_and_constants["value"]};
% else:
static ${types.mem_data_macro_and_type} weights_and_constants [1]={0};
% endif

typedef struct layer_loops_indexes_t
{
    % for lp in sw_for_loops:
    unsigned int ${lp["fullname"]};
    % endfor
}layer_loops_indexes;


% for operand in operands:
typedef struct dimension_${operand}_${func_number}_t{
    dimension_${operand} common_dim;
    % for tile_dim in tiling_sizes[operand].keys():
    int tile_${tile_dim}_size;
    % endfor
}dimension_${operand}_${func_number};

% for mem_level in ordered_operand_memories[operand]+["mem_computation"]:
% if mem_level!="mem_computation": 
static void setup_relative_tile_idxs_${operand}_${mem_level}(
    dimension_${operand}_${func_number}* dim,
    tile_indexes_${operand}* tile_idxs,
    layer_loops_indexes* layer_loops_idxs
){
    % for rel_dim in ordered_relevant_loops[operand]:
    tile_idxs->tile_${rel_dim if operand not in input_operands else input_dim_mapping[rel_dim]}=0;
    % endfor
    % for sw_loop in sw_for_loops:
    % if sw_loop[f"mem_{operand}"]==mem_level and sw_loop["name"] in ordered_relevant_loops[operand]:
    tile_idxs->tile_${sw_loop["name"] if operand not in input_operands else input_dim_mapping[sw_loop["name"]]}+=
    layer_loops_idxs->${sw_loop["fullname"]}
    ## TILED DIMENSION
    % if sw_loop["index"]>0:
    *dim->tile_${sw_loop["fullname"] if operand not in input_operands else f"{input_dim_mapping[sw_loop['name']]}_{sw_loop['index']}"}_size
    % endif
    ;
    % endif
    % endfor
    ## EVALUATE IDX ALSO WITH STRIDES
    % if operand in input_operands:
    % for stride_dim,stride_val in layer_attrs["strides"].items():
    //FIXED: bug for tile number calculation, stride doesn't matter
    //tile_idxs->tile_${stride_dim}*=${stride_val};
    % endfor
    % endif
}
% endif
% if operand in input_operands and layer_has_padding:
static void calc_padding_${operand}_${mem_level}(dimension_${operand}_${func_number}* dim,tile_indexes_${operand}* abs_tile_idxs){
    // IX dimension pads
    dim->common_dim.pad_IX_x=abs_tile_idxs->tile_IX - dim->common_dim.overlap_IX_x  <= 0 ?
        dim->common_dim.overlap_IX_x - abs_tile_idxs->tile_IX : 0;
    dim->common_dim.pad_IX_y=abs_tile_idxs->tile_IX + dim->common_dim.size_IX[${mem_level}]+ 
        dim->common_dim.overlap_IX_y > ${layer_attrs['loop_sizes']["IX"]} ?
        abs_tile_idxs->tile_IX+dim->common_dim.size_IX[${mem_level}]+dim->common_dim.overlap_IX_y - ${layer_attrs['loop_sizes']["IX"]} : 0;
    // IY dimension pads
    dim->common_dim.pad_IY_x=abs_tile_idxs->tile_IY - dim->common_dim.overlap_IY_x  <= 0 ?
        dim->common_dim.overlap_IY_x - abs_tile_idxs->tile_IY : 0;
    dim->common_dim.pad_IY_y=abs_tile_idxs->tile_IY + dim->common_dim.size_IY[${mem_level}]+ 
        dim->common_dim.overlap_IY_y > ${layer_attrs['loop_sizes']["IY"]} ?
        abs_tile_idxs->tile_IY+dim->common_dim.size_IY[${mem_level}]+dim->common_dim.overlap_IY_y - ${layer_attrs['loop_sizes']["IY"]} : 0;
}
% endif
% endfor
% endfor

static void match_initial_setup(
    % for i_operand in input_operands:
    dimension_${i_operand}_${func_number}* dim_${i_operand}, tile_indexes_${i_operand}* tile_idxs_${i_operand},
    % endfor
    % if layer_has_weights:
    dimension_W_${func_number}* dim_W, tile_indexes_W* tile_idxs_W,
    % endif
    dimension_O_${func_number}* dim_O, tile_indexes_O* tile_idxs_O,
    layer_loops_indexes* layer_loops_idxs
){
    ## TILES SIZES
    % for operand in operands:
    % for tlname,tile in tiling_sizes[operand].items():
    dim_${operand}->tile_${tlname}_size=${tile["size"]};
    % endfor
    % endfor
    ## SIZES AT EACH MEMORY LEVEL AND OVERALL DIMENSION SIZES AND ZERO UP TILE IDXS
    % for operand in operands:
    % for rel_dim in ordered_relevant_loops[operand]:
    tile_idxs_${operand}->tile_${rel_dim if operand not in input_operands else input_dim_mapping[rel_dim]}=0;
    dim_${operand}->common_dim.dim_${rel_dim if operand not in input_operands else input_dim_mapping[rel_dim]}_size=${layer_attrs['loop_sizes'][rel_dim]};
    % for mem_name,size_at_mem in size_loops_mem[operand][rel_dim].items():
    dim_${operand}->common_dim.size_${rel_dim if operand not in input_operands else input_dim_mapping[rel_dim]}[${mem_name}]=${size_at_mem};
    % endfor
    % endfor
    % endfor
    ## INPUT DIMENSIONS PADDING AND OVERLAPS SETTING
    % if not layer_has_padding:
    % for i_operand in input_operands:
    dim_${i_operand}->common_dim.overlap_IX_x=0;dim_${i_operand}->common_dim.overlap_IX_y=0;
    dim_${i_operand}->common_dim.overlap_IY_x=0;dim_${i_operand}->common_dim.overlap_IY_y=0;
    dim_${i_operand}->common_dim.pad_IX_x=0;dim_${i_operand}->common_dim.pad_IX_y=0;
    dim_${i_operand}->common_dim.pad_IY_x=0;dim_${i_operand}->common_dim.pad_IY_y=0;
    % endfor
    % else:
    % for i_operand in input_operands:
    dim_${i_operand}->common_dim.overlap_IX_x=${overlaps["IX"][0]}; dim_${i_operand}->common_dim.overlap_IY_x=${overlaps["IY"][0]};
    dim_${i_operand}->common_dim.overlap_IX_y=${overlaps["IX"][1]}; dim_${i_operand}->common_dim.overlap_IY_y=${overlaps["IY"][1]};
    dim_${i_operand}->common_dim.pad_IX_x=${layer_data.padding["IX"][0]}; dim_${i_operand}->common_dim.pad_IY_x=${layer_data.padding["IY"][0]};
    dim_${i_operand}->common_dim.pad_IX_y=${layer_data.padding["IX"][1]}; dim_${i_operand}->common_dim.pad_IY_y=${layer_data.padding["IY"][1]};
    % endfor
    % endif
    % for lp in sw_for_loops:
    layer_loops_idxs->${lp["fullname"]}=0;
    % endfor
    return;
}

static void init_common_kernel_static_params(
    common_kernel* kernel,
    % for i_operand in input_operands:
    dimension_${i_operand}_${func_number}* dim_${i_operand},
    % endfor
    % if layer_has_weights:
    dimension_W_${func_number}* dim_W,
    % endif
    dimension_O_${func_number}* dim_O
){
    init_common_kernel_params(kernel,${pattern_name},${comp_apis.specific_pattern});
    init_kernel_dimension_params_${"_".join(input_operands+(["W","O"] if layer_has_weights else ["O"]))}(
        kernel,
    % for i_operand in input_operands:
    &(dim_${i_operand}->common_dim),${sw_for_loops[::-1][0][f"mem_{i_operand}"]},${layer_data.operand_precision[i_operand]},
    % endfor
    % if layer_has_weights:
    &(dim_W->common_dim),${sw_for_loops[::-1][0]["mem_W"]},${layer_data.operand_precision["W"]},
    % endif
    &(dim_O->common_dim),${sw_for_loops[::-1][0][f"mem_O"]},${layer_data.operand_precision["O"]}
    );
    // TODO: set each attribute of the pattern correctly
    //kernel->dilation_x=----layer_attrs["dilation"][0]---;kernel->dilation_y=---layer_attrs["dilation"][1]---;
    kernel->activation_function=${int(bool("activation" in layer_attrs and layer_attrs["activation"]))};
    kernel->stride_x=${layer_attrs["strides"]["IX"]};kernel->stride_y=${layer_attrs["strides"]["IY"]};
    kernel->right_shift=${0 if "right_shift.param.0" not in weights_and_constants["single_costants"] else weights_and_constants["single_costants"]["right_shift.param.0"]};
}


void __attribute__ ((noinline)) ${func_name}_inner(void* args)
{
    % if "start_codegen" in debug_level:
    printf("Start of ${func_name} codegen function!\n");
    % endif
    ## setup static kernel params
    ${types.kernel_struct} kernel;
    common_kernel comm_kernel;
    ## define dimension structures
    % for i_operand in input_operands:
    dimension_${i_operand}_${func_number} dim_${i_operand}; tile_indexes_${i_operand} abs_tile_idxs_${i_operand};
    % endfor
    % if layer_has_weights:
    dimension_W_${func_number} dim_W; tile_indexes_W abs_tile_idxs_W;
    % endif
    dimension_O_${func_number} dim_O; tile_indexes_O abs_tile_idxs_O;
    layer_loops_indexes layer_loops_idxs;
    // setup(init) dimensions
    match_initial_setup(
        % for i_operand in input_operands:
        &dim_${i_operand}, &abs_tile_idxs_${i_operand},
        % endfor
        % if layer_has_weights:
        &dim_W, &abs_tile_idxs_W,
        % endif
        &dim_O, &abs_tile_idxs_O, &layer_loops_idxs
    );
    kernel.common_kernel=&comm_kernel;
    init_common_kernel_static_params(
        &comm_kernel,
        % for i_operand in input_operands:
        &dim_${i_operand}, 
        % endfor
        % if layer_has_weights:
        &dim_W,
        % endif
        &dim_O
    );
    ${platform_apis.set_task_id}(&comm_kernel);
    // recover args of function
    unsigned int *real_args = (unsigned int *) args;
    % for i_index,i_operand in enumerate(input_operands):
    void * input_${i_operand}_pt = (void *) real_args[${i_index}];
    % endfor
    void * output_pt = (void *) real_args[${len(input_operands)}];
    % for operand in operands:
    unsigned int mem_level_${operand}_sizes ${size_each_level[operand]["shape"]} = ${size_each_level[operand]["value"]};
    % endfor
    % for i_operand in input_operands:
    unsigned int base_${i_operand}_pt=input_${i_operand}_pt;
    % endfor
    % if layer_has_weights:
    unsigned int base_W_pt=weights_and_constants;
    % endif
    unsigned int base_O_pt=output_pt;
    ## PARAMS USEFUL FOR KERNEL
    unsigned int kernel_O_curr_ext=0,kernel_O_curr_int=0,kernel_O_prev_ext=0,kernel_O_prev_int=0,kernel_offset_ext_O=0,iter=0;
    ## to replace with a function tho
    // init memory api call --> malloc L1 memory and other params
    ${mem_apis.startup_memory}(
        &comm_kernel,
        % for init_mem_op in input_operands+[op for op in operands if op not in input_operands and op!='O']+['O']:
        mem_level_${init_mem_op}_sizes,${int(db_opportunities[init_mem_op])},&(dim_${init_mem_op}.common_dim),
        % endfor
        padding_array,strides_array
    );
    
    ${comp_apis.init_other_kernel_params}( &kernel );
    ## TODO: MANAGE LOOPS
    
    ## FOR LOOPS
    % for idx,for_loop in enumerate(sw_for_loops):
    ## MEMCOPY
    % for mem_transfer_operand in memory_transfers[for_loop['fullname']]:
    tile_indexes_${mem_transfer_operand} tile_idxs_${mem_transfer_operand}_${idx}_relative;
    setup_relative_tile_idxs_${mem_transfer_operand}_${default_mem[mem_transfer_operand] if idx==0 else sw_for_loops[idx-1][f"mem_{mem_transfer_operand}"]}(
        &dim_${mem_transfer_operand},
        &tile_idxs_${mem_transfer_operand}_${idx}_relative,
        &layer_loops_idxs    
    );
    add_relative_idxs_${mem_transfer_operand}(&tile_idxs_${mem_transfer_operand}_${idx}_relative,&abs_tile_idxs_${mem_transfer_operand});
    % if mem_transfer_operand in input_operands and layer_has_padding:
    calc_padding_${mem_transfer_operand}_${sw_for_loops[idx][f'mem_{mem_transfer_operand}']}(&dim_${mem_transfer_operand},&abs_tile_idxs_${mem_transfer_operand});
    % endif
    % if last_movements[mem_transfer_operand]!=-1:
    // sync prev transfer in a multi memory system with data dependency
    if(layer_loops_idxs.${for_loop["fullname"]}==0)
        ${sync_apis.sync_multilevel_transfer}(
            &comm_kernel,
            ${default_mem[mem_transfer_operand] if idx==0 else sw_for_loops[idx-1][f'mem_{mem_transfer_operand}']},
            ${for_loop[f'mem_{mem_transfer_operand}']});
    % endif
    unsigned int loop_${mem_transfer_operand}_${idx}_ext_pt = ${mem_apis.pointer_offset[mem_transfer_operand]}(
        &comm_kernel,
        &tile_idxs_${mem_transfer_operand}_${idx}_relative,
        ${default_mem[mem_transfer_operand] if idx==0 else sw_for_loops[idx-1][f"mem_{mem_transfer_operand}"]}
    ) + ${f"base_{mem_transfer_operand}_pt" if last_movements[mem_transfer_operand]==-1 else f"loop_{mem_transfer_operand}_{last_movements[mem_transfer_operand]}_int_pt"};
    unsigned int loop_${mem_transfer_operand}_${idx}_int_pt = ${mem_apis.mem_transfer[mem_transfer_operand]}(
        &comm_kernel,
        &(dim_${mem_transfer_operand}.common_dim),
        loop_${mem_transfer_operand}_${idx}_ext_pt,
        ${default_mem[mem_transfer_operand] if idx==0 else sw_for_loops[idx-1][f'mem_{mem_transfer_operand}']},
        ${for_loop[f'mem_{mem_transfer_operand}']}
    );
    ## USEFUL FOR MULTI MEMORY SYSTEMS
    <%
    last_movements[mem_transfer_operand]=idx
    %>
    % endfor
    ## WRITE LOOP IF NOT LAST
    % if idx!=len(sw_for_loops)-1:
    for(;layer_loops_idxs.${for_loop["fullname"]}<${for_loop["size"]};layer_loops_idxs.${for_loop["fullname"]}++){
    
    % else:
    ## COMPUTATIONAL PART
    % for operand in operands:
    % if last_movements[operand]!=idx:
    ## CALC IDXS AGAIN, THEY MAY HAVE CHANGED FOR SOME OPERAND (UNEVEN MAPPING)
    % if operand in input_operands and layer_has_padding:
    ## WE ARE INSIDE THE LOWEST MEMORY LEVEL SO WE SHOULDNT HAVE A NEGATIVE OFFSET
    dim_${operand}.common_dim.pad_IX_x=dim_${operand}.common_dim.overlap_IX_x;dim_${operand}.common_dim.pad_IX_y=dim_${operand}.common_dim.overlap_IX_y;
    dim_${operand}.common_dim.pad_IY_x=dim_${operand}.common_dim.overlap_IY_x;dim_${operand}.common_dim.pad_IY_y=dim_${operand}.common_dim.overlap_IY_y;
    % endif
    tile_indexes_${operand} tile_idxs_${operand}_kernel_relative;
    setup_relative_tile_idxs_${operand}_${for_loop[f"mem_{operand}"]}(
        &dim_${operand},
        &tile_idxs_${operand}_kernel_relative,
        &layer_loops_idxs
    );
    add_relative_idxs_${operand}(&tile_idxs_${operand}_kernel_relative,&abs_tile_idxs_${operand});
    unsigned int kernel_${operand}_pt = loop_${operand}_${last_movements[operand]}_int_pt+${mem_apis.pointer_offset[operand]}(&comm_kernel,&tile_idxs_${operand}_kernel_relative,${for_loop[f"mem_{operand}"]});
    % if operand in input_operands and layer_has_padding:
    calc_padding_${operand}_mem_computation(&dim_${operand},&abs_tile_idxs_${operand});
    kernel_set_padding(kernel.common_kernel,&(dim_${operand}.common_dim));
    % endif
    % else:
    ## NOTHING HAS CHANGED
    unsigned int kernel_${operand}_pt = loop_${operand}_${idx}_int_pt;
    % if operand in input_operands and layer_has_padding:
    kernel_set_padding(kernel.common_kernel,&(dim_${operand}.common_dim));
    % endif
    % endif
    kernel.common_kernel->${operand}_pt = kernel_${operand}_pt;
    % endfor
    % if layer_has_weights:
    ${mem_apis.pattern_constants_loading}(&kernel,iter,
    &abs_tile_idxs_W,
    % if last_movements["W"]!=idx:
    &tile_idxs_W_kernel_relative,
    % else:
    &tile_idxs_W_${last_movements["W"]}_relative,
    % endif
    weights_and_constants);
    % endif
    ${sync_apis.async_transfers}(&comm_kernel);
    if(iter>0)  ${sync_apis.prev_computation}(&comm_kernel);
    ${comp_apis.innermost_computation}(&kernel);
    ${sync_apis.curr_computation}(&comm_kernel);
    % if last_movements["O"]!=idx:
    kernel_offset_ext_O=${mem_apis.pointer_offset["O"]}(
        &comm_kernel,
        &tile_idxs_O_kernel_relative,
        ${sw_for_loops[last_movements["O"]-1]["mem_O"]}
    );
    % endif
    kernel_O_curr_ext=loop_O_${last_movements["O"]}_ext_pt+kernel_offset_ext_O;
    kernel_O_curr_int=kernel_O_pt;
    ## ITER CAN BE USED TO HAVE DOUBLE BUFFERING
    ${mem_apis.copy_out_curr_computation}(
        &comm_kernel,
        &dim_O,kernel_O_curr_int,kernel_O_curr_ext,
        ${for_loop["mem_O"]},${sw_for_loops[last_movements["O"]-1]["mem_O"]}
    );
    ${sync_apis.wait_output_transfers}(&comm_kernel);
    if(iter>0)  ${mem_apis.copy_out_prev_computation}(
        &comm_kernel,
        &dim_O,kernel_O_prev_int,kernel_O_prev_ext,
        ${for_loop["mem_O"]},${sw_for_loops[last_movements["O"]-1]["mem_O"]}
    );
    kernel_O_prev_ext=kernel_O_curr_ext;kernel_O_prev_int=kernel_O_curr_int;
    % for operand in operands:
    % if last_movements[operand]!=idx:
    substract_relative_idxs_${operand}(&tile_idxs_${operand}_kernel_relative,&abs_tile_idxs_${operand});
    % endif
    % endfor
    iter++;
    % endif
    % if 'O' in memory_transfers[for_loop['fullname']] and idx!=len(sw_for_loops)-1:
    // transfer output (for multi level systems) not supported now!
    % endif
    % endfor
    ## close braces and save output
    % for idx in reversed(range(len(sw_for_loops))):
    % if idx!=len(sw_for_loops)-1:
    }
    % endif
    layer_loops_idxs.${sw_for_loops[idx]['fullname']}=0;
    % for mem_transfer_operand in memory_transfers[sw_for_loops[idx]['fullname']]:
    substract_relative_idxs_${mem_transfer_operand}(&tile_idxs_${mem_transfer_operand}_${idx}_relative,&abs_tile_idxs_${mem_transfer_operand});
    % endfor
    % endfor
    ${sync_apis.prev_computation}(&comm_kernel);
    ${mem_apis.copy_out_prev_computation}(
        &comm_kernel,&dim_O,kernel_O_prev_int,kernel_O_prev_ext,
        ${for_loop["mem_O"]},${sw_for_loops[last_movements["O"]-1]["mem_O"]}
    );
    ${sync_apis.async_transfers}(&comm_kernel);
    ${mem_apis.shutdown_mem}(&comm_kernel);
    % if "end_codegen" in debug_level:
    printf("End of ${func_name} codegen function!\n");
    % endif
}

void __attribute__ ((noinline)) ${func_name}(
    % for i_operand in input_operands:
    void* input_${i_operand}_pt,
    % endfor
    void* output_pt)
{
    ## setup static kernel params
    ${types.kernel_struct} kernel;
    common_kernel comm_kernel;
    ## define dimension structures
    % for i_operand in input_operands:
    dimension_${i_operand}_${func_number} dim_${i_operand}; tile_indexes_${i_operand} abs_tile_idxs_${i_operand};
    % endfor
    % if layer_has_weights:
    dimension_W_${func_number} dim_W; tile_indexes_W abs_tile_idxs_W;
    % endif
    dimension_O_${func_number} dim_O; tile_indexes_O abs_tile_idxs_O;
    layer_loops_indexes layer_loops_idxs;
    // setup(init) dimensions
    match_initial_setup(
        % for i_operand in input_operands:
        &dim_${i_operand}, &abs_tile_idxs_${i_operand},
        % endfor
        % if layer_has_weights:
        &dim_W, &abs_tile_idxs_W,
        % endif
        &dim_O, &abs_tile_idxs_O, &layer_loops_idxs
    );
    kernel.common_kernel=&comm_kernel;
    init_common_kernel_static_params(
        &comm_kernel,
        % for i_operand in input_operands:
        &dim_${i_operand}, 
        % endfor
        % if layer_has_weights:
        &dim_W,
        % endif
        &dim_O
    );
    unsigned int args[${len(input_operands)+1}];
    % for ind,i_operand in enumerate(input_operands):
    args[${ind}] = (unsigned int) input_${i_operand}_pt;
    % endfor
    args[${len(input_operands)}] = (unsigned int) output_pt;
    ${platform_apis.init_platform}(${func_name}_inner, args, &comm_kernel);
    return 0;
}