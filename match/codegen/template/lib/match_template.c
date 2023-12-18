#include <match_dimensions.h>
#include <match_kernel.h>
#include <match_tile_indexes.h>
#include <match_target_params.h>
% for inc_name in include_list:
#include <${inc_name}>
% endfor

% if weights_and_constants["len"]>0:
## define weights for layers containing weights
${types["mem_data_macro"]} uint8_t weights_and_constants_${func_number} ${weights_and_constants['shape']} = ${weights_and_constants['value']};
% endif

static typedef struct layer_loops_indexes_t
{
    % for lp in sw_for_loops:
    unsigned int ${lp["fullname"]};
    % endfor
}layer_loops_indexes;


% for operand in operands:
typedef struct dimension_${operand}_${func_number}_t{
    dimension${operand} match_dim;
    % for rel_loop_dim in ordered_relevant_loops[operand]:
    % if operand in input_operands and rel_loop_dim in padded_dims:
    ## TODO: IS THIS NEEDED?
    int max_${input_dim_mapping[rel_loop_dim]}_kernel_size;
    % endif
    % endfor
    % for tile_dim in tiling_sizes[operand].keys():
    int tile_${tile_dim}_size;
    % endfor
}dimension_${operand}_${func_number};

% for mem_level in ordered_operand_memories[operand]: 
static void setup_relative_tile_idxs_${operand}_${mem_level}(tile_indexes_${operand}* tile_idxs,layer_loops_indexes* layer_loops_idxs){
    % for rel_dim in ordered_relevant_loops[operand]:
    tile_idxs->tile_${rel_dim if operand not in input_operands else input_dim_mapping[rel_dim]}=0;
    % endfor
    % for sw_loop in sw_for_loops:
    % if sw_loop[f"mem_{operand}"]==mem_level and sw_loop["name"] in ordered_relevant_loops[operand]:
    tile_idxs->tile_${sw_loop["name"] if operand not in input_operands else input_dim_mapping[sw_loop["name"]]}+=
    layer_loops_idxs->${sw_loop["fullname"]}
    ## TILED DIMENSION
    % if sw_loop["index"]>0:
    dim->tile_${sw_loop["fullname"] if operand not in input_operands else f"{input_dim_mapping[sw_loop['name']]}_{sw_loop['index']}"}
    % endif
    ;
    % endif
    % endfor
}
% if operand in input_operands and layer_has_padding:
static void calc_padding_${operand}_${mem_level}(){
    // IX dimension pads
    dim->common_dim.pad_IX_x=abs_tile_idxs->tile_IX - dim->common_dim.overlap_IX_x +
        abs_tile_idxs->tile_IX*(${layer_attrs['strides']["IX"]}-1) <= 0 ?
        dim->common_dim.overlap_IX_x - tile_idxs->tile_IX + tile_idxs->tile_IX * (${layer_attrs['strides']["IX"]}-1)
        : 0;
    dim->common_dim.pad_IX_y=abs_tile_idxs->tile_IX + dim->common_dim.size_IX[${mem_level}]+ 
        dim->common_dim.overlap_IX_y + tile_idxs->tile_IX * (${layer_attrs['strides']["IX"]}-1) > ${layer_attrs['loop_sizes']["OX"]} ?
        abs_tile_idxs->tile_IX+dim->common_dim.size_IX[${mem_level}]+dim->common_dim.overlap_IX_y+
        abs_tile_idxs->tile_IX*(${layer_attrs['strides']["IX"]}-1) - ${layer_attrs['loop_sizes']["OX"]} : 0;
    // IY dimension pads
    dim->common_dim.pad_IY_x=abs_tile_idxs->tile_IY - dim->common_dim.overlap_IY_x +
        abs_tile_idxs->tile_IY*(${layer_attrs['strides']["IY"]}-1) <= 0 ?
        dim->common_dim.overlap_IY_x - tile_idxs->tile_IY + tile_idxs->tile_IY * (${layer_attrs['strides']["IY"]}-1)
        : 0;
    dim->common_dim.pad_IY_y=abs_tile_idxs->tile_IY + dim->common_dim.size_IY[${mem_level}]+ 
        dim->common_dim.overlap_IY_y + tile_idxs->tile_IY * (${layer_attrs['strides']["IY"]}-1) > ${layer_attrs['loop_sizes']["OY"]} ?
        abs_tile_idxs->tile_IY+dim->common_dim.size_IY[${mem_level}]+dim->common_dim.overlap_IY_y+
        abs_tile_idxs->tile_IY*(${layer_attrs['strides']["IY"]}-1) - ${layer_attrs['loop_sizes']["OY"]} : 0;
}
% endif
% endfor
% endfor

static void match_initial_setup_${func_number}(
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
    dim_${i_operand}->common_dim.overlap_IX_x=0;dim${i_operand}->common_dim.overlap_IX_y=0;
    dim_${i_operand}->common_dim.overlap_IY_x=0;dim${i_operand}->common_dim.overlap_IY_y=0;
    dim_${i_operand}->common_dim.pad_IX_x=0;dim${i_operand}->common_dim.pad_IX_y=0;
    dim_${i_operand}->common_dim.pad_IY_x=0;dim${i_operand}->common_dim.pad_IY_y=0;
    % endfor
    % else:
    % for i_operand in input_operands:
    dim_${i_operand}->common_dim.overlap_IX_x=${overlaps["OX"][0]}; dim_${i_operand}->common_dim.overlap_IY_x=${overlaps["OY"][0]};
    dim_${i_operand}->common_dim.overlap_IX_y=${overlaps["OX"][1]}; dim_${i_operand}->common_dim.overlap_IY_y=${overlaps["OY"][1]};
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
    kernel->dilation_x=${layer_attrs["dilation"][0]};kernel->dilation_y=${layer_attrs["dilation"][1]};
    kernel->activation_function=${layer_attrs["activation"]};
    kernel->stride_x=${layer_attrs["strides"]["IX"]};kernel->stride_y=${layer_attrs["strides"]["IY"]};
    kernel->right_shift=${0 if "right_shift.param.0" not in weights_and_constants["single_costants"] else weights_and_constants["single_costants"]["right_shift.param.0"]};
    % for operand in operands:
    kernel->dim_${operand}=&(dim_${operand}->common_dim);
    % endfor
    % if not layer_has_padding:
    kernel->pad_IX_x=0;kernel->pad_IX_y=0;kernel->pad_IY_x=0;kernel->pad_IY_y=0;
    % endif
}

void __attribute__ ((noinline)) ${func_name}_inner(void* args)
{
    // recover args of function
    unsigned int *real_args = (unsigned int *) args;
    % for i_index,i_operand in enumerate(input_operands):
    void * input_${i_operand}_pt = (void *) real_args[${i_index}];
    % endfor
    void * output_pt = (void *) real_args[${len(input_operands)}];
    // define dimension structures
    % for i_operand in input_operands:
    dimension${i_operand}_${func_number} dim_${i_operand}; tile_indexes_${i_operand} abs_tile_idxs_${i_operand};
    % endfor
    % if layer_has_weights:
    dimensionW_${func_number} dim_W; tile_indexes_W abs_tile_idxs_W;
    % endif
    dimensionO_${func_number} dim_O; tile_indexes_O abs_tile_idxs_O;
    layer_loops_indexes layer_loops_idxs;
    // setup(init) dimensions
    match_initial_setup_${func_number}(
        % for i_operand in input_operands:
        &dim_${i_operand}, &abs_tile_idxs_${i_operand},
        % endfor
        % if layer_has_weights:
        &dim_W, &abs_tile_idxs_W,
        % endif
        &dim_O, &abs_tile_idxs_O, &layer_loops_idxs
    );
    // init memory api call --> malloc L1 memory and other params
    ${apis["startup_memory_and_pattern"]}(
        % for init_mem_op in input_operands+[op for op in operands if op not in input_operands and op!='O']+['O']:
        ${size_each_level[init_mem_op]},${str(db_opportunities[init_mem_op]).lower()},&dim_${init_mem_op},
        % endfor
        ${padding_c_array},${strides_c_array},${pattern_name},${func_number}
    );
    % for i_operand in input_operands:
    void* base_${i_operand}_pt=input_${i_operand}_pt;
    % endfor
    % if layer_has_weights:
    void* base_W_pt=weights_and_constants_${func_number};
    % endif
    void* base_O_pt=output_pt;
    ## PARAMS USEFUL FOR KERNEL
    unsigned int kernel_O_curr_ext=0,kernel_O_curr_int=0,kernel_O_prev_ext=0,kernel_O_prev_int=0,kernel_offset_ext_O=0,iter=0;
    % if "nn.bias_add" in pattern_operations:
    int starting_bias_pt=0;
    % endif
    ## one sized dims removed, not useful for now
    % if "multiply" in pattern_operations:
    void* starting_batchnorm_mul=0x0;
    void* starting_batchnorm_add=0x0;
    % endif
    ## to replace with a function tho
    ## setup static kernel params
    ${types["kernel_struct"]} kernel;
    init_common_kernel_static_params(
        &kernel,
        % for i_operand in input_operands:
        &dim_${i_operand}, 
        % endfor
        % if layer_has_weights:
        &dim_W,
        % endif
        &dim_O
    );
    ${apis["init_other_kernel_params"]}(
        &kernel,
        % for i_operand in input_operands:
        &(dim_${i_operand}.common_dim), 
        % endfor
        % if layer_has_weights:
        &(dim_W.common_dim),
        % endif
        &(dim_O.common_dim)
    );
    ## TODO: MANAGE LOOPS
    
    ## FOR LOOPS
    % for idx,for_loop in enumerate(sw_for_loops):
    ## MEMCOPY
    % for mem_transfer_operand in memory_transfers[for_loop['fullname']]:
    tile_indexes_${mem_transfer_operand} tile_idxs_${mem_transfer_operand}_${idx}_relative;
    setup_relative_tile_idxs_${mem_transfer_operand}_${default_mem[mem_transfer_operand] if idx==0 else sw_for_loops[idx-1][f"mem_{mem_transfer_operand}"]}(&tile_idxs_${mem_transfer_operand}_${idx}_relative);
    % if mem_transfer_operand in input_operands and layer_has_padding:
    calc_padding_${mem_transfer_operand}_${mem_level}();
    % endif
    unsigned int loop_${mem_transfer_operand}_${idx}_ext_pt = ${apis[f"pointer_offset_{mem_transfer_operand}"]}(
        &tile_idxs_${mem_transfer_operand}_${idx}_relative,
        ${default_mem[mem_transfer_operand] if idx==0 else sw_for_loops[idx-1][f"mem_{mem_transfer_operand}"]}
    ) + ${f"base_{mem_transfer_operand}_pt" if last_movements[mem_transfer_operand]==-1 else f"loop_{mem_transfer_operand}_{last_movements[mem_transfer_operand]}_int_pt"};
    unsigned int loop_${mem_transfer_operand}_${idx}_int_pt = ${apis[f"mem_transfer_{mem_transfer_operand}"]}(
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
    tile_indexes_${operand} tile_idxs_${operand}_kernel_relative;
    setup_relative_tile_idxs_${operand}_${for_loop[f"mem_{operand}"]}(&tile_idxs_${operand}_kernel_relative);
    unsigned int kernel_${operand}_pt = ${apis[f"pointer_offset_{operand}"]}(&tile_idxs_${operand}_kernel_relative,${for_loop[f"mem_{operand}"]});
    % if operand in input_operands and layer_has_padding:
    calc_padding_${operand}_${for_loop[f"mem_{operand}"]}();
    % endif
    % else:
    ## NOTHING HAS CHANGED
    unsigned int kernel_${operand}_pt = loop_${operand}_${idx}_int_pt;
    % endif
    % endfor
    ${sync_apis["any_transfers"]}();
    if(iter>0)  ${sync_apis["prev_computation"]}();
    ${apis["computation"]}(&kernel,${pattern_name});
    % if last_movements["O"]!=idx:
    kernel_offset_ext_O=${apis["pointer_offset_O"]}(
        &tile_idxs_O_kernel_relative,
        ${sw_for_loops[last_movements["O"]-1]["mem_O"]}
    );
    % endif
    kernel_O_curr_ext=loop_${sw_for_loops[last_movements['O']]["fullname"]}_O_tile_pt+kernel_offset_ext_O;
    kernel_O_curr_int=kernel_O_pt;
    ## ITER CAN BE USED TO HAVE DOUBLE BUFFERING
    ${apis["copy_out_curr_computation"]}(
        &dim_O,kernel_O_curr_int,kernel_O_curr_ext,
        ${for_loop["mem_O"]},${sw_for_loops[last_movements["O"]-1]["mem_O"]}
    );
    if(iter>0)  ${apis["copy_out_prev_computation"]}(
        &dim_O,kernel_O_prev_int,kernel_O_prev_ext,
        ${for_loop["mem_O"]},${sw_for_loops[last_movements["O"]-1]["mem_O"]}
    );
    kernel_O_prev_ext=kernel_O_curr_ext;kernel_O_prev_int=kernel_O_curr_int;
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
    substract_relative_idxs_${mem_transfer_operand}();
    % endfor
    % endfor
    ${sync_apis["prev_computation"]}();
    ${apis["copy_out_prev_computation"]}(
        &dim_O,kernel_O_prev_int,kernel_O_prev_ext,
        ${for_loop["mem_O"]},${sw_for_loops[last_movements["O"]-1]["mem_O"]}
    );
    ${sync_apis["any_transfers"]}();
    ${apis["shutdown_mem"]}();
}

void __attribute__ ((noinline)) ${func_name}(
    % for i_operand in input_operands:
    void* input_${i_operand}_pt,
    % endfor
    void* output_pt)
{
    unsigned int args[${len(input_operands)+1}];
    % for ind,i_operand in enumerate(input_operands):
    args[${ind}] = (unsigned int) input_${i_operand}_pt;
    % endfor
    args[${len(input_operands)}] = (unsigned int) output_pt;
    ${apis["init_platform"]}(${func_name}_inner, args);
    return 0;
}