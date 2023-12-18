#include <match.h>

% if layer_has_weights:
## define weights for layers containing weights
${mem_data_macro} uint8_t weights_${func_number} ${weights['shape']} = ${weights['value']};
% endif

% for operand in operands:
typedef struct dimension_${operand}_${func_number}_t{
    dimension${operand} match_dim;
    % for rel_loop_dim in ordered_relevant_loops[operand]:
    % if operand in input_operands and rel_loop_dim in padded_dims:
    ## TODO: IS THIS NEEDED?
    int max_${inp_dim_mapping[rel_loop_dim]}_kernel_size;
    % endif
    % endfor
    % for tile_dim in tiling_sizes[operand].keys():
    int tile_${tile_dim}_size;
    % endfor
}dimension_${operand}_${func_number};

% if operand=='O':
static void memcopyresult_${func_number}(dimensionO_${func_number}* dim,void* src_pt,void* dst_pt,int src_memory_level,int dst_memory_level){
    return memcopyresult_${layer_name}(&dim->accel_dim,src_pt,dst_pt,src_memory_level,dst_memory_level);
}
% elif operand=='W':
static void* memcopy_bias_${func_number}(dimension${operand}_${func_number}* dim,void* startpt,int src_memory_level,int dst_memory_level){
    return memcopy_bias(&dim->accel_dim,startpt,src_memory_level,dst_memory_level);
}
% endif
static void* memcopy_${operand}_${func_number}(dimension${operand}_${func_number}* dim,void* startpt,int src_memory_level,int dst_memory_level){
    return memcopy_${operand}_${layer_name}(&dim->accel_dim,startpt,src_memory_level,dst_memory_level);
}
% endfor

static void add_sizes_to_kernel_O_${func_number}(int tile_K,int tile_OY,int tile_OX,Kernel_parameters_${func_number}* kernel){
    kernel->k=tile_K;
    kernel->oy=tile_OY;
    kernel->ox=tile_OX;
}

% if layer_has_weights:
static void add_sizes_to_kernel_W_${func_number}(int tile_K,int tile_C,int tile_FY,int tile_FX,Kernel_parameters_${func_number}* kernel){
    /* could be useful if W is the last dimension, but not in majority of cases 
    kernel->k=tile_K;
    kernel->c=tile_C;
    */
    kernel->fy=tile_FY;
    kernel->fx=tile_FX;
}
//inline 
static unsigned int get_W_address_${func_number}(int tile_K_id_relative,int tile_K_id_absolute,int tile_C_id_relative,int tile_C_id_absolute,
                        int tile_FY_id_relative,int tile_FY_id_absolute,int tile_FX_id_relative,int tile_FX_id_absolute,dimensionW_${func_number}* dim,int mem_level){
    % if layer_name!='depthwise_conv_2d':
    return (tile_K_id_relative*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]*dim->accel_dim.size_C[mem_level]+
            tile_FY_id_relative*dim->accel_dim.size_FX[mem_level]*dim->accel_dim.size_C[mem_level]+
            tile_FX_id_relative*dim->accel_dim.size_C[mem_level]+tile_C_id_relative);
    % else:
    return (tile_K_id_relative*dim->accel_dim.size_C[mem_level]*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]+
            tile_C_id_relative*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]+
            tile_FY_id_relative*dim->accel_dim.size_FX[mem_level]+tile_FX_id_relative);
    % endif:
}

//inline 
unsigned int get_W_address_kernel_${func_number}(int tile_K_id_relative,int tile_K_id_absolute,int tile_C_id_relative,int tile_C_id_absolute,
                        int tile_FY_id_relative,int tile_FY_id_absolute,int tile_FX_id_relative,int tile_FX_id_absolute,dimensionW_${func_number}* dim,Kernel_parameters_${func_number}* kernel,int mem_level){
    % if layer_name!='depthwise_conv_2d':
    return (tile_K_id_relative*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]*dim->accel_dim.size_C[mem_level]+
            tile_FY_id_relative*dim->accel_dim.size_FX[mem_level]*dim->accel_dim.size_C[mem_level]+
            tile_FX_id_relative*dim->accel_dim.size_C[mem_level]+tile_C_id_relative);
    % else:
    return (tile_K_id_relative*dim->accel_dim.size_C[mem_level]*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]+
            tile_C_id_relative*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]+
            tile_FY_id_relative*dim->accel_dim.size_FX[mem_level]+tile_FX_id_relative);
    % endif:
    
}
% endif

% for input_operand in input_operands:
static void add_sizes_to_kernel_${input_operand}_${func_number}(int tile_C,int tile_IY,int tile_IX,Kernel_parameters_${func_number}* kernel){
    kernel->c=tile_C;
    kernel->cy=tile_IY;
    kernel->cx=tile_IX;
}
//inline 
static int get_${input_operand}_address_${func_number}(int tile_C_id_relative,int tile_C_id_absolute,int tile_IY_id_relative,int tile_IY_id_absolute,
        % if len(padded_dims)>0:
    int stride_Y,
    % endif
                        int tile_IX_id_relative,int tile_IX_id_absolute,
    % if len(padded_dims)>0:
    int stride_X,
    % endif
    dimension${input_operand}_${func_number}* dim,int mem_level){
    % if len(padded_dims)>0:
    int tile_IY_id=tile_IY_id_relative-dim->accel_dim.addsize_IY+dim->accel_dim.pad_IY_x;
    int tile_IX_id=tile_IX_id_relative-dim->accel_dim.addsize_IX+dim->accel_dim.pad_IX_x;
    % else:
    int tile_IY_id=tile_IY_id_relative;
    int tile_IX_id=tile_IX_id_relative;
    % endif
    % if layer_name!='depthwise_conv_2d':
    return (tile_IY_id${'*stride_Y' if len(padded_dims)>0 else ''})*dim->accel_dim.size_IX[mem_level]*dim->accel_dim.size_C[mem_level]+(tile_IX_id${'*stride_X' if len(padded_dims)>0 else ''})*dim->accel_dim.size_C[mem_level]+tile_C_id_relative;
    % else:
    return (tile_IY_id${'*stride_Y' if len(padded_dims)>0 else ''})*dim->accel_dim.size_IX[mem_level]*dim->accel_dim.size_C[mem_level]+(tile_IX_id${'*stride_X' if len(padded_dims)>0 else ''})*dim->accel_dim.size_C[mem_level]+tile_C_id_relative;
    % endif    
}

//inline 
static int get_${input_operand}_address_kernel_${func_number}(int tile_C_id_relative,int tile_C_id_absolute,int tile_IY_id_relative,int tile_IY_id_absolute,
    % if len(padded_dims)>0:
    int stride_Y,
    % endif
                        int tile_IX_id_relative,int tile_IX_id_absolute,
    % if len(padded_dims)>0:
    int stride_X,
    % endif
    dimension${input_operand}_${func_number}* dim,Kernel_parameters_${func_number}* kernel,int mem_level){
    int tile_IY_id=tile_IY_id_relative;
    int tile_IX_id=tile_IX_id_relative;
    % if layer_name!='depthwise_conv_2d':
    return (tile_IY_id${'*stride_Y' if len(padded_dims)>0 else ''})*dim->accel_dim.size_IX[mem_level]*dim->accel_dim.size_C[mem_level]+(tile_IX_id${'*stride_X' if len(padded_dims)>0 else ''})*dim->accel_dim.size_C[mem_level]+tile_C_id_relative;
    % else:
    return (tile_C_id_relative*dim->accel_dim.size_IY[mem_level]*dim->accel_dim.size_IX[mem_level])+((tile_IY_id${'*stride_Y' if len(padded_dims)>0 else ''})*dim->accel_dim.size_IX[mem_level])+(tile_IX_id${'*stride_X' if len(padded_dims)>0 else ''});
    % endif
}
% endfor

//inline 
static unsigned int get_O_address_${func_number}(int tile_K_id_relative,int tile_K_id_absolute,int tile_OY_id_relative,int tile_OY_id_absolute,
                        int tile_OX_id_relative,int tile_OX_id_absolute,dimensionO_${func_number}* dim,int mem_level){
    % if layer_name!='depthwise_conv_2d':
    return (tile_OY_id_relative*dim->accel_dim.size_OX[mem_level]*dim->accel_dim.size_K[mem_level]+tile_OX_id_relative*dim->accel_dim.size_K[mem_level]+tile_K_id_relative);
    % else:
    return (tile_OY_id_relative*dim->accel_dim.size_OX[mem_level]*dim->accel_dim.size_K[mem_level]+tile_OX_id_relative*dim->accel_dim.size_K[mem_level]+tile_K_id_relative);
    % endif
}

//inline 
static unsigned int get_O_address_kernel_${func_number}(int tile_K_id_relative,int tile_K_id_absolute,int tile_OY_id_relative,int tile_OY_id_absolute,
                        int tile_OX_id_relative,int tile_OX_id_absolute,dimensionO_${func_number}* dim,Kernel_parameters_${func_number}* kernel,int mem_level){
    % if layer_name!='depthwise_conv_2d':
    return (tile_OY_id_relative*dim->accel_dim.size_OX[mem_level]*dim->accel_dim.size_K[mem_level]+tile_OX_id_relative*dim->accel_dim.size_K[mem_level]+tile_K_id_relative);
    % else:
    return (tile_K_id_relative*dim->accel_dim.size_OY[mem_level]*dim->accel_dim.size_OX[mem_level])+(tile_OY_id_relative*dim->accel_dim.size_OX[mem_level])+tile_OX_id_relative;
    % endif
}

static unsigned int get_bias_address_${func_number}(int tile_K_bias_id_relative,int tile_K_bias_id_absolute){
    return tile_K_bias_id_relative*4;
}

static unsigned int get_bias_address_kernel_${func_number}(int tile_K_bias_kernel_id_relative,int tile_K_bias_kernel_id_absolute){
    return tile_K_bias_kernel_id_relative*4;
}


static void match_wrapper_computational_function_${func_number}(Kernel_parameters* kernel,)

static void setup_dimensions_${func_number}(
    % for i_operand in input_operands:
    dimension${i_operand}_${func_number}* dim${i_operand},
    % endfor
    % if len(input_operands)==1:
    dimensionW_${func_number}* dimW,
    % endif
    dimensionO_${func_number}* dimO){
    % for tlname,tile in tiling_sizes.items():
    % for operand in operands:
    % if tile["name"] in ordered_relevant_loops[operand]:
    dim${dimname}->tile_${input_dim_mapping[tile["name"]] if operand in input_operands else tile["name"]}${"" if tile["index"]==0 else f'_{tile["index"]}'}_size=${tile["size"]};
    % endif
    % endfor
    % endfor
    % for operand in operands:
    % for rel_dim in ordered_relevant_loops[operand]:
    dim${operand}->accel_dim.dim_${rel_dim if operand not in input_operands else inp_dim_mapping[rel_dim]}_size=${attrs['loop_sizes'][rel_dim]};
    % for mem_name,size_at_mem in size_loops_mem[operand][rel_dim].items():
    dim${operand}->accel_dim.size_${rel_dim if operand not in input_operands else inp_dim_mapping[rel_dim]}[${mem_name}]=${size_at_mem};
    % endfor
    % endfor
    % endfor
    % if len(padded_dims)==0:
    % for i_operand in input_operands:
    dim${i_operand}->accel_dim.overlap_IX_x=0;dim${i_operand}->accel_dim.overlap_IX_y=0;
    dim${i_operand}->accel_dim.overlap_IY_x=0;dim${i_operand}->accel_dim.overlap_IY_y=0;
    dim${i_operand}->accel_dim.pad_IX_x=0;dim${i_operand}->accel_dim.pad_IX_y=0;
    dim${i_operand}->accel_dim.pad_IY_x=0;dim${i_operand}->accel_dim.pad_IY_y=0;
    % endfor
    % else:
    % for p_dim in padded_dims:
    % for i_operand in input_operands:
    dim${i_operand}->accel_dim.overlap_${input_dim_mapping[p_dim]}_x=${overlap[input_dim_mapping[p_dim]][0]}
    dim${i_operand}->accel_dim.overlap_${input_dim_mapping[p_dim]}_y=${overlap[input_dim_mapping[p_dim]][1]}
    dim${i_operand}->accel_dim.pad_${input_dim_mapping[p_dim]}_x=${attrs['padding'][inp_dim_mapping[p_dim]][0]};
    dim${i_operand}->accel_dim.pad_${input_dim_mapping[p_dim]}_y=${attrs['padding'][input_dim_mapping[p_dim]][1]};
    % endfor
    % endfor
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
    dimension${i_operand}_${func_number} dim_${i_operand};
    % endfor
    % if layer_has_weights:
    dimensionW_${func_number} dim_W;
    % endif
    dimensionO_${func_number} dim_O;
    // setup(init) dimensions
    setup_dimensions_${func_number}(
        % for i_operand in input_operands:
        &dim_${i_operand},
        % endfor
        % if len(input_operands)==1:
        &dim_W,
        % endif
        &dim_O
    );
    // init memory api call --> malloc L1 memory and other params
    ${init_memory_api}(
        % for init_mem_op in input_operands+[op for op in operands if op not in input_operands and op!='O']+['O']:
        ${size_each_level[init_mem_op]},${db_opportunities[init_mem_op]},dim_
        % endfor
        ${attrs['loop_sizes']['K']},dim_${input_operands[0]}.accel_dim.size_IY[shared_l1],${attrs['loop_sizes']['FY']},${attrs['padding']['IY'][0]},${attrs['padding']['IY'][1]},${len(input_operands)}
    );

    % for i_operand in input_operands:
    void* base_${i_operand}_pt=input_${i_operand}_pt;
    % endfor
    % if layer_has_weights:
    void* base_W_pt=weights_${func_number};
    % endif
    void* base_O_pt=output_pt;
    int offset_pt=1;
    int firstpad=0;
    int lastpad=0;
    int iter=0;
    unsigned char* kernel_O_prev_tile_pt=0x0;
    void* kernel_O_prev_pt=0x0;
    % if layer_has_bias:
    int starting_bias_pt=0;
    % endif
    ## setup attributes like the ones required by the operator (disabled now)
    % for operand in operands:
    % for rel_loop in ordered_relevant_loops[operand]:
    int tile_${input_dim_mapping[rel_loop] if operand in input_operands else rel_loop}_${operand}_id_absolute=0;
    % endfor
    % endfor
    ## setup cycles name in case some may be used
    % for for_loop in for_loops:
    int ${for_loop['fullname']}=0;
    % endfor
    % for name in one_sized_dims:
    int ${name}=0;
    % endfor
    % if layer_has_batchnorm:
    void* starting_batchnorm_mul=0x0;
    void* starting_batchnorm_add=0x0;
    % endif
    ## setup kernel params
    ${kernel_struct_name} kernel;
    kernel.dilation_x=${attrs["dilation"][0]};
    kernel.dilation_y=${attrs["dilation"][1]};
    kernel.activation_function=${attrs["activation"]};
    kernel.stride_x=${attrs["strides"]["IX"]};
    kernel.stride_y=${attrs["strides"]["IY"]};
    kernel.output_shift=${attrs['output_shift']};
    ${init_other_kernel_params}(&kernel);
    ## FOR LOOPS
    % for idx,for_loop in enumerate(sw_for_loops):
    ## MEMCOPY
    % for mem_transfer_layer in for_loop_mem_transfers[for_loop['fullname']]:
    ## SIZE_LOOP SI PUO CALCOLARE GIA PRIMA --> SKIP
    ## SINGLE MEM APPROACH
    % for rel_loop_dim,prev_rel_loops in {dim:[lp for lp in sw_for_loops[last_movements[mem_transfer_layer]:idx+1] if lp['name']==dim] for dim in ordered_relevant_loops[mem_transfer_layer]}.items():
    ## GET RELATIVE TILE NUMBER
    int tile_${rel_loop_dim if mem_transfer_layer not in input_operands else inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_${idx}_id_relative=0;
    % for prl in prev_rel_loops:
    tile_${rel_loop_dim if mem_transfer_layer not in input_operands else inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_${idx}_id_relative+=${prl['fullname']}
    % if prl['index']>0:
    *dim_${mem_transfer_layer}.tile_${f"{prl['name'] if mem_transfer_layer not in input_operands else inp_dim_mapping[prl['name']]}_{'' if prl['index']==0 else str(prl['index'])+'_'}size" };
    % else:
    ;
    % endif
    % endfor
    ## ADD RELATIVE TO ABSOLUTE
    tile_${rel_loop_dim if mem_transfer_layer not in input_operands else inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute+=tile_${rel_loop_dim if mem_transfer_layer not in input_operands else inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_${idx}_id_relative;
    ## LAYER IS INPUT AND PADDING
    % if mem_transfer_layer in input_operands and rel_loop_dim in padded_dims:
    ## HANDLE PADDING
    dim_${mem_transfer_layer}.accel_dim.pad_${inp_dim_mapping[rel_loop_dim]}_x=tile_${inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute - dim_${mem_transfer_layer}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]}+tile_${inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute*(${attrs['strides'][inp_dim_mapping[rel_loop_dim]]}-1)<=0? dim_${mem_transfer_layer}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]} -tile_${inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute+ tile_${inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute*(${attrs['strides'][inp_dim_mapping[rel_loop_dim]]}-1): 0;
    dim_${mem_transfer_layer}.accel_dim.pad_${inp_dim_mapping[rel_loop_dim]}_y=tile_${inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute+dim_${mem_transfer_layer}.accel_dim.size_${inp_dim_mapping[rel_loop_dim]}[${for_loop[f'mem_{mem_transfer_layer}']}]+dim_${mem_transfer_layer}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]}+tile_${inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute*(${attrs['strides'][inp_dim_mapping[rel_loop_dim]]}-1)>${attrs['loop_sizes'][inp_dim_mapping[rel_loop_dim]]}?
        tile_${inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute+dim_${mem_transfer_layer}.accel_dim.size_${inp_dim_mapping[rel_loop_dim]}[${for_loop[f'mem_{mem_transfer_layer}']}]+dim_${mem_transfer_layer}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]}+tile_${inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute*(${attrs['strides'][inp_dim_mapping[rel_loop_dim]]}-1)-${attrs['loop_sizes'][inp_dim_mapping[rel_loop_dim]]} : 0;
    % endif
    % endfor
    unsigned char* loop_${for_loop['fullname']}_${mem_transfer_layer}_tile_pt=base_${mem_transfer_layer}_pt+get_${mem_transfer_layer}_address_${func_number}(
        % for rel_loop_dim in ordered_relevant_loops[mem_transfer_layer]:
        tile_${rel_loop_dim if mem_transfer_layer not in input_operands else inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_${idx}_id_relative,
        tile_${rel_loop_dim if mem_transfer_layer not in input_operands else inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute,
        % if mem_transfer_layer in input_operands and rel_loop_dim in padded_dims:
        ${attrs['strides'][inp_dim_mapping[rel_loop_dim]]},
        % endif
        % endfor
        &dim_${mem_transfer_layer},${default_mem[mem_transfer_layer] if idx==0 else sw_for_loops[idx-1][f'mem_{mem_transfer_layer}']}
    );
    void* loop_${for_loop['fullname']}_${mem_transfer_layer}_pt=memcopy_${mem_transfer_layer}_${func_number}(&dim_${mem_transfer_layer},loop_${for_loop['fullname']}_${mem_transfer_layer}_tile_pt,${default_mem[mem_transfer_layer] if idx==0 else sw_for_loops[idx-1][f'mem_{mem_transfer_layer}']},${for_loop[f'mem_{mem_transfer_layer}']});
    <%
    last_movements[mem_transfer_layer]=idx
    %>
    % if mem_transfer_layer=='W' and 'bias_add' in attrs and attrs['bias_add']:
    ## BIAS TO LOAD
    if(iter==0)
        starting_bias_pt=memcopy_bias_${func_number}(&dim_W,base_W_pt+(dim_W.accel_dim.size_K[dram]*dim_W.accel_dim.size_C[dram]*dim_W.accel_dim.size_FY[dram]*dim_W.accel_dim.size_FX[dram]),${default_mem[mem_transfer_layer] if idx==0 else sw_for_loops[idx-1][f'mem_{mem_transfer_layer}']},${for_loop[f'mem_{mem_transfer_layer}']});
    % endif
    % if mem_transfer_layer=='W' and 'batchnorm' in attrs and attrs['batchnorm']:
    starting_batchnorm_mul=memcopy_batchnorm_mul(&dim_W.accel_dim,base_W_pt+(dim_W.accel_dim.size_K[dram]*dim_W.accel_dim.size_C[dram]*dim_W.accel_dim.size_FY[dram]*dim_W.accel_dim.size_FX[dram]));
    starting_batchnorm_add=memcopy_batchnorm_add(&dim_W.accel_dim,base_W_pt+(dim_W.accel_dim.size_K[dram]*dim_W.accel_dim.size_C[dram]*dim_W.accel_dim.size_FY[dram]*dim_W.accel_dim.size_FX[dram])+dim_W.accel_dim.size_K[dram]*4);
    % endif
    % endfor
    ## LAST LOOP --> KERNEL PT COMPUTATION AND KERNEL FUNCTION CALL
    % if idx==len(sw_for_loops)-1:
    
    % for kernel_operand in operands:
    % for rel_loop_dim,prev_rel_loops in {dim:[lp for lp in sw_for_loops[last_movements[kernel_operand]:idx] if lp['name']==dim] for dim in ordered_relevant_loops[kernel_operand]}.items():
    ## GET RELATIVE TILE NUMBER
    int tile_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_kernel_id_relative=0;
    % for prl in prev_rel_loops:
    tile_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_kernel_id_relative+=${prl['fullname']}
    % if prl['index']>0:
    *dim_${kernel_operand}.tile_${f"{prl['name'] if kernel_operand not in input_operands else inp_dim_mapping[prl['name']]}_{'' if prl['index']==0 else str(prl['index'])+'_'}size" };
    % else:
    ;
    % endif
    % endfor
    ## ADD RELATIVE TO ABSOLUTE
    tile_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute+=tile_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_kernel_id_relative;
    ## OPERAND IS INPUT AND PADDING
    % if kernel_operand in input_operands and rel_loop_dim in padded_dims:
    ## HANDLE PADDING
    firstpad=tile_${inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute-dim_${kernel_operand}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]}+tile_${inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute*(${attrs['strides'][inp_dim_mapping[rel_loop_dim]]}-1)<=0? dim_${kernel_operand}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]} -tile_${inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute +tile_${inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute*(${attrs['strides'][inp_dim_mapping[rel_loop_dim]]}-1) : 0;
    lastpad=tile_${inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute+dim_${kernel_operand if last_movements[kernel_operand]>last_movements['O'] else 'O'}.accel_dim.size_${inp_dim_mapping[rel_loop_dim] if last_movements[kernel_operand]>last_movements['O'] else rel_loop_dim}[shared_l1]+dim_${kernel_operand}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]}+tile_${inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute*(${attrs['strides'][inp_dim_mapping[rel_loop_dim]]}-1)>${attrs['loop_sizes'][inp_dim_mapping[rel_loop_dim]]}?
    tile_${inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute+dim_${kernel_operand if last_movements[kernel_operand]>last_movements['O'] else 'O'}.accel_dim.size_${inp_dim_mapping[rel_loop_dim] if last_movements[kernel_operand]>last_movements['O'] else rel_loop_dim}[${for_loop[f'mem_{kernel_operand}']}]+dim_${kernel_operand}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]}+tile_${inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute*(${attrs['strides'][inp_dim_mapping[rel_loop_dim]]}-1)-${attrs['loop_sizes'][inp_dim_mapping[rel_loop_dim]]} : 0;
    dim_${kernel_operand}.max_${inp_dim_mapping[rel_loop_dim]}_kernel_size=dim_${kernel_operand if last_movements[kernel_operand]>last_movements['O'] else 'O'}.accel_dim.size_${inp_dim_mapping[rel_loop_dim] if last_movements[kernel_operand]>last_movements['O'] else rel_loop_dim}[${for_loop[f'mem_{kernel_operand}']}]+dim_${kernel_operand}.accel_dim.addsize_${inp_dim_mapping[rel_loop_dim]};
    kernel.pad_${inp_dim_mapping[rel_loop_dim]}_x=firstpad;
    kernel.pad_${inp_dim_mapping[rel_loop_dim]}_y=lastpad;
    % endif
    % endfor
    void* loop_kernel_${kernel_operand}_tile_pt=loop_${sw_for_loops[last_movements[kernel_operand]]["fullname"]}_${kernel_operand}_pt
    % if kernel_operand=='O':
    ;
    unsigned int kernel_offset_O=
    % else:
    +
    % endif
    get_${kernel_operand}_address_kernel_${func_number}(
        % for rel_loop_dim in ordered_relevant_loops[kernel_operand]:
        tile_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_kernel_id_relative,
        tile_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute,
        % if kernel_operand in input_operands and rel_loop_dim in padded_dims:
        ${attrs['strides'][inp_dim_mapping[rel_loop_dim]]},
        % endif
        % endfor
        &dim_${kernel_operand},&kernel,${for_loop[f'mem_{kernel_operand}']}
    );
    % if kernel_operand=='O':
    loop_kernel_O_tile_pt+=kernel_offset_O;
    % elif kernel_operand=='W' and 'bias_add' in attrs and attrs['bias_add']:
    void* loop_kernel_bias_tile_pt=starting_bias_pt+get_bias_address_kernel_${func_number}(tile_K_W_kernel_id_relative,tile_K_W_id_absolute);
    % elif kernel_operand=='W' and 'batchnorm' in attrs and attrs['batchnorm']:
    void* loop_kernel_batch_mul_pt=starting_batchnorm_mul+4*tile_K_W_kernel_id_relative;
    void* loop_kernel_batch_add_pt=starting_batchnorm_add+4*tile_K_W_kernel_id_relative;
    % endif
    add_sizes_to_kernel_${kernel_operand}_${func_number}(
        % for rel_loop_dim in ordered_relevant_loops[kernel_operand]:
        dim_${kernel_operand}.accel_dim.size_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}[${for_loop[f'mem_{kernel_operand}']}],
        % endfor
        &kernel
    );
    % endfor
    % if 'W' not in operands:
    void* loop_kernel_bias_tile_pt=0x0;
    % endif
    dma_transfer_wait(transfer);
    % if len(input_operands)==1:
    if (iter > 0) 
      pi_team_offload_wait();
    % endif
    % if 'bias' in attrs and attrs['bias']:
    kernel.bias_pt=loop_kernel_bias_tile_pt;
    % endif
    kernel.dim_O=&dim_O;
    kernel.mem_O=${for_loop['mem_O']};
    kernel.O_pt=loop_kernel_O_tile_pt;
    % if 'batchnorm' in attrs and attrs['batchnorm']:
    kernel.batchnorm_mul=starting_batchnorm_mul;
    kernel.batchnorm_add=starting_batchnorm_add;
    % endif
    % for i_operand in input_operands:
    kernel.dim_${i_operand}=&dim_${i_operand};
    kernel.mem_${i_operand}=${for_loop[f'mem_{i_operand}']};
    kernel.${i_operand}_pt=loop_kernel_${i_operand}_tile_pt;
    % endfor
    % if len(input_operands)==1:
    kernel.dim_W=&dim_W;
    kernel.mem_W=${for_loop['mem_W']};
    kernel.W_pt=loop_kernel_W_tile_pt;
    % endif
    kernel.im2col=im2col;
    kernel.pwtbuf=pwtbuf;
    pi_team_offload_preset(kernel_function_${func_number}, &kernel);
    % if len(input_operands)==1:
    if(iter>0)
        memcopyresult_${func_number}(&dim_O,kernel_O_prev_pt,kernel_O_prev_tile_pt,shared_l1,dram);
    kernel_O_prev_tile_pt=loop_${sw_for_loops[last_movements['O']]["fullname"]}_O_tile_pt+kernel_offset_O;
    kernel_O_prev_pt=loop_kernel_O_tile_pt;
    % else:
    pi_team_offload_wait();
    kernel_O_prev_tile_pt=loop_${sw_for_loops[last_movements['O']]["fullname"]}_O_tile_pt+kernel_offset_O;
    kernel_O_prev_pt=loop_kernel_O_tile_pt;
    memcopyresult_${func_number}(&dim_O,kernel_O_prev_pt,kernel_O_prev_tile_pt,shared_l1,dram);
    % endif
    iter++;
    % for kernel_operand in operands:
    % for rel_loop_dim in ordered_relevant_loops[kernel_operand]:
    tile_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_id_absolute-=tile_${rel_loop_dim if kernel_operand not in input_operands else inp_dim_mapping[rel_loop_dim]}_${kernel_operand}_kernel_id_relative;
    % endfor
    % endfor
    % endif
    % if idx!=len(sw_for_loops)-1:
    for(${for_loop['fullname']}=0;${for_loop['fullname']}<${for_loop['size']};${for_loop['fullname']}++){
    % endif
    % endfor
    ## close braces and save output
    % for idx in reversed(range(len(sw_for_loops))):
    % if idx!=len(sw_for_loops)-1:
    }
    % endif
    ${sw_for_loops[idx]['fullname']}=0;
    % for mem_transfer_layer in for_loop_mem_transfers[sw_for_loops[idx]['fullname']]:
    % for rel_loop_dim in ordered_relevant_loops[mem_transfer_layer]:
    tile_${rel_loop_dim if mem_transfer_layer not in input_operands else inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_id_absolute-=tile_${rel_loop_dim if mem_transfer_layer not in input_operands else inp_dim_mapping[rel_loop_dim]}_${mem_transfer_layer}_${idx}_id_relative;
    % endfor
    % endfor
    % endfor
    % if len(input_operands)==1:
    pi_team_offload_wait();
    memcopyresult_${func_number}(&dim_O,kernel_O_prev_pt,kernel_O_prev_tile_pt,shared_l1,dram);
    dma_transfer_wait(transfer);
    % endif
    release_mem();
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
    ${init_plaftform_api}(${func_name}_inner, args);
    return 0;
}