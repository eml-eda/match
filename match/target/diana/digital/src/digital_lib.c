#include <digital_lib.h>

void* memory=0x0;
int last_dram=0;
int last_act=0;
int act_O=-1;
int act_I=-1;
int act_X=-1;
int act_Y=-1;
uint32_t dory_dma_channel=0;
/*
//inline 
unsigned int get_W_address_${func_number}(int tile_K_id_relative,int tile_K_id_absolute,int tile_C_id_relative,int tile_C_id_absolute,
                        int tile_FY_id_relative,int tile_FY_id_absolute,int tile_FX_id_relative,int tile_FX_id_absolute,dimensionW_${func_number}* dim,int mem_level){
    return (tile_K_id_relative*dim->accel_dim.size_C[mem_level]*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]+
            tile_C_id_relative*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]+(((int)(tile_K_id_relative/${16 if layer_name!='depthwise_conv_2d' else 4}))*${64/(attrs[f'{layer_name}_prec']['W']/8) if layer_name!='dense' else 256}))${'*'+str(attrs[f'{layer_name}_prec']['W']//8) if attrs[f'{layer_name}_prec']['W']>8 else ''};
}

//inline 
unsigned int get_W_address_kernel_${func_number}(int tile_K_id_relative,int tile_K_id_absolute,int tile_C_id_relative,int tile_C_id_absolute,
                        int tile_FY_id_relative,int tile_FY_id_absolute,int tile_FX_id_relative,int tile_FX_id_absolute,dimensionW_${func_number}* dim,Kernel_parameters_${func_number}* kernel,int mem_level){
    return (tile_K_id_relative*dim->accel_dim.size_C[mem_level]*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]+
            tile_C_id_relative*dim->accel_dim.size_FY[mem_level]*dim->accel_dim.size_FX[mem_level]+(((int)(tile_K_id_relative/${16 if layer_name!='depthwise_conv_2d' else 4}))*${64/(attrs[f'{layer_name}_prec']['W']/8) if layer_name!='dense' else 256}))${'*'+str(attrs[f'{layer_name}_prec']['W']//8) if attrs[f'{layer_name}_prec']['W']>8 else ''};
}
% endif

% for input_operand in input_operands:
void add_sizes_to_kernel_${input_operand}_${func_number}(int tile_C,int tile_IY,int tile_IX,Kernel_parameters_${func_number}* kernel){
    kernel->c=tile_C;
    kernel->cy=tile_IY;
    kernel->cx=tile_IX;
}
//inline 
unsigned int get_${input_operand}_address_${func_number}(int tile_C_id_relative,int tile_C_id_absolute,int tile_IY_id_relative,int tile_IY_id_absolute,
        % if len(padded_dims)>0:
    int stride_Y,
    % endif
                        int tile_IX_id_relative,int tile_IX_id_absolute,
    % if len(padded_dims)>0:
    int stride_X,
    % endif
    dimension${input_operand}_${func_number}* dim,int mem_level){
    return (tile_C_id_relative*dim->accel_dim.size_IY[mem_level]*dim->accel_dim.size_IX[mem_level]+(tile_IY_id_relative${'*stride_Y' if len(padded_dims)>0 else ''})*dim->accel_dim.size_IX[mem_level]+(tile_IX_id_relative${'*stride_X' if len(padded_dims)>0 else ''}));
}

//inline 
unsigned int get_${input_operand}_address_kernel_${func_number}(int tile_C_id_relative,int tile_C_id_absolute,int tile_IY_id_relative,int tile_IY_id_absolute,
    % if len(padded_dims)>0:
    int stride_Y,
    % endif
                        int tile_IX_id_relative,int tile_IX_id_absolute,
    % if len(padded_dims)>0:
    int stride_X,
    % endif
    dimension${input_operand}_${func_number}* dim,Kernel_parameters_${func_number}* kernel,int mem_level){
    return (tile_C_id_relative*dim->accel_dim.size_IY[mem_level]*dim->accel_dim.size_IX[mem_level]+(tile_IY_id_relative${'*stride_Y' if len(padded_dims)>0 else ''})*dim->accel_dim.size_IX[mem_level]+(tile_IX_id_relative${'*stride_X' if len(padded_dims)>0 else ''}));
}
% endfor

//inline 
unsigned int get_O_address_${func_number}(int tile_K_id_relative,int tile_K_id_absolute,int tile_OY_id_relative,int tile_OY_id_absolute,
                        int tile_OX_id_relative,int tile_OX_id_absolute,dimensionO_${func_number}* dim,int mem_level){
    return (tile_K_id_relative*dim->accel_dim.size_OY[mem_level]*dim->accel_dim.size_OX[mem_level]+
    tile_OY_id_relative*dim->accel_dim.size_OX[mem_level]+
    tile_OX_id_relative);
}

//inline 
unsigned int get_O_address_kernel_${func_number}(int tile_K_id_relative,int tile_K_id_absolute,int tile_OY_id_relative,int tile_OY_id_absolute,
                        int tile_OX_id_relative,int tile_OX_id_absolute,dimensionO_${func_number}* dim,Kernel_parameters_${func_number}* kernel,int mem_level){
    return (tile_K_id_relative*dim->accel_dim.size_OY[mem_level]*dim->accel_dim.size_OX[mem_level]+tile_OY_id_relative*dim->accel_dim.size_OX[mem_level]+tile_OX_id_relative);
}

*/

void* diana_digital_kernel_wrapper(match_kernel* kernel){
    #ifdef MATCH_PROFILE_KERN_TOTAL
    start_g_perf_counter();
    #endif
    Layer_parameters diana_kernel;
    int i_channels=kernel->common_kernel->c_i>kernel->common_kernel->c_w?kernel->common_kernel->c_w:kernel->common_kernel->c_i;
    int o_channels=kernel->common_kernel->k_o;
    int i_width=kernel->common_kernel->ix_i
    +kernel->common_kernel->dim_I->overlap_IX_x+kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->iy_i
    +kernel->common_kernel->dim_I->overlap_IY_x+kernel->common_kernel->dim_I->overlap_IY_y
    -kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_width=kernel->common_kernel->ox;
    int o_height=kernel->common_kernel->oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=kernel->common_kernel->batchnorm_add!=0x0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    diana_kernel.c=kernel->common_kernel->specific_pattern!=depthwise_conv_2d? i_channels:o_channels;
    diana_kernel.k=o_channels;
    diana_kernel.cx=i_width;
    diana_kernel.cy=i_height;
    diana_kernel.fx=w_y_width;
    diana_kernel.fy=w_y_height;
    diana_kernel.ox=o_width;
    diana_kernel.oy=o_height;
    diana_kernel.activation_function=act;
    diana_kernel.output_shift=kernel->common_kernel->right_shift;
    diana_kernel.dilation=1;
    diana_kernel.stride=kernel->common_kernel->stride_x;
    diana_kernel.ox_unroll=0;
    diana_kernel.padding=(p_right>p_left? p_right : p_left)+(p_top<<8)+(p_bottom<<12);//+(p_left<<4);
    #ifdef MATCH_PROFILE_KERN_CALL
    start_g_perf_counter();
    #endif
    global_sync_digital();
    switch(kernel->common_kernel->specific_pattern){
        case conv_2d:
            digital_conv_2d(0x0,kernel->common_kernel->I_pt,kernel->common_kernel->W_pt,0x0,kernel->common_kernel->O_pt,&diana_kernel);
            break;
        case dense:
            digital_fully_connected(0x0,kernel->common_kernel->I_pt,kernel->common_kernel->W_pt,0x0,kernel->common_kernel->O_pt,&diana_kernel);
            break;
        case depthwise_conv_2d:
            digital_depthwise_conv_2d(0x0,kernel->common_kernel->I_pt,kernel->common_kernel->W_pt,0x0,kernel->common_kernel->O_pt,&diana_kernel);
            break;
        case element_wise_sum:
            digital_element_wise_sum(0x0,kernel->common_kernel->X_pt,0x0,kernel->common_kernel->Y_pt,kernel->common_kernel->O_pt,&diana_kernel);
            break;
        default:
            break;
    }
    
    global_sync_digital();
    #if defined(MATCH_PROFILE_KERN_TOTAL) || defined(MATCH_PROFILE_KERN_CALL) || defined(MATCH_PROFILE_KERN_JOB)
    stop_g_perf_counter();
    #endif
}


void* digital_memalloc(int size,int memorylevel,int operator){
    void* dst=memory;
    switch(memorylevel){
        case dram:
            if(last_dram+size>=dram_SIZE) last_dram=0;
            dst+=dram_OFF+last_dram;
            last_dram+=size;
            break;
        case act_mem:
            switch (operator)
            {
                case operator_O:
                    if(act_O<0){
                        dst+=act_mem_OFF+last_act;
                        act_O=last_act;
                        last_act+=size;
                    }
                    else dst+=act_mem_OFF+act_O;
                    break;
                case operator_I:
                    if(act_I<0){
                        dst+=act_mem_OFF+last_act;
                        act_I=last_act;
                        last_act+=size;
                    }
                    else dst+=act_mem_OFF+act_I;
                    break;
                case operator_X:
                    if(act_X<0){
                        dst+=act_mem_OFF+last_act;
                        act_X=last_act;
                        last_act+=size;
                    }
                    else dst+=act_mem_OFF+act_X;
                    break;
                case operator_Y:
                    if(act_Y<0){
                        dst+=act_mem_OFF+last_act;
                        act_Y=last_act;
                        last_act+=size;
                    }
                    else dst+=act_mem_OFF+act_Y;
                    break;
                default:
                    break;
            }
        case weight_mem:
            dst+= weight_mem_OFF;
            break;
        default:
            break;
    }
    return dst;
}

void* digital_memcopy_O(common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    int size=dim->size_K[int_mem]*dim->size_OY[int_mem]*dim->size_OX[int_mem];
    return digital_memalloc(size,int_mem,operator_O);
}

void digital_memcopyresult(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    unsigned char* dst=(unsigned char*) ext_pt;
    unsigned char* src=(unsigned char*) int_pt;
    if(dim->size_OY[int_mem]==dim->size_OY[ext_mem] && dim->size_OX[int_mem]==dim->size_OX[ext_mem])
    {
        unsigned int* dstdig=dst;
        unsigned int srcdig=src;
        unsigned int len=dim->size_K[int_mem]*dim->size_OY[int_mem]*dim->size_OX[int_mem];
        memcpy_dig(dstdig, srcdig, len, 1, 1);
        global_sync_digital();
        return;
    }
    else{
        if(dim->size_OX[ext_mem]==dim->size_OX[int_mem]){
            for(int K_src=0;K_src<dim->size_K[int_mem];K_src++){
                unsigned int* dstdig=dst+ K_src* dim->size_OY[ext_mem]* dim->size_OX[ext_mem];
                unsigned int srcdig=src+ K_src* dim->size_OY[int_mem]* dim->size_OX[int_mem];
                unsigned int len=dim->size_OY[int_mem]*dim->size_OX[int_mem];
                memcpy_dig(dstdig, srcdig, len, 1, 1);
                global_sync_digital();
            }
            return;
        }
        else{
            for(int K_src=0;K_src<dim->size_K[int_mem];K_src++){
                for(int OY_src=0;OY_src<dim->size_OY[int_mem];OY_src++){
                    unsigned int* dstdig=dst+ K_src* dim->size_OY[ext_mem]* dim->size_OX[ext_mem]+ OY_src* dim->size_OX[ext_mem];
                    unsigned int srcdig=src+ K_src* dim->size_OY[int_mem]* dim->size_OX[int_mem]+ OY_src* dim->size_OX[int_mem];
                    unsigned int len=dim->size_OX[int_mem];
                    memcpy_dig(dstdig, srcdig, len, 1, 1);
                    global_sync_digital();
                }
            }
            return;
        }
    }
}

void* digital_memcopy_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    return ext_pt;
}

void* digital_memcopy_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    int size=dim->size_C[int_mem]*dim->size_IY[int_mem]*dim->size_IX[int_mem];
    unsigned char* dst=(unsigned char*) digital_memalloc(size,int_mem,operator_I);
    unsigned char* src=(unsigned char*) ext_pt;
    unsigned int inner_ix=dim->size_IX[int_mem]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y;
    unsigned int inner_iy=dim->size_IY[ext_mem]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y;
    if(dim->size_IY[int_mem]==dim->size_IY[ext_mem] && dim->size_IX[int_mem]==dim->size_IX[ext_mem])
    {
        unsigned int dstdig=dst;
        unsigned int* srcdig=src;
        unsigned int len=dim->size_C[int_mem]*inner_iy*inner_ix;
        memcpy_dig(srcdig, dstdig, len, 0, 1);
        global_sync_digital();
        return dst;
    }
    else{
        if(dim->size_IX[int_mem]==dim->size_IX[ext_mem]){
            for(int C_dst=0;C_dst<dim->size_C[int_mem];C_dst++){
                unsigned int dstdig=dst+ C_dst* inner_iy* inner_ix;
                unsigned int* srcdig=src+ C_dst* dim->size_IY[ext_mem]* dim->size_IX[ext_mem];
                unsigned int len=inner_iy*inner_ix;
                memcpy_dig(srcdig, dstdig, len, 0, 1);
                global_sync_digital();
            }
            return dst;
        }
        else{
            for(int C_dst=0;C_dst<dim->size_C[int_mem];C_dst++){
                for(int IY_dst=0;IY_dst<dim->size_IY[int_mem];IY_dst++){
                    unsigned int dstdig=dst+ (C_dst* inner_iy* inner_ix)+ (IY_dst* inner_ix);
                    unsigned int* srcdig=src+ C_dst* dim->size_IY[ext_mem]* dim->size_IX[ext_mem]+ IY_dst* dim->size_IX[ext_mem];
                    memcpy_dig(srcdig, dstdig, inner_ix, 0, 1);
                    global_sync_digital();
                }
            }
            return dst;
        }
    }
    return dst;
}

void* digital_memcopy_X(common_kernel* common_kernel,dimension_X* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    int size=dim->size_C[int_mem]*dim->size_IY[int_mem]*dim->size_IX[int_mem];
    unsigned char* dst=(unsigned char*) digital_memalloc(size,int_mem,operator_X);
    unsigned char* src=(unsigned char*) ext_pt;
    unsigned int inner_ix=dim->size_IX[int_mem]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y;
    unsigned int inner_iy=dim->size_IY[ext_mem]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y;
    if(dim->size_IY[int_mem]==dim->size_IY[ext_mem] && dim->size_IX[int_mem]==dim->size_IX[ext_mem])
    {
        unsigned int dstdig=dst;
        unsigned int* srcdig=src;
        unsigned int len=dim->size_C[int_mem]*inner_iy*inner_ix;
        memcpy_dig(srcdig, dstdig, len, 0, 1);
        global_sync_digital();
        return dst;
    }
    else{
        if(dim->size_IX[int_mem]==dim->size_IX[ext_mem]){
            for(int C_dst=0;C_dst<dim->size_C[int_mem];C_dst++){
                unsigned int dstdig=dst+ C_dst* inner_iy* inner_ix;
                unsigned int* srcdig=src+ C_dst* dim->size_IY[ext_mem]* dim->size_IX[ext_mem];
                unsigned int len=inner_iy*inner_ix;
                memcpy_dig(srcdig, dstdig, len, 0, 1);
                global_sync_digital();
            }
            return dst;
        }
        else{
            for(int C_dst=0;C_dst<dim->size_C[int_mem];C_dst++){
                for(int IY_dst=0;IY_dst<dim->size_IY[int_mem];IY_dst++){
                    unsigned int dstdig=dst+ (C_dst* inner_iy* inner_ix)+ (IY_dst* inner_ix);
                    unsigned int* srcdig=src+ C_dst* dim->size_IY[ext_mem]* dim->size_IX[ext_mem]+ IY_dst* dim->size_IX[ext_mem];
                    memcpy_dig(srcdig, dstdig, inner_ix, 0, 1);
                    global_sync_digital();
                }
            }
            return dst;
        }
    }
    return dst;
}

void* digital_memcopy_Y(common_kernel* common_kernel,dimension_Y* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    int size=dim->size_C[int_mem]*dim->size_IY[int_mem]*dim->size_IX[int_mem];
    unsigned char* dst=(unsigned char*) digital_memalloc(size,int_mem,operator_Y);
    unsigned char* src=(unsigned char*) ext_pt;
    unsigned int inner_ix=dim->size_IX[int_mem]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y;
    unsigned int inner_iy=dim->size_IY[ext_mem]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y;
    if(dim->size_IY[int_mem]==dim->size_IY[ext_mem] && dim->size_IX[int_mem]==dim->size_IX[ext_mem])
    {
        unsigned int dstdig=dst;
        unsigned int* srcdig=src;
        unsigned int len=dim->size_C[int_mem]*inner_iy*inner_ix;
        memcpy_dig(srcdig, dstdig, len, 0, 1);
        global_sync_digital();
        return dst;
    }
    else{
        if(dim->size_IX[int_mem]==dim->size_IX[ext_mem]){
            for(int C_dst=0;C_dst<dim->size_C[int_mem];C_dst++){
                unsigned int dstdig=dst+ C_dst* inner_iy* inner_ix;
                unsigned int* srcdig=src+ C_dst* dim->size_IY[ext_mem]* dim->size_IX[ext_mem];
                unsigned int len=inner_iy*inner_ix;
                memcpy_dig(srcdig, dstdig, len, 0, 1);
                global_sync_digital();
            }
            return dst;
        }
        else{
            for(int C_dst=0;C_dst<dim->size_C[int_mem];C_dst++){
                for(int IY_dst=0;IY_dst<dim->size_IY[int_mem];IY_dst++){
                    unsigned int dstdig=dst+ (C_dst* inner_iy* inner_ix)+ (IY_dst* inner_ix);
                    unsigned int* srcdig=src+ C_dst* dim->size_IY[ext_mem]* dim->size_IX[ext_mem]+ IY_dst* dim->size_IX[ext_mem];
                    memcpy_dig(srcdig, dstdig, inner_ix, 0, 1);
                    global_sync_digital();
                }
            }
            return dst;
        }
    }
    return dst;
}


unsigned int digital_W_pointer_offset(common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level){
    return (tile_idxs->tile_K*common_kernel->dim_W->size_C[memory_level]*common_kernel->dim_W->size_FY[memory_level]*common_kernel->dim_W->size_FX[memory_level]+
            tile_idxs->tile_C*common_kernel->dim_W->size_FY[memory_level]*common_kernel->dim_W->size_FX[memory_level]+
            (((int)(tile_idxs->tile_K/(common_kernel->specific_pattern!=depthwise_conv_2d?16:4)))*(common_kernel->specific_pattern!=dense?64/(common_kernel->prec_W/8):256)))*common_kernel->prec_W/8;
}

void digital_set_channel(common_kernel* common_kernel,int* first_op_sizes,unsigned char first_op_db,dimension_I* dim_I,
                                int* second_op_sizes,unsigned char second_op_db,dimension_W* dim_W,
                                int* third_op_sizes,unsigned char third_op_db,dimension_O* dim_O,
                                int* paddings,int* strides){
    dory_dma_channel = dory_dma_allocate();
}

void digital_free_channel(){
    dory_dma_deallocate(dory_dma_channel);
}