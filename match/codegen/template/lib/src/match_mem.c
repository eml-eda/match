#include <match_mem.h>

void match_init_platform(void (*inner_function)(unsigned int* args_inner_function),unsigned int* args){
    // API to offload execution to a controller
    inner_function(args);
}

void match_task_id(common_kernel* comm_kernel){
    return;
}

void match_copy_out_prev_computation(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    return;
}

unsigned int match_pointer_offset_NCHW_O(common_kernel* common_kernel,tile_indexes_O* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_O/8;
    return (tile_idxs->tile_K*common_kernel->dim_O->size_OY[memory_level]*common_kernel->dim_O->size_OX[memory_level]+
            tile_idxs->tile_OY*common_kernel->dim_O->size_OX[memory_level]+tile_idxs->tile_OX)*bytes_per_obj;
}

unsigned int match_pointer_offset_NCHW_I(common_kernel* common_kernel,tile_indexes_I* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_I/8;
    return (tile_idxs->tile_C*common_kernel->dim_I->size_IY[memory_level]*common_kernel->dim_I->size_IX[memory_level]+
            (tile_idxs->tile_IY-common_kernel->dim_I->overlap_IY_x+common_kernel->dim_I->pad_IY_x)*
            common_kernel->dim_I->size_IX[memory_level]+
            (tile_idxs->tile_IX-common_kernel->dim_I->overlap_IX_x+common_kernel->dim_I->pad_IX_x))*bytes_per_obj;
}

unsigned int match_pointer_offset_NCHW_X(common_kernel* common_kernel,tile_indexes_X* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_X/8;
    return (tile_idxs->tile_C*common_kernel->dim_X->size_IY[memory_level]*common_kernel->dim_X->size_IX[memory_level]+
            tile_idxs->tile_IY*common_kernel->dim_X->size_IX[memory_level]+tile_idxs->tile_IX)*bytes_per_obj;
}

unsigned int match_pointer_offset_NCHW_Y(common_kernel* common_kernel,tile_indexes_Y* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_I/8;
    return (tile_idxs->tile_C*common_kernel->dim_Y->size_IY[memory_level]*common_kernel->dim_Y->size_IX[memory_level]+
            tile_idxs->tile_IY*common_kernel->dim_Y->size_IX[memory_level]+tile_idxs->tile_IX)*bytes_per_obj;
}

unsigned int match_pointer_offset_NCHW_W(common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_W/8;
    return (tile_idxs->tile_K*common_kernel->dim_W->size_C[memory_level]*
            common_kernel->dim_W->size_FY[memory_level]*common_kernel->dim_W->size_FX[memory_level]+
            tile_idxs->tile_C*common_kernel->dim_W->size_FY[memory_level]*common_kernel->dim_W->size_FX[memory_level]+
            tile_idxs->tile_FY*common_kernel->dim_W->size_FX[memory_level]+tile_idxs->tile_FX)*bytes_per_obj;
}

unsigned int match_pointer_offset_NHWC_O(common_kernel* common_kernel,tile_indexes_O* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_O/8;
    return (tile_idxs->tile_OY*common_kernel->dim_O->size_K[memory_level]*common_kernel->dim_O->size_OX[memory_level]+
            tile_idxs->tile_OX*common_kernel->dim_O->size_K[memory_level]+tile_idxs->tile_K)*bytes_per_obj;
}

unsigned int match_pointer_offset_NHWC_I(common_kernel* common_kernel,tile_indexes_I* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_I/8;
    return ((tile_idxs->tile_IY-common_kernel->dim_I->overlap_IY_x+common_kernel->dim_I->pad_IY_x)*
            common_kernel->dim_I->size_C[memory_level]*common_kernel->dim_I->size_IX[memory_level]+
            (tile_idxs->tile_IX-common_kernel->dim_I->overlap_IX_x+common_kernel->dim_I->pad_IX_x)*
            common_kernel->dim_I->size_C[memory_level]+tile_idxs->tile_C)*bytes_per_obj;
}

unsigned int match_pointer_offset_NHWC_X(common_kernel* common_kernel,tile_indexes_X* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_X/8;
    return (tile_idxs->tile_IY*common_kernel->dim_X->size_C[memory_level]*common_kernel->dim_X->size_IX[memory_level]+
            tile_idxs->tile_IX*common_kernel->dim_X->size_C[memory_level]+tile_idxs->tile_C)*bytes_per_obj;
}

unsigned int match_pointer_offset_NHWC_Y(common_kernel* common_kernel,tile_indexes_Y* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_Y/8;
    return (tile_idxs->tile_IY*common_kernel->dim_Y->size_C[memory_level]*common_kernel->dim_Y->size_IX[memory_level]+
            tile_idxs->tile_IX*common_kernel->dim_Y->size_C[memory_level]+tile_idxs->tile_C)*bytes_per_obj;
}

unsigned int match_pointer_offset_NHWC_W(common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level){
    int bytes_per_obj=common_kernel->prec_W/8;
    return (tile_idxs->tile_K*common_kernel->dim_W->size_C[memory_level]*
            common_kernel->dim_W->size_FY[memory_level]*common_kernel->dim_W->size_FX[memory_level]+
            tile_idxs->tile_FY*common_kernel->dim_W->size_C[memory_level]*common_kernel->dim_W->size_FX[memory_level]+
            tile_idxs->tile_FX*common_kernel->dim_W->size_C[memory_level]+tile_idxs->tile_C)*bytes_per_obj;
}

void match_pattern_constants_loading(match_kernel* kernel,unsigned int iter,tile_indexes_W* abs_tile_idx,
                                    tile_indexes_W* relative_tile_idx,void* weights_and_constant_buf){
    return;
}


unsigned int match_mem_transfer_O(common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    return ext_pt;
}

unsigned int match_mem_transfer_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    return ext_pt;
}

unsigned int match_mem_transfer_X(common_kernel* common_kernel,dimension_X* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    return ext_pt;
}

unsigned int match_mem_transfer_Y(common_kernel* common_kernel,dimension_Y* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    return ext_pt;
}

unsigned int match_mem_transfer_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    return ext_pt;
}