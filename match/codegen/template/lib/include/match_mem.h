#ifndef _MATCH_MEM_H
#define _MATCH_MEM_H
#include <match_tile_indexes.h>
#include <match_kernel.h>

void match_init_platform(void (*inner_function)(unsigned int* args_inner_function),unsigned int* args,common_kernel* comm_kernel);

void match_task_id(common_kernel* comm_kernel);

void match_copy_out_prev_computation(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);

unsigned int match_pointer_offset_NCHW_O(common_kernel* common_kernel,tile_indexes_O* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NCHW_I(common_kernel* common_kernel,tile_indexes_I* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NCHW_X(common_kernel* common_kernel,tile_indexes_X* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NCHW_Y(common_kernel* common_kernel,tile_indexes_Y* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NCHW_W(common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_O(common_kernel* common_kernel,tile_indexes_O* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_I(common_kernel* common_kernel,tile_indexes_I* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_X(common_kernel* common_kernel,tile_indexes_X* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_Y(common_kernel* common_kernel,tile_indexes_Y* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_W(common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level);

void match_pattern_constants_loading(match_kernel* kernel,unsigned int iter,tile_indexes_W* abs_tile_idx,
                                    tile_indexes_W* relative_tile_idx,void* weights_and_constant_buf);

unsigned int match_mem_transfer_O(common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int match_mem_transfer_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int match_mem_transfer_X(common_kernel* common_kernel,dimension_X* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int match_mem_transfer_Y(common_kernel* common_kernel,dimension_Y* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int match_mem_transfer_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem);

#endif