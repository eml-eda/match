#ifndef _MATCH_MEM_H
#define _MATCH_MEM_H
#include <match_tile_indexes.h>
#include <match_kernel.h>

void match_init_platform(void (*inner_function)(unsigned int* args_inner_function),unsigned int* args);

unsigned int match_task_id();

void match_copy_out_prev_computation(unsigned int task_id,common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);

unsigned int match_pointer_offset_NCHW_O(unsigned int task_id,common_kernel* common_kernel,tile_indexes_O* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NCHW_I(unsigned int task_id,common_kernel* common_kernel,tile_indexes_I* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NCHW_X(unsigned int task_id,common_kernel* common_kernel,tile_indexes_X* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NCHW_Y(unsigned int task_id,common_kernel* common_kernel,tile_indexes_Y* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NCHW_W(unsigned int task_id,common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_O(unsigned int task_id,common_kernel* common_kernel,tile_indexes_O* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_I(unsigned int task_id,common_kernel* common_kernel,tile_indexes_I* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_X(unsigned int task_id,common_kernel* common_kernel,tile_indexes_X* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_Y(unsigned int task_id,common_kernel* common_kernel,tile_indexes_Y* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_NHWC_W(unsigned int task_id,common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level);

void match_pattern_constants_loading(unsigned int task_id,match_kernel* kernel,unsigned int iter,void* weights_and_constant_buf);

#endif