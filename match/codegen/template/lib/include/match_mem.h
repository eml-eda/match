#ifndef _MATCH_MEM_H
#define _MATCH_MEM_H
#include <match_tile_indexes.h>
#include <match_kernel.h>

void match_init_platform(void (*inner_function)(unsigned int* args_inner_function),unsigned int* args);

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

void match_pattern_constants_loading(match_kernel* kernel,unsigned int iter,unsigned char* weights_and_constant_buf);

#endif