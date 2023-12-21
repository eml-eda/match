#ifndef _MATCH_MEM_H
#define _MATCH_MEM_H
#include <match_tile_indexes.h>
#include <match_kernel.h>

void match_init_platform(void (*inner_function)(unsigned int* args_inner_function),unsigned int* args);

unsigned int match_pointer_offset_O(tile_indexes_O* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_I(tile_indexes_I* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_X(tile_indexes_X* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_Y(tile_indexes_Y* tile_idxs,unsigned int memory_level);

unsigned int match_pointer_offset_W(tile_indexes_W* tile_idxs,unsigned int memory_level);

void match_pattern_constants_loading(unsigned int iter,match_kernel* kernel);

#endif