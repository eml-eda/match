#ifndef _CLUSTER_MEM_H
#define _CLUSTER_MEM_H
#include <match_target_params.h>
#include <match_dimensions.h>
#include <match_tile_indexes.h>
#include <cluster_comp.h>
#include "dory_dma.h"
#include <gap9_cluster.h>

void cluster_init_platform(void (*inner_function)(unsigned int* args_inner_function),unsigned int* args);

void cluster_startup_memory_and_set_pattern(int* first_op_sizes,unsigned char first_op_db,dimension_I* dim_I,
                                int* second_op_sizes,unsigned char second_op_db,dimension_W* dim_W,
                                int* third_op_sizes,unsigned char third_op_db,dimension_O* dim_O,
                                int* paddings,int* strides,unsigned int pattern);

void cluster_shutdown_mem();

unsigned int cluster_mem_transfer_O(dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem);

void cluster_copy_out_curr_computation(dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);

void cluster_copy_out_prev_computation(dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);

unsigned int cluster_mem_transfer_I(dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int cluster_mem_transfer_X(dimension_X* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int cluster_mem_transfer_Y(dimension_Y* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int cluster_mem_transfer_W(dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem);

void cluster_wait_any_transfer();

void cluster_wait_prev_computation();

void cluster_wait_curr_computation();

unsigned int cluster_pointer_offset_O(tile_indexes_O* tile_idxs,int mem_level);

unsigned int cluster_pointer_offset_I(tile_indexes_I* tile_idxs,int mem_level);

unsigned int cluster_pointer_offset_X(tile_indexes_X* tile_idxs,int mem_level);

unsigned int cluster_pointer_offset_Y(tile_indexes_Y* tile_idxs,int mem_level);

unsigned int cluster_pointer_offset_W(tile_indexes_W* tile_idxs,int mem_level);

void cluster_pattern_constant_loading(unsigned int iter,unsigned char* weights_and_constant_buf,cluster_kernel* kernel);

unsigned int get_im2col_pt();

unsigned int get_pwtbuf();
#endif