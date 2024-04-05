#ifndef _CLUSTER_MEM_H
#define _CLUSTER_MEM_H
#include <match_target_params.h>
#include <match_dimensions.h>
#include <match_tile_indexes.h>
#include <cluster_comp.h>
#include "dory_dma.h"
#include <gap9_cluster.h>

void cluster_init_l1_memory();

unsigned int cluster_get_l1_memory_addr();


void cluster_init_platform(void (inner_function)(unsigned int* args_inner_function),unsigned int* args,common_kernel* common_kernel);

void cluster_init_l1_memory();

void cluster_startup_memory(common_kernel* common_kernel,int* first_op_sizes,unsigned char first_op_db,dimension_I* dim_I,
                                int* second_op_sizes,unsigned char second_op_db,dimension_W* dim_W,
                                int* third_op_sizes,unsigned char third_op_db,dimension_O* dim_O,
                                int* paddings,int* strides);

void cluster_shutdown_mem(common_kernel* common_kernel);

unsigned int cluster_mem_transfer_O(common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem);

void cluster_copy_out_curr_computation(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);

void cluster_copy_out_prev_computation(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);

unsigned int cluster_mem_transfer_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int cluster_mem_transfer_X(common_kernel* common_kernel,dimension_X* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int cluster_mem_transfer_Y(common_kernel* common_kernel,dimension_Y* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int cluster_mem_transfer_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem);

void cluster_wait_any_transfer(common_kernel* common_kernel);

void cluster_wait_prev_computation(common_kernel* common_kernel);

void cluster_wait_curr_computation(common_kernel* common_kernel);

void cluster_pattern_constant_loading(cluster_kernel* kernel,unsigned int iter,tile_indexes_W* abs_tile_idx,
                                    tile_indexes_W* relative_tile_idx,void* weights_and_constant_buf);

unsigned int get_im2col_pt();

unsigned int get_pwtbuf();

unsigned int cluster_get_l1_memory_addr();
#endif