#ifndef _NE16_MEM_H
#define _NE16_MEM_H
#include <match_target_params.h>
#include <match_dimensions.h>
#include <match_tile_indexes.h>
#include <match_kernel.h>
#include <ne16_params.h>
#include <cluster_mem.h>
#include "dory_dma.h"

void ne16_init_platform(void (inner_function)(unsigned int* args_inner_function),unsigned int* args,common_kernel* common_kernel);

void ne16_set_task_id(common_kernel* kernel);

void ne16_startup_memory(common_kernel* common_kernel,int* first_op_sizes,unsigned char first_op_db,dimension_I* dim_I,
                                int* second_op_sizes,unsigned char second_op_db,dimension_W* dim_W,
                                int* third_op_sizes,unsigned char third_op_db,dimension_O* dim_O,
                                int* paddings,int* strides);

void ne16_shutdown_mem(common_kernel* common_kernel);

unsigned int ne16_mem_transfer_O(common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem);

void ne16_copy_out_curr_computation(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);

unsigned int ne16_mem_transfer_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int ne16_mem_transfer_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem);

void ne16_wait_input_transfers(common_kernel* common_kernel);

void ne16_wait_output_transfers(common_kernel* common_kernel);

void ne16_wait_curr_computation(common_kernel* common_kernel);

void ne16_pattern_constant_loading(match_kernel* kernel,unsigned int iter,tile_indexes_W* abs_tile_idx,
                                    tile_indexes_W* relative_tile_idx,void* weights_and_constant_buf);

static inline void execute_wait(ne16_task_t *task) {
  while (!ne16_nnx_resolve_check(match_ne16_get_nnx_dev(), task))
    ;
}

#endif