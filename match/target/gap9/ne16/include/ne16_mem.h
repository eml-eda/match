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

void ne16_startup_memory(unsigned int task_id,common_kernel* common_kernel,int* first_op_sizes,unsigned char first_op_db,dimension_I* dim_I,
                                int* second_op_sizes,unsigned char second_op_db,dimension_W* dim_W,
                                int* third_op_sizes,unsigned char third_op_db,dimension_O* dim_O,
                                int* paddings,int* strides);

void ne16_shutdown_mem(unsigned int task_id,common_kernel* common_kernel);

unsigned int ne16_mem_transfer_O(unsigned int task_id,common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem);

void ne16_copy_out_curr_computation(unsigned int task_id,common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);

unsigned int ne16_mem_transfer_I(unsigned int task_id,common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem);

unsigned int ne16_mem_transfer_W(unsigned int task_id,common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem);

void ne16_wait_input_transfers(unsigned int task_id,common_kernel* common_kernel);

void ne16_wait_output_transfers(unsigned int task_id,common_kernel* common_kernel);

void ne16_wait_curr_computation(unsigned int task_id,common_kernel* common_kernel);

void ne16_pattern_constant_loading(unsigned int task_id,match_kernel* kernel,unsigned int iter,void* weights_and_constant_buf);

static inline void execute_wait(ne16_task_t *task) {
#if 1
//__PLATFORM__ == ARCHI_PLATFORM_GVSOC && defined GAP_SDK
  // Temporary hack because the gvsoc model of ne16 in gap_sdk
  // has a broken running_id.
  while(!ne16_task_queue_empty(match_ne16_get_nnx_dev()))
    ;
#else
  while(!nnx_resolve_check(task))
    ;
#endif
}

#endif