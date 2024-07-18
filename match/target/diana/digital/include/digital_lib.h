#ifndef _DIGITAL_LIB_H
#define _DIGITAL_LIB_H
#define dram_SIZE 512*1024
#define act_mem_SIZE 24*1024
#define weight_mem_SIZE 64*1024
#define TOTAL_MEM_SIZE (dram_SIZE+act_mem_SIZE+weight_mem_SIZE)
#define weight_mem_OFF (act_mem_SIZE)
#define act_mem_OFF 0
#define dram_OFF 0


#include <match_target_params.h>
#include <match_dimensions.h>
#include <match_tile_indexes.h>
#include <dory.h>
#include <archi/hwme/hwme_v1.h>
#include <hal/hwme/hwme_v1.h>
#include <hal/pulp.h>
#include <pulp.h>
typedef enum{
    operator_O,
    operator_I,
    operator_X,
    operator_Y,
    operator_W
}operatorEnum;
void* diana_digital_kernel_wrapper(match_kernel* kernel);
void* digital_memalloc(int size,int memorylevel,int operator);
void digital_memcopyresult(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem);
void* digital_memcopy_O(common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem);
void* digital_memcopy_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem);
void* digital_memcopy_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem);
void* digital_memcopy_X(common_kernel* common_kernel,dimension_X* dim,unsigned int ext_pt,int ext_mem,int int_mem);
void* digital_memcopy_Y(common_kernel* common_kernel,dimension_Y* dim,unsigned int ext_pt,int ext_mem,int int_mem);
unsigned int digital_W_pointer_offset(common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level);

void digital_set_channel(common_kernel* common_kernel,int* first_op_sizes,unsigned char first_op_db,dimension_I* dim_I,
                                int* second_op_sizes,unsigned char second_op_db,dimension_W* dim_W,
                                int* third_op_sizes,unsigned char third_op_db,dimension_O* dim_O,
                                int* paddings,int* strides);
void digital_free_channel();

#endif