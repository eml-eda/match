#ifndef _MATCH_KERNEL_H
#define _MATCH_KERNEL_H
#include <match_dimensions.h>

typedef struct common_kernel_t {
  unsigned int pattern_name;
  unsigned int specific_pattern;
  // dimensions
  unsigned int c_w;
  unsigned int c_i;
  unsigned int c_x;
  unsigned int c_y;
  unsigned int k_o;
  unsigned int k_w;
  unsigned int ix_i;
  unsigned int iy_i;
  unsigned int ix_x;
  unsigned int iy_x;
  unsigned int ix_y;
  unsigned int iy_y;
  unsigned int fx;
  unsigned int fy;
  unsigned int ox;
  unsigned int oy;
  int activation_function;
  unsigned int right_shift;
  // pads
  unsigned int pad_IX_x;
  unsigned int pad_IX_y;
  unsigned int pad_IY_x;
  unsigned int pad_IY_y;
  unsigned int dilatation_x;
  unsigned int dilatation_y;
  // strides
  unsigned int stride_x;
  unsigned int stride_y;
  // fused layers pts
  unsigned int bias_pt;
  unsigned int batchnorm_mul;
  unsigned int batchnorm_add;
  // output dim
  dimension_O* dim_O;
  int mem_O;
  unsigned int O_pt;
  unsigned int prec_O;
  // 1 input
  dimension_I* dim_I;
  int mem_I;
  unsigned int I_pt;
  unsigned int prec_I;
  // 2 inputs
  dimension_X* dim_X;
  int mem_X;
  unsigned int X_pt;
  unsigned int prec_X;
  dimension_Y* dim_Y;
  int mem_Y;
  unsigned int Y_pt;
  unsigned int prec_Y;
  // weights
  dimension_W* dim_W;
  int mem_W;
  unsigned int W_pt;
  unsigned int prec_W;
} common_kernel;

typedef struct match_kernel_t
{
  common_kernel* common_kernel;
}match_kernel;

void match_init_other_kernel_params(unsigned int task_id,match_kernel* kernel);

void init_common_kernel_params(common_kernel* kernel,unsigned int pattern_name,unsigned int specific_pattern);

void init_kernel_dimension_params_I_W_O(
    common_kernel* kernel,dimension_I* dim_I,unsigned int innermost_I_mem_level,unsigned int prec_I,
    dimension_W* dim_W,unsigned int innermost_W_mem_level,unsigned int prec_W,
    dimension_O* dim_O,unsigned int innermost_O_mem_level,unsigned int prec_O
);

void init_kernel_dimension_params_X_Y_O(
    common_kernel* kernel,dimension_X* dim_X,unsigned int innermost_X_mem_level,unsigned int prec_X,
    dimension_Y* dim_Y,unsigned int innermost_Y_mem_level,unsigned int prec_Y,
    dimension_O* dim_O,unsigned int innermost_O_mem_level,unsigned int prec_O
);

void match_innermost_computation(unsigned int task_id,match_kernel* kernel,unsigned int pattern_name);

void kernel_set_padding(common_kernel* kernel,dimension_I* dim);

#endif