#include <match_dimensions.h>

typedef struct common_kernel_t {
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
  unsigned int pad_ix_x;
  unsigned int pad_ix_y;
  unsigned int pad_iy_x;
  unsigned int pad_iy_y;
  unsigned int dilatation_x;
  unsigned int dilatation_y;
  // strides
  unsigned int stride_x;
  unsigned int stride_y;
  // fused layers pts
  void* bias_pt;
  void* batchnorm_mul;
  void* batchnorm_add;
  // output dim
  dimension_O* dim_O;
  int mem_O;
  void* O_pt;
  // 1 input
  dimension_I* dim_I;
  int mem_I;
  void* I_pt;
  // 2 inputs
  dimension_X* dim_X;
  int mem_X;
  void* X_pt;
  dimension_Y* dim_Y;
  int mem_Y;
  void* Y_pt;
  // weights
  dimension_W* dim_W;
  int mem_W;
  void* W_pt;
} common_kernel;

typedef struct match_kernel_t
{
  common_kernel* common_kernel;
}match_kernel;


void init_common_kernel_params(common_kernel* kernel);