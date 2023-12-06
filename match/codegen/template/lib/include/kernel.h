typedef struct Common_kernel_parameters_t {
  // dimensions
  int c_w;
  int c_i;
  int c_x;
  int c_y:
  int k_o;
  int k_w;
  int ix_i;
  int iy_i;
  int ix_x;
  int iy_x;
  int ix_y;
  int iy_y;
  int fx;
  int fy;
  int ox;
  int oy;
  int activation_function;
  int right_shift;
  // pads
  unsigned int pad_ix_x;
  unsigned int pad_ix_y;
  unsigned int pad_iy_x;
  unsigned int pad_iy_y;
  int dilation;
  // strides
  unsigned int stride_x;
  unsigned int stride_y;
  // fused layers pts
  void* bias_pt;
  void* batchnorm_mul;
  void* batchnorm_add;
  // output dim
  dimensionO* dim_O;
  int mem_O;
  void* O_pt;
  // 1 input
  dimensionI* dim_I;
  int mem_I;
  void* I_pt;
  // 2 inputs
  dimensionI* dim_X;
  int mem_X;
  void* X_pt;
  dimensionY* dim_Y;
  int mem_Y;
  void* Y_pt;
  // weights
  dimensionW* dim_W;
  int mem_W;
  void* W_pt;
} Common_kernel_parameters;