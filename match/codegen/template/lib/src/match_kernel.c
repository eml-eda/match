#include <match_kernel.h>

void init_common_kernel_params(common_kernel* kernel){
    // dimensions
    kernel->c_w=0;kernel->c_i=0;kernel->c_x=0;kernel->c_y=0;
    kernel->k_o=0;kernel->k_w=0;
    kernel->ix_i=0;kernel->iy_i=0;
    kernel->ix_x=0;kernel->iy_x=0;
    kernel->ix_y=0;kernel->iy_y=0;
    kernel->fx=0;
    kernel->fy=0;
    kernel->ox=0;
    kernel->oy=0;
    /*
    kernel->activation_function=0;
    kernel->right_shift=0;
    */
    // pads
    kernel->pad_ix_x=0;
    kernel->pad_ix_y=0;
    kernel->pad_iy_x=0;
    kernel->pad_iy_y=0;
    /*
    kernel->dilatation_x=0;
    kernel->dilatation_y=0;
    */
    // strides
    kernel->stride_x=0;
    kernel->stride_y=0;
    // fused layers pts
    kernel->bias_pt=0x0;
    kernel->batchnorm_mul=0x0;
    kernel->batchnorm_add=0x0;
    // output dim
    kernel->dim_O=0x0;
    kernel->mem_O=0;
    kernel->O_pt=0x0;
    // 1 input
    kernel->dim_I=0x0;
    kernel->mem_I=0;
    kernel->I_pt=0x0;
    // 2 inputs
    kernel->dim_X=0x0;
    kernel->mem_X=0;
    kernel->X_pt=0x0;
    kernel->dim_Y=0x0;
    kernel->mem_Y=0;
    kernel->Y_pt=0x0;
    // weights
    kernel->dim_W=0x0;
    kernel->mem_W=0;
    kernel->W_pt=0x0;
}