#include <match_kernel.h>

void init_common_kernel_params(common_kernel* kernel,unsigned int pattern_name,unsigned int specific_pattern){
    kernel->pattern_name=pattern_name;
    kernel->specific_pattern=specific_pattern;
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
    // pads
    kernel->pad_IX_x=0;
    kernel->pad_IX_y=0;
    kernel->pad_IY_x=0;
    kernel->pad_IY_y=0;
    // fused layers pts
    kernel->bias_pt=0x0;
    kernel->batchnorm_mul=0x0;
    kernel->batchnorm_add=0x0;
    // output dim
    kernel->dim_O=0x0;
    kernel->mem_O=0;
    kernel->O_pt=0x0;
    kernel->prec_O=0;
    // 1 input
    kernel->dim_I=0x0;
    kernel->mem_I=0;
    kernel->I_pt=0x0;
    kernel->prec_I=0;
    // 2 inputs
    kernel->dim_X=0x0;
    kernel->mem_X=0;
    kernel->X_pt=0x0;
    kernel->prec_X=0;
    kernel->dim_Y=0x0;
    kernel->mem_Y=0;
    kernel->Y_pt=0x0;
    kernel->prec_Y=0;
    // weights
    kernel->dim_W=0x0;
    kernel->mem_W=0;
    kernel->W_pt=0x0;
    kernel->prec_W=0;
}

void init_kernel_dimension_params_I_W_O(
    common_kernel* kernel,dimension_I* dim_I,unsigned int innermost_I_mem_level,unsigned int prec_I,
    dimension_W* dim_W,unsigned int innermost_W_mem_level,unsigned int prec_W,
    dimension_O* dim_O,unsigned int innermost_O_mem_level,unsigned int prec_O
){
    kernel->dim_I=dim_I;
    kernel->prec_I=prec_I;
    kernel->dim_W=dim_W;
    kernel->prec_W=prec_W;
    kernel->dim_O=dim_O;
    kernel->prec_O=prec_O;
    kernel->c_w=dim_W->size_C[innermost_W_mem_level];kernel->c_i=dim_I->size_C[innermost_I_mem_level];
    kernel->k_o=dim_O->size_K[innermost_O_mem_level];kernel->k_w=dim_W->size_K[innermost_W_mem_level];
    kernel->ix_i=dim_I->size_IX[innermost_I_mem_level];kernel->iy_i=dim_I->size_IY[innermost_I_mem_level];
    kernel->fx=dim_W->size_FX[innermost_W_mem_level];
    kernel->fy=dim_W->size_FY[innermost_W_mem_level];
    kernel->ox=dim_O->size_OX[innermost_O_mem_level];
    kernel->oy=dim_O->size_OY[innermost_O_mem_level];
}

void init_kernel_dimension_params_X_Y_O(
    common_kernel* kernel,dimension_X* dim_X,unsigned int innermost_X_mem_level,unsigned int prec_X,
    dimension_Y* dim_Y,unsigned int innermost_Y_mem_level,unsigned int prec_Y,
    dimension_O* dim_O,unsigned int innermost_O_mem_level,unsigned int prec_O
){
    kernel->dim_X=dim_X;
    kernel->prec_X=prec_X;
    kernel->dim_Y=dim_Y;
    kernel->prec_Y=prec_Y;
    kernel->dim_O=dim_O;
    kernel->prec_O=prec_O;
    kernel->c_x=dim_X->size_C[innermost_X_mem_level];kernel->c_y=dim_Y->size_C[innermost_Y_mem_level];
    kernel->k_o=dim_O->size_K[innermost_O_mem_level];
    kernel->ix_x=dim_X->size_IX[innermost_X_mem_level];kernel->iy_x=dim_X->size_IY[innermost_X_mem_level];
    kernel->ix_y=dim_Y->size_IX[innermost_Y_mem_level];kernel->iy_y=dim_Y->size_IY[innermost_Y_mem_level];
    kernel->ox=dim_O->size_OX[innermost_O_mem_level];
    kernel->oy=dim_O->size_OY[innermost_O_mem_level];
}

void match_innermost_computation(match_kernel* kernel,unsigned int pattern_name){
    return;
}

void kernel_set_padding(common_kernel* kernel,dimension_I* dim){
    kernel->pad_IX_x=dim->pad_IX_x;
    kernel->pad_IX_y=dim->pad_IX_y;
    kernel->pad_IY_x=dim->pad_IY_x;
    kernel->pad_IY_y=dim->pad_IY_y;
}