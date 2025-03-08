#include <maupiti.h>

void maupiti_kernels(match_kernel* kernel){
    if(kernel->common_kernel->specific_pattern==conv2d_bnorm_requant)
        conv_2d_bnorm_requant_comp_kernel(kernel);
    else if(kernel->common_kernel->specific_pattern==dense_out)
        dense_out_comp_kernel(kernel);
    else if(kernel->common_kernel->specific_pattern==maxpool2d)
        maxpool2d_comp_kernel(kernel);
}

void conv_2d_bnorm_requant_comp_kernel(void *args) {
    match_kernel* kernel = (match_kernel*)args;
    // Signed here
    int8_t* ibuffer=kernel->common_kernel->I_pt;
    int8_t* weight_buffer=kernel->common_kernel->W_pt;
    int8_t* bias_buffer=kernel->common_kernel->bias_pt;
    int8_t* obuffer=kernel->common_kernel->O_pt;
    int i_channels=kernel->common_kernel->c_i;
    int i_width=kernel->common_kernel->ix_i;
    int i_height=kernel->common_kernel->iy_i;
    int o_channels=kernel->common_kernel->k_o;
    int o_width=kernel->common_kernel->ox;
    int fx=kernel->common_kernel->fx;
    int fy=kernel->common_kernel->fy;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;

    int stride_x=kernel->common_kernel->stride_x;
    int stride_y=kernel->common_kernel->stride_y;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=kernel->common_kernel->batchnorm_add!=0x0;
    int right_shift=kernel->common_kernel->right_shift;
    int8_t* bn_mul_buffer=kernel->common_kernel->bn_mul_pt;
    int8_t* bn_add_buffer=kernel->common_kernel->bn_add_pt;


    maupiti_nn_conv_i8_i8_i8(
        ibuffer,
        im2col_buffer,
        bias_buffer,
        obuffer,
        weight_buffer,
        bn_mul_buffer,
        bn_add_buffer,
        1, // Shift
        right_shift,
        iw, ih, ic,
        ow, oh, oc,
        fx, fy,
        p_top, p_bottom, p_left, p_right,
        stride_x, stride_y,
        act, batch_norm,
        1 // Requantize
    )
}