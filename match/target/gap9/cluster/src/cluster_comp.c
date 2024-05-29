#include <cluster_comp.h>
#include <cluster_mem.h>

void cluster_init_other_kernel_params(cluster_kernel* kernel){
    kernel->im2col_pt=get_im2col_pt();
    kernel->pwtbuf_pt=get_pwtbuf();
}
void pw_conv_2d_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    void* im_2_col=kernel->im2col_pt;
    int i_channels=kernel->common_kernel->c_i;
    int i_width=kernel->common_kernel->ix_i
    +kernel->common_kernel->dim_I->overlap_IX_x+kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->iy_i
    +kernel->common_kernel->dim_I->overlap_IY_x+kernel->common_kernel->dim_I->overlap_IY_y
    -kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_channels=kernel->common_kernel->k_o;
    int o_width=kernel->common_kernel->ox;
    int o_height=kernel->common_kernel->oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=kernel->common_kernel->batchnorm_add!=0x0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    pulp_nn_pointwise_HoWo_parallel(
        kernel->common_kernel->I_pt,
        im_2_col,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->common_kernel->W_pt,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->common_kernel->right_shift,
        i_width,i_height,i_channels,
        o_width,o_height,o_channels,
        w_y_width,w_y_height,
        p_top,p_bottom,p_left,p_right,
        kernel->common_kernel->stride_x,kernel->common_kernel->stride_y,
        act,batch_norm
    );
}

void dw_conv_2d_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    void* im_2_col=kernel->im2col_pt;
    int i_channels=kernel->common_kernel->c_i;
    int i_width=kernel->common_kernel->ix_i
    +kernel->common_kernel->dim_I->overlap_IX_x+kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->iy_i
    +kernel->common_kernel->dim_I->overlap_IY_x+kernel->common_kernel->dim_I->overlap_IY_y
    -kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_channels=kernel->common_kernel->k_o;
    int o_width=kernel->common_kernel->ox;
    int o_height=kernel->common_kernel->oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=kernel->common_kernel->batchnorm_add!=0x0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    //printf("Conv 2d :: pad ^ %d v %d < %d > %d\n",p_top,p_bottom,p_left,p_right);
    //printf("Conv 2d :: I [ C %d IY %d IX %d ] O [ K %d OY %d OX %d ] W [ FY %d FX %d ]\n",i_channels,i_height,i_width,o_channels,o_height,o_width,w_y_height,w_y_width);
    //printf("Conv 2d :: addr I %d O %d W %d\n",kernel->common_kernel->I_pt,kernel->common_kernel->O_pt,kernel->common_kernel->W_pt);
    pulp_nn_depthwise_generic(
        kernel->common_kernel->I_pt,
        im_2_col,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->common_kernel->W_pt,
        kernel->pwtbuf_pt,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->common_kernel->right_shift,
        i_width,i_height,i_channels,
        o_width,o_height,o_channels,
        w_y_width,w_y_height,
        p_top,p_bottom,p_left,p_right,
        kernel->common_kernel->stride_x,kernel->common_kernel->stride_y,
        act,batch_norm
    );
}

void dw_less_four_fs_conv_2d_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    void* im_2_col=kernel->im2col_pt;
    int i_channels=kernel->common_kernel->c_i;
    int i_width=kernel->common_kernel->ix_i
    +kernel->common_kernel->dim_I->overlap_IX_x+kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->iy_i
    +kernel->common_kernel->dim_I->overlap_IY_x+kernel->common_kernel->dim_I->overlap_IY_y
    -kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_channels=kernel->common_kernel->k_o;
    int o_width=kernel->common_kernel->ox;
    int o_height=kernel->common_kernel->oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=kernel->common_kernel->batchnorm_add!=0x0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    pulp_nn_depthwise_generic_less_4_weights(
        kernel->common_kernel->I_pt,
        im_2_col,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->common_kernel->W_pt,
        kernel->pwtbuf_pt,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->common_kernel->right_shift,
        i_width,i_height,i_channels,
        o_width,o_height,o_channels,
        w_y_width,w_y_height,
        p_top,p_bottom,p_left,p_right,
        kernel->common_kernel->stride_x,kernel->common_kernel->stride_y,
        act,batch_norm
    );
}


void conv_2d_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    void* im_2_col=kernel->im2col_pt;
    int i_channels=kernel->common_kernel->c_i;
    int i_width=kernel->common_kernel->ix_i
    +kernel->common_kernel->dim_I->overlap_IX_x+kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->iy_i
    +kernel->common_kernel->dim_I->overlap_IY_x+kernel->common_kernel->dim_I->overlap_IY_y
    -kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_channels=kernel->common_kernel->k_o;
    int o_width=kernel->common_kernel->ox;
    int o_height=kernel->common_kernel->oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=kernel->common_kernel->batchnorm_add!=0x0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    //printf("Conv 2d :: pad ^ %d v %d < %d > %d\n",p_top,p_bottom,p_left,p_right);
    //printf("Conv 2d :: I [ C %d IY %d IX %d ] O [ K %d OY %d OX %d ] W [ FY %d FX %d ]\n",i_channels,i_height,i_width,o_channels,o_height,o_width,w_y_height,w_y_width);
    //printf("Conv 2d :: addr I %d O %d W %d\n",kernel->common_kernel->I_pt,kernel->common_kernel->O_pt,kernel->common_kernel->W_pt);
    pulp_nn_conv_Ho_parallel(
        kernel->common_kernel->I_pt,
        im_2_col,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->common_kernel->W_pt,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->common_kernel->right_shift,
        i_width,i_height,i_channels,
        o_width,o_height,o_channels,
        w_y_width,w_y_height,
        p_top,p_bottom,p_left,p_right,
        kernel->common_kernel->stride_x,kernel->common_kernel->stride_y,
        act,batch_norm
    );
}

void add_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    int o_channels=kernel->common_kernel->k_o;
    int o_width=kernel->common_kernel->ox;
    int o_height=kernel->common_kernel->oy;
    pulp_nn_add(
        kernel->common_kernel->X_pt,
        kernel->common_kernel->Y_pt,
        kernel->common_kernel->O_pt,
        1,1,
        kernel->common_kernel->right_shift,
        o_width,o_height,o_channels
    );
}

void dense_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    int i_channels=kernel->common_kernel->c_i>kernel->common_kernel->c_w?kernel->common_kernel->c_w:kernel->common_kernel->c_i;
    int o_channels=kernel->common_kernel->k_o;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=kernel->common_kernel->batchnorm_add!=0x0;
    pulp_nn_linear(
        kernel->common_kernel->I_pt,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->common_kernel->W_pt,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->common_kernel->right_shift,
        i_channels,o_channels,
        act,batch_norm
    );
}

void dense_out_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    int i_channels=kernel->common_kernel->c_i>kernel->common_kernel->c_w?kernel->common_kernel->c_w:kernel->common_kernel->c_i;
    int o_channels=kernel->common_kernel->k_o;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=kernel->common_kernel->batchnorm_add!=0x0;
    pulp_nn_linear_out_32(
        kernel->common_kernel->I_pt,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->common_kernel->W_pt,
        i_channels,o_channels
    );
}
void cluster_kernel_function_wrapper(cluster_kernel* kernel){
    if(kernel->common_kernel->specific_pattern==pointwise_conv2d)
        pi_team_offload_preset(pw_conv_2d_comp, kernel);
    else if(kernel->common_kernel->specific_pattern==depthwise_conv2d_less_4)
        pi_team_offload_preset(dw_less_four_fs_conv_2d_comp, kernel);
    else if(kernel->common_kernel->specific_pattern==depthwise_conv2d)
        pi_team_offload_preset(dw_conv_2d_comp, kernel);
    else if(kernel->common_kernel->specific_pattern==conv2d)
        pi_team_offload_preset(conv_2d_comp, kernel);
    else if(kernel->common_kernel->specific_pattern==dense)
        pi_team_offload_preset(dense_comp, kernel);
    else if(kernel->common_kernel->specific_pattern==dense_out)
        pi_team_offload_preset(dense_out_comp, kernel);
    else if(kernel->common_kernel->specific_pattern==elemwise_add)
        pi_team_offload_preset(add_comp, kernel);
}