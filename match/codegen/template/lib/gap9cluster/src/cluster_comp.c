void init_other_kernel_params(cluster_kernel* kernel){
    kernel->im2col=get_im2col_pt();
    kernel->pwtbuf=get_pwtbuf();
}

void pw_conv_2d_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    void* im_2_col=kernel->im2col;
    int i_channels=kernel->common_kernel->i_c;
    int i_width=kernel->common_kernel->i_ix+kernel->common_kernel->dim_I->overlap_IX_x+
    kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->i_iy+kernel->common_kernel->dim_I->overlap_IY_x+
    kernel->common_kernel->dim_I->overlap_IY_y-kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_channels=kernel->common_kernel->k;
    int o_width=kernel->common_kernel->o_ox;
    int o_height=kernel->common_kernel->o_oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    pulp_nn_pointwise_HoWo_parallel(
        kernel->I_pt,
        im_2_col,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->W_pt,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->output_shift,
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
    void* im_2_col=kernel->im2col;
    int i_channels=kernel->common_kernel->i_c;
    int i_width=kernel->common_kernel->i_ix+kernel->common_kernel->dim_I->overlap_IX_x+
    kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->i_iy+kernel->common_kernel->dim_I->overlap_IY_x+
    kernel->common_kernel->dim_I->overlap_IY_y-kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_channels=kernel->common_kernel->k;
    int o_width=kernel->common_kernel->o_ox;
    int o_height=kernel->common_kernel->o_oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    pulp_nn_depthwise_generic(
        kernel->I_pt,
        im_2_col,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->W_pt,
        kernel->pwtbuf,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->output_shift,
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
    void* im_2_col=kernel->im2col;
    int i_channels=kernel->common_kernel->i_c;
    int i_width=kernel->common_kernel->i_ix+kernel->common_kernel->dim_I->overlap_IX_x+
    kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->i_iy+kernel->common_kernel->dim_I->overlap_IY_x+
    kernel->common_kernel->dim_I->overlap_IY_y-kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_channels=kernel->common_kernel->k;
    int o_width=kernel->common_kernel->o_ox;
    int o_height=kernel->common_kernel->o_oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    pulp_nn_depthwise_generic_less_4_weights(
        kernel->I_pt,
        im_2_col,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->W_pt,
        kernel->pwtbuf,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->output_shift,
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
    void* im_2_col=kernel->im2col;
    int i_channels=kernel->common_kernel->i_c;
    int i_width=kernel->common_kernel->i_ix+kernel->common_kernel->dim_I->overlap_IX_x+
    kernel->common_kernel->dim_I->overlap_IX_y
    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
    int i_height=kernel->common_kernel->i_iy+kernel->common_kernel->dim_I->overlap_IY_x+
    kernel->common_kernel->dim_I->overlap_IY_y-kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
    int o_channels=kernel->common_kernel->k;
    int o_width=kernel->common_kernel->o_ox;
    int o_height=kernel->common_kernel->o_oy;
    int w_y_width=kernel->common_kernel->fx;
    int w_y_height=kernel->common_kernel->fy;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=0;
    int p_top=kernel->common_kernel->pad_IY_x;
    int p_bottom=kernel->common_kernel->pad_IY_y;
    int p_left=kernel->common_kernel->pad_IX_x;
    int p_right=kernel->common_kernel->pad_IX_y;
    pulp_nn_conv_Ho_parallel(
        kernel->I_pt,
        im_2_col,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->W_pt,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->output_shift,
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
    int o_channels=kernel->common_kernel->k;
    int o_width=kernel->common_kernel->o_ox;
    int o_height=kernel->common_kernel->o_oy;
    pulp_nn_add(
        kernel->common_kernel->X_pt,
        kernel->common_kernel->Y_pt,
        kernel->common_kernel->O_pt,
        1,
        kernel->output_shift,
        o_width,o_height,o_channels,
    );
}

void dense_comp(void* args){
    cluster_kernel* kernel = (cluster_kernel*)args;
    int i_channels=kernel->common_kernel->i_c;
    int o_channels=kernel->common_kernel->k;
    int act=kernel->common_kernel->activation_function;
    int batch_norm=0;
    pulp_nn_linear(
        kernel->common_kernel->I_pt,
        kernel->common_kernel->bias_pt,
        kernel->common_kernel->O_pt,
        kernel->common_kernel->W_pt,
        kernel->common_kernel->batchnorm_mul,
        kernel->common_kernel->batchnorm_add,
        1,
        kernel->output_shift,
        i_channels,o_channels,
        act,batch_norm
    );
}
void kernel_function_wrapper(cluster_kernel* kernel,cluster_patterns pattern){
    if(pattern==gap9cluster_conv2d && kernel->common_kernel->dim_W->size_FX[l2_mem]==1
        && kernel->common_kernel->dim_W->size_FY[l2_mem]==1) pw_flag=true;
    if(pattern==gap9cluster_conv2d && kernel->common_kernel->dim_W->size_C[l2_mem]==1
        && kernel->common_kernel->dim_W->size_K[l2_mem]!=1) dw_flag=true;
    if(pattern==gap9cluster_conv2d && 
    (kernel->common_kernel->dim_W->size_C[l2_mem]*kernel->common_kernel->dim_W->size_K[l2_mem])<4) less_four_fs_flag=true;
    if(pattern==gap9cluster_conv2d && pw_flag)
        pi_team_offload_preset(pw_conv_2d_comp, kernel);
    else if(pattern==gap9cluster_conv2d && dw_flag && less_four_fs_flag)
        pi_team_offload_preset(dw_less_four_fs_conv_2d_comp, kernel);
    else if(pattern==gap9cluster_conv2d && dw_flag)
        pi_team_offload_preset(dw_conv_2d_comp, kernel);
    else if(pattern==gap9cluster_conv2d)
        pi_team_offload_preset(conv_2d_comp, kernel);
    else if(pattern==gapcluster_dense)
        pi_team_offload_preset(dense_comp, kernel);
    else if(pattern==gap9cluster_add)
        pi_team_offload_preset(add_comp, kernel);
}