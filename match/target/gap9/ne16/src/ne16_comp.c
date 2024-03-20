#include <ne16_comp.h>
#include <ne16_mem.h>

void ne16_kernel_function_wrapper(unsigned int task_id,match_kernel* kernel){
    if(task_id==EXECUTE_TASK){
        monitor_consume_begin(get_nnx_monitor()->input);
        monitor_produce_begin(get_nnx_monitor()->output);
        //if(kernel->common_kernel->stride_x==1){
            ne16_nnx_dispatch_wait(match_ne16_get_nnx_dev());
            ne16_nnx_dispatch(match_ne16_get_nnx_dev(), match_ne16_get_nnx_task(get_nnx_db_O(task_id)));
        //}
        //else{
        //    int i_channels=kernel->common_kernel->c_i;
        //    int i_width=kernel->common_kernel->ix_i;
        //    +kernel->common_kernel->dim_I->overlap_IX_x+kernel->common_kernel->dim_I->overlap_IX_y
        //    -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
        //    int i_height=kernel->common_kernel->iy_i;
        //    +kernel->common_kernel->dim_I->overlap_IY_x+kernel->common_kernel->dim_I->overlap_IY_y
        //    -kernel->common_kernel->dim_I->pad_IY_x-kernel->common_kernel->dim_I->pad_IY_y;
        //    int o_channels=kernel->common_kernel->k_o;
        //    int o_width=kernel->common_kernel->ox;
        //    int o_height=kernel->common_kernel->oy;
        //    int w_y_width=kernel->common_kernel->fx;
        //    int w_y_height=kernel->common_kernel->fy;
        //    ne16_nnx_dispatch_stride2x2(match_ne16_get_nnx_dev(), match_ne16_get_nnx_task(get_nnx_db_O(task_id)),
        //                         i_width, i_channels,
        //                         o_height, o_width,
        //                         o_channels, w_y_height,
        //                         w_y_width);
        //}
    }
    else if(task_id==LOADER_TASK){
        int i_channels=kernel->common_kernel->c_i;
        int i_width=kernel->common_kernel->ix_i;
        +kernel->common_kernel->dim_I->overlap_IX_x+kernel->common_kernel->dim_I->overlap_IX_y
        -kernel->common_kernel->dim_I->pad_IX_x-kernel->common_kernel->dim_I->pad_IX_y;
        int i_height=kernel->common_kernel->iy_i;
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
        if(kernel->common_kernel->stride_x==1){
            ne16_task_set_dims(match_ne16_get_nnx_task(get_nnx_db_O(task_id)), i_width, i_channels,
                    i_height, i_width, o_height,
                    o_width, o_channels, o_height,
                    o_width, p_top, p_bottom,
                    p_right, p_left);
            ne16_task_set_addr_conv(match_ne16_get_nnx_task(get_nnx_db_O(task_id)), kernel->common_kernel->I_pt,
                                i_width, i_width,p_top, p_left,
                                kernel->common_kernel->O_pt,kernel->common_kernel->W_pt);
            // ne16_task_set_addr_norm_quant      
        }
        else{
            ne16_task_set_dims_stride2x2(match_ne16_get_nnx_task(get_nnx_db_O(task_id)), i_height,i_width,i_channels,
                    i_height,i_width,o_height,o_width,o_channels,o_height,o_width,
                    w_y_height,w_y_width,p_top,p_bottom,p_left,p_right);
            ne16_task_set_addr_conv(match_ne16_get_nnx_task(get_nnx_db_O(task_id)), kernel->common_kernel->I_pt,
                                i_width, i_width,p_top, p_left,
                                kernel->common_kernel->O_pt,kernel->common_kernel->W_pt);
            // ne16_task_set_addr_norm_quant
        }
        monitor_produce_end(get_nnx_monitor()->input);
    }
}