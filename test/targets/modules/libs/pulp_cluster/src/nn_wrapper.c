#include <pulp_cluster/nn_wrapper.h>
/*
    PULP-NN Wrapper
*/
void pulp_nn_dense_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    pulp_nn_linear(
        // activations pt  
        tensors[0].pt, // acts pt
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        tensors[0].tiles[L1_SCRATCHPAD*2+1].size, // input channels
        tensors[num_tensors-1].tiles[L1_SCRATCHPAD*2+1].size, // output channels
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_dense_out_int_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    pulp_nn_linear_out_32(
        // activations pt  
        tensors[0].pt, // acts pt
        // bias pt
        tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        tensors[0].tiles[L1_SCRATCHPAD*2+1].size, // input channels
        tensors[num_tensors-1].tiles[L1_SCRATCHPAD*2+1].size // output channels
    );
}

void pulp_nn_dw_conv2d_less_4_wrapper(void* args){
    // TODO: implement, currently not used
    return;
}

void pulp_nn_dw_conv2d_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    #ifdef CLUSTER_LIB_DEBUG
    if(pi_core_id()==0)
        printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_depthwise_generic(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        get_im2col_pt(),
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        get_pwt_pt(), // pwt buffer pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_width, // input width
        inp_height, // input height
        inp_ch, // input channels
        out_width, // out width
        out_height, // out height
        out_ch, // out ch
        conv_attrs->kernel_size[1], // filter width
        conv_attrs->kernel_size[0], // filter height
        pad_top, // pad top
        pad_bottom, // pad bottom
        pad_left, // pad left
        pad_right, // pad right
        conv_attrs->strides[1], // stride width
        conv_attrs->strides[0], // stride height
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_pw_conv2d_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    #ifdef CLUSTER_LIB_DEBUG
    if(pi_core_id()==0)
        printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_pointwise_HoWo_parallel(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        get_im2col_pt(),
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_width, // input width
        inp_height, // input height
        inp_ch, // input channels
        out_width, // out width
        out_height, // out height
        out_ch, // out ch
        conv_attrs->kernel_size[1], // filter width
        conv_attrs->kernel_size[0], // filter height
        pad_top, // pad top
        pad_bottom, // pad bottom
        pad_left, // pad left
        pad_right, // pad right
        conv_attrs->strides[1], // stride width
        conv_attrs->strides[0], // stride height
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_hoparallel_conv2d_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    #ifdef CLUSTER_LIB_DEBUG
    if(pi_core_id()==0)
        printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif
    pulp_nn_conv_Ho_parallel(
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        get_im2col_pt(),
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_width, // input width
        inp_height, // input height
        inp_ch, // input channels
        out_width, // out width
        out_height, // out height
        out_ch, // out ch
        conv_attrs->kernel_size[1], // filter width
        conv_attrs->kernel_size[0], // filter height
        pad_top, // pad top
        pad_bottom, // pad bottom
        pad_left, // pad left
        pad_right, // pad right
        conv_attrs->strides[1], // stride width
        conv_attrs->strides[0], // stride height
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_conv3d_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    MatchConv3DAttrs* conv_attrs = (MatchConv3DAttrs*)ctx->ops->ops[0].attrs;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*5+3].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*5+2].size; // out height
    int out_depth = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*5+1].size; // out depth
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*5+4].size; // out ch
    // inp
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*5+3].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*5+2].size; // out height
    int inp_depth = tensors[0].tiles[L1_SCRATCHPAD*5+1].size; // out depth
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*5+4].size; // out ch
    // pad
    int pad_front = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+1]));
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+2]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+3]));
    int pad_back = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+1]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+2]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*5+3]));
    #ifdef CLUSTER_LIB_DEBUG
    if(pi_core_id()==0)
        printf("Out tile [%d %d %d %d] Inp tile [%d %d %d %d] pad \\ %d / %d ^ %d v %d < %d > %d\n",
            out_ch, out_depth, out_height, out_width,
            inp_ch, inp_depth, inp_height, inp_width,
            pad_front, pad_back,
            pad_top, pad_bottom, pad_left, pad_right
        );
    #endif
    #ifdef CLUSTER_LIB_USE_NAIVE_CONV3D
    pulp_nn_conv3d_naive
    #else
    pulp_nn_conv3d_Co_parallel
    #endif
    (
        // activations pt  
        tensors[0].pt, // acts pt
        // im2col
        get_im2col_pt(),
        // bias pt
        num_tensors>4? NULL:tensors[2].pt, // bias pt
        // output pt
        tensors[num_tensors-1].pt, // output pt
        // weights pt
        tensors[1].pt, // weights pt
        num_tensors>4? tensors[2].pt:NULL, // bnorm mul pt
        num_tensors>4? tensors[3].pt:NULL, // bnorm add pt
        1, // requant mult factor
        right_shift, // requant shift factor
        inp_width, // input width
        inp_height, // input height
        inp_depth, // input depth
        inp_ch, // input channels
        out_width, // out width
        out_height, // out height
        out_depth, // out depth
        out_ch, // out ch
        conv_attrs->kernel_size[2], // filter width
        conv_attrs->kernel_size[1], // filter height
        conv_attrs->kernel_size[0], // filter depth
        pad_top, // pad top
        pad_bottom, // pad bottom
        pad_left, // pad left
        pad_right, // pad right
        pad_front, // pad front
        pad_back, // pad back
        conv_attrs->strides[2], // stride width
        conv_attrs->strides[1], // stride height
        conv_attrs->strides[0], // stride depth
        1, // activation is on
        num_tensors>4 // using bnorm or bias --> using bnorm on this pattern
    );
}

void pulp_nn_add_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    // out
    int out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
    int out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
    pulp_nn_add(
        // activations pt  
        tensors[0].pt, // acts 1 pt
        tensors[1].pt, // acts 2 pt
        tensors[num_tensors-1].pt, // out
        1, // out mult 1
        1, // out mult 2
        right_shift,
        // sizes of tile
        out_width, out_height, out_ch
    );
}