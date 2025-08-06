#include <pulp_cluster/train_wrapper.h>
/*
    PULP-TrainLib Wrapper
*/
void pulp_train_conv2d_fp32_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int is_bias = num_tensors > 3? 1: 0; // check if bias is present
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    
    int out_width, out_height, out_ch;
    int inp_width, inp_height, inp_ch;
    int stride_h, stride_w;
    int pad_top, pad_bottom, pad_left, pad_right;
    if (conv_attrs->data_layout == "NCHW"){
        // out chw
        out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out width
        out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out height
        out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out ch
        // inp chw
        inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out width
        inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out height
        inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out ch
        // pad
        pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
        pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+3]));
        pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
        pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+3]));
    
    } else {
        // out hwc
        out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
        out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
        out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
        // inp hwc
        inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out width
        inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out height
        inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out ch        
        // pad
        pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
        pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
        pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
        pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    }
    stride_h = conv_attrs->strides[0];
    stride_w = conv_attrs->strides[1];

    #ifdef CLUSTER_LIB_DEBUG
    printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    printf("Num tensors: %d\n", num_tensors);
    #endif

    // setup the arguments. FIXME: merge with the precedent
    struct blob layer1_in, layer1_wgt, layer1_bias, layer1_out;

    /* check if all the fields of the layers are assigned*/
    layer1_in.data = tensors[0].pt;
    layer1_in.dim = inp_height*inp_width*inp_ch;
    layer1_in.W = inp_width;
    layer1_in.H = inp_height;
    layer1_in.C = inp_ch;
  
    layer1_out.data = tensors[num_tensors-1].pt; // output pt 
    layer1_out.dim = out_height*out_width*out_ch;
    layer1_out.W = out_width;
    layer1_out.H = out_height;
    layer1_out.C = out_ch;
  
    layer1_wgt.data = tensors[1].pt; // weights pt   
    int is_fw = tensors[1].tensor_type == MATCH_CONST_TENSOR; 
    layer1_wgt.dim = conv_attrs->kernel_size[0]*conv_attrs->kernel_size[1]*inp_ch*out_ch;
    layer1_wgt.W = conv_attrs->kernel_size[1];
    layer1_wgt.H = conv_attrs->kernel_size[0];
    layer1_wgt.C = inp_ch;

    if ( is_bias){
        layer1_bias.data = tensors[2].pt; // bias pt
        layer1_bias.dim = out_ch;
    }

    int MATMUL_TYPE = 9; // 2x2
    int HWC_LAYOUT = 1 - (conv_attrs->data_layout == "NCHW"); // Choose if data layout is CHW (=0) or HWC (=1)
    // special pointwise acceleration
    if (layer1_wgt.W == 1 && layer1_wgt.H ==1 && !is_bias && 
        stride_h == 1 && stride_w == 1 &&
        pad_top == 0 && pad_bottom == 0 && pad_left == 0 && pad_right == 0){

        struct PointWise_Conv_args PW_args;
        PW_args.input = &layer1_in;
        PW_args.coeff = &layer1_wgt;
        PW_args.output = &layer1_out;
        PW_args.skip_wg_grad = 0;
        PW_args.skip_in_grad = 1;
        PW_args.opt_matmul_type_fw = MATMUL_TYPE;
        PW_args.opt_matmul_type_wg = MATMUL_TYPE;
        PW_args.opt_matmul_type_ig = MATMUL_TYPE;
        PW_args.HWC = HWC_LAYOUT;

        if(is_fw)   pulp_conv_pw_fp32_fw_cl(&PW_args);
        else        pulp_conv_pw_fp32_bw_cl(&PW_args);

    } else {
    // generic kernel convolution

        struct Conv2D_args C2D_args;
        C2D_args.input = &layer1_in;
        C2D_args.coeff = &layer1_wgt;
        C2D_args.output = &layer1_out;
        C2D_args.Lpad = pad_left;
        C2D_args.Rpad = pad_right;
        C2D_args.Upad = pad_top;
        C2D_args.Dpad = pad_bottom;
        C2D_args.stride_h = stride_h;
        C2D_args.stride_w = stride_w;
        C2D_args.i2c_buffer = get_im2col_pt();
        C2D_args.bt_buffer = NULL;
        C2D_args.skip_wg_grad = 0;
        C2D_args.skip_in_grad = 1;
        C2D_args.HWC = HWC_LAYOUT;
        C2D_args.opt_matmul_type_fw = MATMUL_TYPE;
        C2D_args.opt_matmul_type_wg = MATMUL_TYPE;
        C2D_args.opt_matmul_type_ig = MATMUL_TYPE;
        C2D_args.USE_IM2COL = (C2D_args.i2c_buffer != NULL); 
        C2D_args.USE_DMA_IM2COL = 0; // OK checkme

        
        if ( is_bias){
            C2D_args.bias = &layer1_bias; 
            C2D_args.USE_BIASES = 1; 
        } else {
            C2D_args.bias = NULL; 
            C2D_args.USE_BIASES = 0; 
        }

        if (is_fw)  pulp_conv2d_fp32_fw_cl(&C2D_args);
        else    pulp_conv2d_fp32_bw_cl(&C2D_args);
    }

}

void pulp_train_conv2ddw_fp32_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    
    int out_width, out_height, out_ch;
    int inp_width, inp_height, inp_ch;
    int pad_top, pad_bottom, pad_left, pad_right;

    if (conv_attrs->data_layout == "NCHW"){
        // out chw
        out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out width
        out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out height
        out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out ch
        // inp chw
        inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out width
        inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out height
        inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out ch
        // pad
        pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
        pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+3]));
        pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
        pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+3]));
    
    } else {
        // out hwc
        out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
        out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
        out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+3].size; // out ch
        // inp hwc
        inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // out width
        inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // out height
        inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+3].size; // out ch        
        // pad
        pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
        pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
        pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
        pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    }
    #ifdef CLUSTER_LIB_DEBUG
    printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    printf("Num tensors: %d\n", num_tensors);
    #endif

    // setup the arguments. FIXME: merge with the precedent
    struct blob layer1_in, layer1_wgt, layer1_bias, layer1_out;

    /* check if all the fields of the layers are assigned*/
    layer1_in.data = tensors[0].pt;
    layer1_in.dim = inp_height*inp_width*inp_ch;
    layer1_in.W = inp_width;
    layer1_in.H = inp_height;
    layer1_in.C = inp_ch;
  
    layer1_out.data = tensors[num_tensors-1].pt; // output pt 
    layer1_out.dim = out_height*out_width*out_ch;
    layer1_out.W = out_width;
    layer1_out.H = out_height;
    layer1_out.C = out_ch;
  
    layer1_wgt.data = tensors[1].pt; // weights pt   
    int is_fw = tensors[1].tensor_type == MATCH_CONST_TENSOR; 
    layer1_wgt.dim = conv_attrs->kernel_size[0]*conv_attrs->kernel_size[1]*inp_ch*out_ch;
    layer1_wgt.W = conv_attrs->kernel_size[1];
    layer1_wgt.H = conv_attrs->kernel_size[0];
    layer1_wgt.C = inp_ch;

    int HWC_LAYOUT = 1 - (conv_attrs->data_layout == "NCHW"); // Choose if data layout is CHW (=0) or HWC (=1)

    struct DepthWise_Conv_args DW_args;
    DW_args.input = &layer1_in;
    DW_args.coeff = &layer1_wgt;
    DW_args.output = &layer1_out;
    DW_args.Lpad = pad_left;
    DW_args.Rpad = pad_right;
    DW_args.Upad = pad_top;
    DW_args.Dpad = pad_bottom;
    DW_args.skip_wg_grad = 0;
    DW_args.skip_in_grad = 0;
    DW_args.HWC = HWC_LAYOUT;


    if(is_fw)   pulp_conv_dw_fp32_fw_cl(&DW_args);
    else        pulp_conv_dw_fp32_bw_cl(&DW_args);

}