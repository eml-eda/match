#include <pulp_cluster/train_wrapper.h>
/*
    PULP-TrainLib Wrapper
*/
void pulp_train_conv2d_fp32_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors-1;
    int is_bias = num_tensors > 3? 1: 0; // check if bias is present
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    
    int ch_idx = 1; // channel index
    int height_idx = 2; // height index
    int width_idx = 3; // width index
    int HWC_LAYOUT = 0;

    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW"){
        ch_idx = 3; // channel index
        height_idx = 1; // height index
        width_idx = 2; // width index
        HWC_LAYOUT = 1; // Choose if data layout is CHW (=0) or HWC (=1)
    }
    #endif

    // out chw
    int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size; // out batches
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch params
    // inp chw
    int inp_batches = tensors[0].tiles[L1_SCRATCHPAD*4+0].size; // in batches
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int stride_h = conv_attrs->strides[0];
    int stride_w = conv_attrs->strides[1];
    int groups = conv_attrs->groups;
    int dilation_h = conv_attrs->dilation[0];
    int dilation_w = conv_attrs->dilation[1];
    int kernel_h = conv_attrs->kernel_size[0];
    int kernel_w = conv_attrs->kernel_size[1];
    int is_dw = conv_attrs->depthwise;
    #ifdef CLUSTER_LIB_DEBUG
    printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    printf("Num tensors: %d, is fw? %d\n", num_tensors, is_fw);
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
    layer1_wgt.dim = conv_attrs->kernel_size[0]*conv_attrs->kernel_size[1]*inp_ch*out_ch;
    layer1_wgt.W = conv_attrs->kernel_size[1];
    layer1_wgt.H = conv_attrs->kernel_size[0];
    layer1_wgt.C = inp_ch;

    if ( is_bias){
        layer1_bias.data = tensors[2].pt; // bias pt
        layer1_bias.dim = out_ch;
    }

    int MATMUL_TYPE = 9; // 2x2
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

        pulp_conv_pw_fp32_fw_cl(&PW_args);

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

        pulp_conv2d_fp32_fw_cl(&C2D_args);
    }

}

void pulp_train_conv2ddw_fp32_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors-1;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    
    int ch_idx = 1; // channel index
    int height_idx = 2; // height index
    int width_idx = 3; // width index
    int HWC_LAYOUT = 0;

    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW"){
        ch_idx = 3; // channel index
        height_idx = 1; // height index
        width_idx = 2; // width index
        HWC_LAYOUT = 1; // Choose if data layout is CHW (=0) or HWC (=1)
    }
    #endif

    // out chw
    int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size; // out batches
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch params
    // inp chw
    int inp_batches = tensors[0].tiles[L1_SCRATCHPAD*4+0].size; // in batches
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int stride_h = conv_attrs->strides[0];
    int stride_w = conv_attrs->strides[1];
    int groups = conv_attrs->groups;
    int dilation_h = conv_attrs->dilation[0];
    int dilation_w = conv_attrs->dilation[1];
    int kernel_h = conv_attrs->kernel_size[0];
    int kernel_w = conv_attrs->kernel_size[1];
    int is_dw = conv_attrs->depthwise;
    int is_fw = tensors[1].tensor_type == MATCH_CONST_TENSOR; 
    #ifdef CLUSTER_LIB_DEBUG
    printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    printf("Num tensors: %d, is fw? %d\n", num_tensors, is_fw);
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
    layer1_wgt.dim = conv_attrs->kernel_size[0]*conv_attrs->kernel_size[1]*inp_ch*out_ch;
    layer1_wgt.W = conv_attrs->kernel_size[1];
    layer1_wgt.H = conv_attrs->kernel_size[0];
    layer1_wgt.C = inp_ch;

    struct DepthWise_Conv_args DW_args;
    DW_args.input = &layer1_in;
    if(!is_fw){
        DW_args.coeff = &layer1_wgt;
        DW_args.output = &layer1_out;
        DW_args.coeff->diff = layer1_out.data;
        DW_args.output->diff = layer1_wgt.data; // outDiff
    }
    else{
        DW_args.coeff = &layer1_wgt;
        DW_args.output = &layer1_out;
    }
    DW_args.Lpad = pad_left;
    DW_args.Rpad = pad_right;
    DW_args.Upad = pad_top;
    DW_args.Dpad = pad_bottom;
    DW_args.skip_wg_grad = 0;
    DW_args.skip_in_grad = 1;
    DW_args.HWC = HWC_LAYOUT;


    if(is_fw)   pulp_conv_dw_fp32_fw_cl(&DW_args);
    else{
        float* inData = DW_args.input->data;
        float* coeffDiff = DW_args.coeff->diff;
        float* outDiff = DW_args.output->diff;
        for(int i=0;i<out_batches;i++){
            DW_args.input->data = inData + i*inp_height*inp_width*inp_ch;
            DW_args.coeff->diff = coeffDiff + i*out_height*out_width*out_ch;
            pulp_conv_dw_fp32_bw_cl(&DW_args);
        }
    }
}

struct fill_i2c_args {
    float *i2c_buffer;
    float *activation_pt;
    int out_height;
    int out_width;
    int inp_height;
    int inp_width;
    int inp_ch;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int pad_top;
    int pad_left;
};

void fill_im2col_buffer(
    void* args
) {
    struct fill_i2c_args* im2col_args = (struct fill_i2c_args*)args;
    float * __restrict__ i2c_buffer = im2col_args->i2c_buffer;
    float * __restrict__ activation_pt = im2col_args->activation_pt;
    const int out_height = im2col_args->out_height;
    const int out_width = im2col_args->out_width;
    const int inp_height = im2col_args->inp_height;
    const int inp_width = im2col_args->inp_width;
    const int inp_ch = im2col_args->inp_ch;
    const int kernel_h = im2col_args->kernel_h;
    const int kernel_w = im2col_args->kernel_w;
    const int stride_h = im2col_args->stride_h;
    const int stride_w = im2col_args->stride_w;
    const int dilation_h = im2col_args->dilation_h;
    const int dilation_w = im2col_args->dilation_w;
    const int pad_top = im2col_args->pad_top;
    const int pad_left = im2col_args->pad_left;
    
    const int block_size_c = (inp_ch + NUM_CORES - 1) / NUM_CORES;
    const int start_c = pi_core_id() * block_size_c;
    const int stop_c = (start_c + block_size_c > inp_ch) ? inp_ch : start_c + block_size_c;
    
    // Pre-compute constants to reduce arithmetic in loops
    const int out_spatial_size = out_height * out_width;
    const int inp_spatial_size = inp_height * inp_width;
    const int kernel_size = kernel_h * kernel_w;
    
    // Reorder loops for better cache locality and reduced index calculations
    for(int c = start_c; c < stop_c; c++) {
        const int c_offset = c * inp_spatial_size;
        const int c_kernel_offset = (c - start_c) * kernel_size;
        
        for(int k_h = 0; k_h < kernel_h; k_h++) {
            const int k_h_dil = k_h * dilation_h;
            const int k_h_offset = k_h * kernel_w;
            
            for(int k_w = 0; k_w < kernel_w; k_w++) {
                const int k_w_dil = k_w * dilation_w;
                const int i2c_base_idx = (c_kernel_offset + k_h_offset + k_w) * out_spatial_size;
                
                for(int o_h_idx = 0; o_h_idx < out_height; o_h_idx++) {
                    const int inp_h = o_h_idx * stride_h - pad_top + k_h_dil;
                    const int o_h_offset = o_h_idx * out_width;
                    
                    // Check if this entire row is within bounds
                    if (inp_h >= 0 && inp_h < inp_height) {
                        const int inp_h_offset = c_offset + inp_h * inp_width;
                        
                        for(int o_w_idx = 0; o_w_idx < out_width; o_w_idx++) {
                            const int inp_w = o_w_idx * stride_w - pad_left + k_w_dil;
                            const int i2c_idx = i2c_base_idx + o_h_offset + o_w_idx;
                            
                            if (inp_w >= 0 && inp_w < inp_width) {
                                i2c_buffer[i2c_idx] = activation_pt[inp_h_offset + inp_w];
                            } else {
                                i2c_buffer[i2c_idx] = 0.0f; // Padding
                            }
                        }
                    } else {
                        // Entire row is padding
                        for(int o_w_idx = 0; o_w_idx < out_width; o_w_idx++) {
                            i2c_buffer[i2c_base_idx + o_h_offset + o_w_idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void pulp_train_conv2d_bw_fp32_wrapper(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors-1;
    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    
    float * __restrict__ activations_pt = tensors[0].pt;
    float * __restrict__ parameters_pt = tensors[1].pt;
    float * __restrict__ output_pt = tensors[num_tensors-1].pt;

    float * __restrict__ bias_pt = NULL;
    if (num_tensors > 3) bias_pt = tensors[2].pt;

    int ch_idx = 1; // channel index
    int height_idx = 2; // height index
    int width_idx = 3; // width index
    int HWC_LAYOUT = 0;

    #ifdef ODL_SUPPORT_NHWC
    if (conv_attrs->data_layout != "NCHW"){
        ch_idx = 3; // channel index
        height_idx = 1; // height index
        width_idx = 2; // width index
        HWC_LAYOUT = 1; // Choose if data layout is CHW (=0) or HWC (=1)
    }
    #endif

    // out chw
    int out_batches = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+0].size; // out batches
    int out_width = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int out_height = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int out_ch = tensors[output_tensor_idx].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    int out_ch_params = tensors[1].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch params
    // inp chw
    int inp_batches = tensors[0].tiles[L1_SCRATCHPAD*4+0].size; // in batches
    int inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+width_idx].size; // out width
    int inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+height_idx].size; // out height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*4+ch_idx].size; // out ch
    // pad
    int pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+height_idx]));
    int pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+width_idx]));
    int stride_h = conv_attrs->strides[0];
    int stride_w = conv_attrs->strides[1];
    int groups = conv_attrs->groups;
    int dilation_h = conv_attrs->dilation[0];
    int dilation_w = conv_attrs->dilation[1];
    int kernel_h = conv_attrs->kernel_size[0];
    int kernel_w = conv_attrs->kernel_size[1];
    int is_dw = conv_attrs->depthwise;
    #ifdef CLUSTER_LIB_DEBUG
    printf("Out tile [%d %d %d] Inp tile [%d %d %d] pad ^ %d v %d < %d > %d\n", out_ch, out_height, out_width, inp_ch, inp_height, inp_width,
            pad_top, pad_bottom, pad_left, pad_right);
    #endif

    void* i2c_buffer = get_im2col_pt();
    int MATMUL_TYPE = 9; // 2x2
    int use_pw = kernel_h == 1 && kernel_w ==1 && bias_pt==NULL && 
        stride_h == 1 && stride_w == 1 &&
        pad_top == 0 && pad_bottom == 0 && pad_left == 0 && pad_right == 0;
    struct matMul_args matMul_args;
    struct fill_i2c_args fill_i2c_args;
    // Initialize the fill_i2c_args
    fill_i2c_args.i2c_buffer = i2c_buffer;
    fill_i2c_args.activation_pt = activations_pt;
    fill_i2c_args.out_height = out_height;
    fill_i2c_args.out_width = out_width;
    fill_i2c_args.inp_height = inp_height;
    fill_i2c_args.inp_width = inp_width;
    fill_i2c_args.inp_ch = inp_ch;
    fill_i2c_args.kernel_h = kernel_h;
    fill_i2c_args.kernel_w = kernel_w;
    fill_i2c_args.stride_h = stride_h;
    fill_i2c_args.stride_w = stride_w;
    fill_i2c_args.dilation_h = dilation_h;
    fill_i2c_args.dilation_w = dilation_w;
    fill_i2c_args.pad_top = pad_top;
    fill_i2c_args.pad_left = pad_left;
    // Initialize the matMul_args
    matMul_args.A = parameters_pt;
    matMul_args.B = i2c_buffer!=NULL? i2c_buffer : activations_pt;
    matMul_args.C = output_pt;
    matMul_args.N = out_ch; 
    matMul_args.K = inp_ch * kernel_h * kernel_w; 
    matMul_args.M = out_height*out_width; 
    matMul_args.trans_B = use_pw;

    matMul_args.HWC = HWC_LAYOUT;
    matMul_args.bias = bias_pt;
    matMul_args.USE_BIASES = bias_pt != NULL;

    matMul_args.pH = kernel_h;
    matMul_args.pW = kernel_w;

    matMul_args.bias_dim = out_ch;
    struct mm_manager_args man_args;
    man_args.mm_args = &matMul_args;
    man_args.layer_type = use_pw? LAYER_PW_CONV : LAYER_CONV2D;
    man_args.step_type = STEP_WGT_GRAD;
    man_args.matmul_type = MATMUL_TYPE; //MATMUL_TYPE;
    for(int b=0; b<out_batches; b++){
        if(use_pw){
            matMul_args.B = activations_pt + b*inp_height*inp_width*inp_ch;
        }else{
            fill_i2c_args.activation_pt = activations_pt + b*inp_height*inp_width*inp_ch;
            // Fill the im2col buffer
            pi_cl_team_fork(NUM_CORES, fill_im2col_buffer, &fill_i2c_args);
        }
        matMul_args.C = output_pt + b*out_height*out_width*out_ch;
        
        pi_cl_team_fork(NUM_CORES, im2col_conv2d_param_grad_kernel, &man_args);
    }
}