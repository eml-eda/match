#include <pulp_neural/neural_engine.h>

void neural_engine_lib_init(MatchCtx* ctx){

    match_ne16_set_nnx_dev();
    ne16_nnx_init(match_ne16_get_nnx_dev(),(ne16_pulp_conf_t){
        .max_stall = 8
    });

    // alloc 
    hwpe_soft_clear(&(match_ne16_get_nnx_dev()->hwpe_dev));
    #ifdef DEBUG_GVSOC
    nnx_activate_gvsoc_logging(GVSOC_LOG_LEVEL_CONFIG, GVSOC_LOGGING_FORMAT_DECIMAL);
    #endif
    MatchConv2DAttrs* conv_attrs = ctx->pattern_name==conv2d || ctx->pattern_name==depthwise_conv2d?
                                    (MatchConv2DAttrs*)ctx->ops->ops[0].attrs:
                                    NULL;
    int stride_height = 1;
    int filter_height = 1;
    int activation = 1;
    int inp_bits = ctx->tensors->tensors[0].bits;
    int weights_bits = ctx->tensors->tensors[1].bits;
    int out_bits = ctx->tensors->tensors[ctx->tensors->num_tensors-1].bits;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[ctx->ops->num_ops-3].attrs)->right_shift;
    if (conv_attrs!=NULL){
        stride_height = conv_attrs->strides[0];
        filter_height = conv_attrs->kernel_size[0];
    }
    // Init nnx task(/s can be buffered later)
    ne16_task_init(match_ne16_get_nnx_task(0));
    ne16_task_set_op_to_conv(
        match_ne16_get_nnx_task(0),
        filter_height,
        ctx->pattern_name==depthwise_conv2d,
        stride_height
    );
    ne16_task_set_bits(
        match_ne16_get_nnx_task(0),
        inp_bits,
        out_bits,
        weights_bits
    );
    ne16_task_set_norm_quant(
        match_ne16_get_nnx_task(0),
        (ne16_quant_t) {
            .shift_amount = right_shift,
            .function = activation?quantFunctionRelu:quantFunctionIdentity,
            .flag_rounding = ne16TaskFlagFalse
        }, (ne16_norm_t) {
            .mode  = normMode32Bit,
            .flag_bias  = ne16TaskFlagTrue,
            .flag_shift = ne16TaskFlagFalse
        }
    );
    ne16_task_set_weight_offset(match_ne16_get_nnx_task(0), weightOffsetModeLayerWise, -128);
    cluster_lib_init_dma_transfers();
}

void neural_engine_lib_close(MatchCtx* ctx){
    // Terminate
    cluster_lib_cleanup_dma_transfers();
    ne16_nnx_term(match_ne16_get_nnx_dev());
}

void neural_engine_compute_tile(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_ops = ctx->ops->num_ops;
    int num_tensors = ctx->tensors->num_tensors;
    int IS_A_CONV2D_OP = ctx->pattern_name!=dense;
    int right_shift = ((MatchRightShiftAttrs*)ctx->ops->ops[num_ops-3].attrs)->right_shift;
    int stride_height = 1;
    int filter_height = 1;
    int filter_width = 1;
    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
    // out
    int out_width = 1; // out width
    int out_height = 1; // out height
    int out_ch = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*(IS_A_CONV2D_OP?4:2)+(IS_A_CONV2D_OP?3:1)].size; // out ch
    // inp
    int inp_width = 1; // inp width
    int inp_height = 1; // inp height
    int inp_ch = tensors[0].tiles[L1_SCRATCHPAD*(IS_A_CONV2D_OP?4:2)+(IS_A_CONV2D_OP?3:1)].size; // inp ch
    // pts
    void* activations_pt = tensors[0].pts[L1_SCRATCHPAD];
    void* output_pt = tensors[num_tensors-1].pts[L1_SCRATCHPAD];
    void* weights_pt = tensors[1].pts[L1_SCRATCHPAD];
    void* bnorm_scale_pt = tensors[2].pts[L1_SCRATCHPAD];
    void* bnorm_bias_pt = num_tensors>4? tensors[3].pts[L1_SCRATCHPAD]: NULL;
    if(IS_A_CONV2D_OP){
        MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
        stride_height = conv_attrs->strides[0];
        // they are the same, either 1x1 or 3x3
        filter_height = conv_attrs->kernel_size[0];
        filter_width = filter_height;
        // spatial dims
        out_width = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+2].size; // out width
        out_height = tensors[num_tensors-1].tiles[L1_SCRATCHPAD*4+1].size; // out height
        inp_width = tensors[0].tiles[L1_SCRATCHPAD*4+2].size; // inp width
        inp_height = tensors[0].tiles[L1_SCRATCHPAD*4+1].size; // inp height
        // pad
        pad_top = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
        pad_left = match_get_pad_x_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
        pad_bottom = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+1]));
        pad_right = match_get_pad_y_of_tile(&(tensors[0].tiles[L1_SCRATCHPAD*4+2]));
    }
    if(stride_height==1){
        ne16_task_set_dims(
            match_ne16_get_nnx_task(0),
            inp_width, inp_ch,
            inp_width*inp_ch, inp_ch, out_height,
            out_width, out_ch, out_width*out_ch,
            out_ch, pad_top, pad_bottom,
            pad_left, pad_right
        );
        ne16_task_set_addr_conv(
            match_ne16_get_nnx_task(0),
            activations_pt,
            inp_width, inp_ch, pad_top, pad_left,
            output_pt,
            weights_pt
        );
        ne16_task_set_addr_norm_quant(
            match_ne16_get_nnx_task(0),
            bnorm_scale_pt,
            0X0, // why?
            bnorm_bias_pt
        );
        ne16_nnx_dispatch_wait(match_ne16_get_nnx_dev());
        ne16_nnx_dispatch(
            match_ne16_get_nnx_dev(),
            match_ne16_get_nnx_task(0)
        );
    }
    else{
        ne16_task_set_dims_stride2x2(
            match_ne16_get_nnx_task(0),
            inp_height, inp_width, inp_ch,
            inp_width*inp_ch, inp_ch,
            out_height, out_width, out_ch,
            out_width*out_ch, out_ch,
            filter_height, filter_width,
            pad_top, pad_bottom,
            pad_left, pad_right
        );
        ne16_task_set_addr_conv(
            match_ne16_get_nnx_task(0),
            activations_pt,
            inp_width, inp_ch,
            pad_top, pad_left,
            output_pt,
            weights_pt
        );
        ne16_task_set_addr_norm_quant(
            match_ne16_get_nnx_task(0),
            bnorm_scale_pt,
            0x0, // why?
            bnorm_bias_pt
        );
        ne16_nnx_dispatch_stride2x2(
            match_ne16_get_nnx_dev(),
            match_ne16_get_nnx_task(0),
            inp_width, inp_ch,
            out_height, out_width,
            out_ch, filter_height,
            filter_width
        );
    }
}

void wait_neural_engine_compute(MatchCtx* ctx) {
    while (!ne16_nnx_resolve_check(match_ne16_get_nnx_dev(), match_ne16_get_nnx_task(0)))
      ;
}