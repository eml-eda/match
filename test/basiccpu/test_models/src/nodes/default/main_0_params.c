#include <nodes/default/main_0_params.h>

// DIMS
const char* main_0_dims_names_[] = {
    "FunctionVar_0_0_dim_0"
    , "FunctionVar_0_0_dim_1"
    , "FunctionVar_0_0_dim_2"
    , "FunctionVar_0_0_dim_3"
    , "FunctionVar_0_1_dim_0"
    , "FunctionVar_0_1_dim_2"
    , "FunctionVar_0_1_dim_3"
    , "conv2d_out_h"
    , "conv2d_out_w"
    , "FunctionVar_0_2_dim_0"
};
MatchDim main_0_dims_[10] = {
    (MatchDim){
        .size = 1,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 3,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 32,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 32,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 1,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 3,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 3,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 16,
        .dynamic = 0,//0,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 16,
        .dynamic = 0,//0,
        .global_idx = 0,
        .curr_size = 0,
    }
    , (MatchDim){
        .size = 1,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 0,
    }
};

MatchDim* main_0_FunctionVar_0_0_dim_0_dim = &(main_0_dims_[0]);
MatchDim* main_0_FunctionVar_0_0_dim_1_dim = &(main_0_dims_[1]);
MatchDim* main_0_FunctionVar_0_0_dim_2_dim = &(main_0_dims_[2]);
MatchDim* main_0_FunctionVar_0_0_dim_3_dim = &(main_0_dims_[3]);
MatchDim* main_0_FunctionVar_0_1_dim_0_dim = &(main_0_dims_[4]);
MatchDim* main_0_FunctionVar_0_1_dim_2_dim = &(main_0_dims_[5]);
MatchDim* main_0_FunctionVar_0_1_dim_3_dim = &(main_0_dims_[6]);
MatchDim* main_0_conv2d_out_h_dim = &(main_0_dims_[7]);
MatchDim* main_0_conv2d_out_w_dim = &(main_0_dims_[8]);
MatchDim* main_0_FunctionVar_0_2_dim_0_dim = &(main_0_dims_[9]);

MatchDims main_0_dims_cnt_ = (MatchDims){
    .num_dims = 10,
    .dims_names = main_0_ops_names_,
    .get_dim = default_match_ctx_get_dim,
    .dims = main_0_dims_
};


// TILES
MatchTensorTile main_0_FunctionVar_0_0_tiles_[1][4] = {
    {
        (MatchTensorTile){
            .dim = &(main_0_dims_[0]),
            .size = 1,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[1]),
            .size = 3,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[2]),
            .size = 32,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[3]),
            .size = 32,
            .start_idx = 0
        }
    }
};
MatchTensorTile** main_0_FunctionVar_0_0_tiles = (MatchTensorTile**)main_0_FunctionVar_0_0_tiles_;
MatchTensorTile main_0_FunctionVar_0_1_tiles_[1][4] = {
    {
        (MatchTensorTile){
            .dim = &(main_0_dims_[4]),
            .size = 1,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[1]),
            .size = 3,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[5]),
            .size = 3,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[6]),
            .size = 3,
            .start_idx = 0
        }
    }
};
MatchTensorTile** main_0_FunctionVar_0_1_tiles = (MatchTensorTile**)main_0_FunctionVar_0_1_tiles_;
MatchTensorTile main_0_FunctionVar_0_2_tiles_[1][1] = {
    {
        (MatchTensorTile){
            .dim = &(main_0_dims_[9]),
            .size = 1,
            .start_idx = 0
        }
    }
};
MatchTensorTile** main_0_FunctionVar_0_2_tiles = (MatchTensorTile**)main_0_FunctionVar_0_2_tiles_;
MatchTensorTile main_0_relu_tiles_[1][4] = {
    {
        (MatchTensorTile){
            .dim = &(main_0_dims_[0]),
            .size = 1,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[4]),
            .size = 1,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[7]),
            .size = 16,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[8]),
            .size = 16,
            .start_idx = 0
        }
    }
};
MatchTensorTile** main_0_relu_tiles = (MatchTensorTile**)main_0_relu_tiles_;
MatchTensorTile main_0_conv2d_tiles_[1][4] = {
    {
        (MatchTensorTile){
            .dim = &(main_0_dims_[0]),
            .size = 1,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[4]),
            .size = 1,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[7]),
            .size = 16,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[8]),
            .size = 16,
            .start_idx = 0
        }
    }
};
MatchTensorTile** main_0_conv2d_tiles = (MatchTensorTile**)main_0_conv2d_tiles_;
MatchTensorTile main_0_bias_add_tiles_[1][4] = {
    {
        (MatchTensorTile){
            .dim = &(main_0_dims_[0]),
            .size = 1,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[4]),
            .size = 1,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[7]),
            .size = 16,
            .start_idx = 0
        }
        , (MatchTensorTile){
            .dim = &(main_0_dims_[8]),
            .size = 16,
            .start_idx = 0
        }
    }
};
MatchTensorTile** main_0_bias_add_tiles = (MatchTensorTile**)main_0_bias_add_tiles_;
// VARS
const char* main_0_vars_names_[] = {
    "FunctionVar_0_0"
};

MatchVarTensor main_0_vars_[1] = {
    (MatchVarTensor){
        .num_dims = 4,
        .num_tiles = 1,
        .curr_tile = 0,
        .bits = 8,
        .base_pt = 0x0,
        .pts = {0},
        .tiles = main_0_FunctionVar_0_0_tiles_
    }
};

MatchVarTensor* main_0_FunctionVar_0_0_var = &(main_0_vars_[0]);

MatchVars main_0_vars_cnt_ = (MatchVars){
    .num_vars = 1,
    .vars_names = main_0_vars_names_,
    .get_var = default_match_ctx_get_var,
    .tensors = main_0_vars_
};

// CONSTS
const char* main_0_consts_names_[] = {
    "FunctionVar_0_1"
    , "FunctionVar_0_2"
};

MatchConstTensor main_0_consts_[2] = {
    (MatchConstTensor){
        .num_dims = 4,
        .num_tiles = 1,
        .curr_tile = 0,
        .bits = 8,
        .base_pt = 0x0,
        .pts = {0},
        .tiles = main_0_FunctionVar_0_1_tiles_
    }
    , (MatchConstTensor){
        .num_dims = 1,
        .num_tiles = 1,
        .curr_tile = 0,
        .bits = 32,
        .base_pt = 0x0,
        .pts = {0},
        .tiles = main_0_FunctionVar_0_2_tiles_
    }
};
MatchConstTensor* main_0_FunctionVar_0_1_const = &(main_0_consts_[0]);
MatchConstTensor* main_0_FunctionVar_0_2_const = &(main_0_consts_[1]);
MatchConsts main_0_consts_cnt_ = (MatchConsts){
    .num_consts = 2,
    .consts_names = main_0_consts_names_,
    .get_const = default_match_ctx_get_const,
    .tensors = main_0_consts_
};

// OUTPUTS
const char* main_0_outputs_names_[] = {
    "relu"
};
MatchOutputTensor main_0_outputs_[1] = {
    (MatchOutputTensor){
        .num_dims = 4,
        .num_tiles = 1,
        .curr_tile = 0,
        .bits = 32,
        .base_pt = 0x0,
        .pts = {0},
        .tiles = main_0_relu_tiles_
    }
};
MatchOutputTensor* main_0_relu_out = &(main_0_outputs_[0]);

MatchOutputs main_0_outputs_cnt_ = (MatchOutputs){
    .num_outputs = 1,
    .outputs_names = main_0_ops_names_,
    .get_out = default_match_ctx_get_out,
    .tensors = main_0_outputs_
};

// ops
const char* main_0_ops_names_[] = {
    "conv2d"
    , "bias_add"
    , "relu"
};
MatchConv2DAttrs main_0_op_conv2d_attrs_ = (MatchConv2DAttrs){
    .idx = 0
    ,.groups = 1
    ,.depthwise = 0
    ,.strides = {2, 2}
    ,.padding = {1, 1, 1, 1}
    ,.dilation = {1, 1}
    ,.kernel_size = {3, 3}
};
MatchConv2DAttrs* main_0_conv2d_attrs = &(main_0_op_conv2d_attrs_);
MatchBiasAddAttrs main_0_op_bias_add_attrs_ = (MatchBiasAddAttrs){
    .idx = 1
    ,.axis = 0
};
MatchBiasAddAttrs* main_0_bias_add_attrs = &(main_0_op_bias_add_attrs_);
MatchReLUAttrs main_0_op_relu_attrs_ = (MatchReLUAttrs){
    .idx = 2
};
MatchReLUAttrs* main_0_relu_attrs = &(main_0_op_relu_attrs_);
MatchOp main_0_ops_[3] = {
    (MatchOp){
        .op_code = 3,
        .attrs = &main_0_op_conv2d_attrs_
    }
    , (MatchOp){
        .op_code = 2,
        .attrs = &main_0_op_bias_add_attrs_
    }
    , (MatchOp){
        .op_code = 0,
        .attrs = &main_0_op_relu_attrs_
    }
};
MatchOp* main_0_conv2d_op = &(main_0_ops_[0]);
MatchOp* main_0_bias_add_op = &(main_0_ops_[1]);
MatchOp* main_0_relu_op = &(main_0_ops_[2]);

MatchOps main_0_ops_cnt_ = (MatchOps){
    .num_ops = 3,
    .ops_names = main_0_ops_names_,
    .get_op = default_match_ctx_get_op,
    .ops = main_0_ops_
};

MatchCtx main_0_ctx_ = (MatchCtx){
    .ctx_extension = 0x0,
    .vars = &main_0_vars_cnt_,
    .consts = &main_0_consts_cnt_,
    .outputs = &main_0_outputs_cnt_,
    .ops = &main_0_ops_cnt_,
    .dims = &main_0_dims_cnt_,
    .pattern_family = vec_conv,
    .pattern_name = vec_conv
};

MatchCtx* main_0_ctx = &main_0_ctx_;




// loops iters counters
int loop_OX_3_iter = 0;
int loop_OX_2_iter = 0;
int loop_OX_1_iter = 0;
int loop_OX_iter = 0;
int loop_OY_4_iter = 0;
int loop_OY_3_iter = 0;
int loop_OY_2_iter = 0;
int loop_OY_1_iter = 0;
int loop_FX_iter = 0;
int loop_FY_iter = 0;
int loop_C_iter = 0;