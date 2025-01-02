#ifndef __MATCH_NODE_PARAMS_tvmgen_default_match_main_0_H__
#define __MATCH_NODE_PARAMS_tvmgen_default_match_main_0_H__

#include <match/ctx.h>
#include <match/utils.h>
#include <neoptex.h>
#include <neoptexvectcpu/neoptexvectcpu.h>
#include <nodes/default/main_0_data.h>

// DIMS
extern const char* main_0_dims_names_[];
extern MatchDim main_0_dims_[10];
extern MatchDim* main_0_FunctionVar_0_0_dim_0_dim;
extern MatchDim* main_0_FunctionVar_0_0_dim_1_dim;
extern MatchDim* main_0_FunctionVar_0_0_dim_2_dim;
extern MatchDim* main_0_FunctionVar_0_0_dim_3_dim;
extern MatchDim* main_0_FunctionVar_0_1_dim_0_dim;
extern MatchDim* main_0_FunctionVar_0_1_dim_2_dim;
extern MatchDim* main_0_FunctionVar_0_1_dim_3_dim;
extern MatchDim* main_0_conv2d_out_h_dim;
extern MatchDim* main_0_conv2d_out_w_dim;
extern MatchDim* main_0_FunctionVar_0_2_dim_0_dim;
extern MatchDims main_0_dims_cnt_;

// TILES
extern MatchTensorTile main_0_FunctionVar_0_0_tiles_[1][4];
extern MatchTensorTile** main_0_FunctionVar_0_0_tiles;
extern MatchTensorTile main_0_FunctionVar_0_1_tiles_[1][4];
extern MatchTensorTile** main_0_FunctionVar_0_1_tiles;
extern MatchTensorTile main_0_FunctionVar_0_2_tiles_[1][1];
extern MatchTensorTile** main_0_FunctionVar_0_2_tiles;
extern MatchTensorTile main_0_relu_tiles_[1][4];
extern MatchTensorTile** main_0_relu_tiles;
extern MatchTensorTile main_0_conv2d_tiles_[1][4];
extern MatchTensorTile** main_0_conv2d_tiles;
extern MatchTensorTile main_0_bias_add_tiles_[1][4];
extern MatchTensorTile** main_0_bias_add_tiles;
// VARS
extern const char* main_0_vars_names_[];
extern MatchVarTensor main_0_vars_[1];
extern MatchVarTensor* main_0_FunctionVar_0_0_var;
extern MatchVars main_0_vars_cnt_;

// CONSTS
extern const char* main_0_consts_names_[];
extern MatchConstTensor main_0_consts_[2];
extern MatchConstTensor* main_0_FunctionVar_0_1_const;
extern MatchConstTensor* main_0_FunctionVar_0_2_const;
extern MatchConsts main_0_consts_cnt_;

// OUTPUTS
extern const char* main_0_outputs_names_[];
extern MatchOutputTensor main_0_outputs_[1];
extern MatchOutputTensor* main_0_relu_out;
extern MatchOutputs main_0_outputs_cnt_;

// ops
extern const char* main_0_ops_names_[];
extern MatchOp main_0_ops_[3];
extern MatchConv2DAttrs main_0_op_conv2d_attrs_;
extern MatchConv2DAttrs* main_0_conv2d_attrs;
extern MatchOp* main_0_conv2d_op;
extern MatchBiasAddAttrs main_0_op_bias_add_attrs_;
extern MatchBiasAddAttrs* main_0_bias_add_attrs;
extern MatchOp* main_0_bias_add_op;
extern MatchReLUAttrs main_0_op_relu_attrs_;
extern MatchReLUAttrs* main_0_relu_attrs;
extern MatchOp* main_0_relu_op;
extern MatchOps main_0_ops_cnt_;

extern MatchCtx main_0_ctx_;

extern MatchCtx* main_0_ctx;


// loops iters counters
extern int loop_OX_3_iter;
extern int loop_OX_2_iter;
extern int loop_OX_1_iter;
extern int loop_OX_iter;
extern int loop_OY_4_iter;
extern int loop_OY_3_iter;
extern int loop_OY_2_iter;
extern int loop_OY_1_iter;
extern int loop_FX_iter;
extern int loop_FY_iter;
extern int loop_C_iter;

// loops iters counters
inline void update_FunctionVar_0_0_dim_2(){
    main_0_FunctionVar_0_0_dim_2_dim->global_idx = 
    (2*main_0_conv2d_out_h_dim->global_idx)
     + (1*main_0_FunctionVar_0_1_dim_2_dim->global_idx) -1
    ;
}
inline void set_FunctionVar_0_0_dim_2(){
    main_0_FunctionVar_0_0_dim_2_dim->curr_size = 
    (2*main_0_conv2d_out_h_dim->curr_size)
     + (1*main_0_FunctionVar_0_1_dim_2_dim->curr_size)
    ;
    // update_FunctionVar_0_0_dim_2();
}
inline void update_FunctionVar_0_0_dim_3(){
    main_0_FunctionVar_0_0_dim_3_dim->global_idx = 
    (2*main_0_conv2d_out_w_dim->global_idx)
     + (1*main_0_FunctionVar_0_1_dim_3_dim->global_idx) -1
    ;
}

inline void set_FunctionVar_0_0_dim_3(){
    main_0_FunctionVar_0_0_dim_3_dim->curr_size = 
    (2*main_0_conv2d_out_w_dim->curr_size)
     + (1*main_0_FunctionVar_0_1_dim_3_dim->curr_size)
    ;
    // update_FunctionVar_0_0_dim_3();
}

inline void loop_OX_3_set(){
    loop_OX_3_iter = 0;
    main_0_conv2d_out_w_dim->curr_size = 16;
    // set_FunctionVar_0_0_dim_3();
}
inline int loop_OX_3_reset(){
    // loop_OX_3_iter = 0;
    main_0_conv2d_out_w_dim->global_idx -= 2*8;
    return 0;
    // update_FunctionVar_0_0_dim_3();
}
inline void loop_OX_3_update(){
    loop_OX_3_iter += 1;
    main_0_conv2d_out_w_dim->global_idx += 8;
    // update_FunctionVar_0_0_dim_3();
}
inline int loop_OX_3_end(){
    return loop_OX_3_iter >= 2 ? loop_OX_3_reset() : 1;
}
inline void loop_OX_2_set(){
    loop_OX_2_iter = 0;
    main_0_conv2d_out_w_dim->curr_size = 8;
    // set_FunctionVar_0_0_dim_3();
}
inline int loop_OX_2_reset(){
    // loop_OX_2_iter = 0;
    main_0_conv2d_out_w_dim->global_idx -= 2*4;
    return 0;
    // update_FunctionVar_0_0_dim_3();
}
inline void loop_OX_2_update(){
    loop_OX_2_iter += 1;
    main_0_conv2d_out_w_dim->global_idx += 4;
    // update_FunctionVar_0_0_dim_3();
}
inline int loop_OX_2_end(){
    return loop_OX_2_iter >= 2 ? loop_OX_2_reset() : 1;
}
inline void loop_OX_1_set(){
    loop_OX_1_iter = 0;
    main_0_conv2d_out_w_dim->curr_size = 4;
    // set_FunctionVar_0_0_dim_3();
}
inline int loop_OX_1_reset(){
    // loop_OX_1_iter = 0;
    main_0_conv2d_out_w_dim->global_idx -= 2*2;
    return 0;
    // update_FunctionVar_0_0_dim_3();
}
inline void loop_OX_1_update(){
    loop_OX_1_iter += 1;
    main_0_conv2d_out_w_dim->global_idx += 2;
    // update_FunctionVar_0_0_dim_3();
}
inline int loop_OX_1_end(){
    return loop_OX_1_iter >= 2? loop_OX_1_reset() :  1;
}
inline void loop_OX_set(){
    loop_OX_iter = 0;
    main_0_conv2d_out_w_dim->curr_size = 2;
    // set_FunctionVar_0_0_dim_3();
}
inline int loop_OX_reset(){
    // loop_OX_iter = 0;
    main_0_conv2d_out_w_dim->global_idx -= 2;
    return 0;
    // update_FunctionVar_0_0_dim_3();
}
inline void loop_OX_update(){
    loop_OX_iter += 1;
    main_0_conv2d_out_w_dim->global_idx += 1;
    // update_FunctionVar_0_0_dim_3();
}
inline int loop_OX_end(){
    return loop_OX_iter >= 2 ? loop_OX_reset() :  1;
}
inline void loop_OY_4_set(){
    loop_OY_4_iter = 0;
    main_0_conv2d_out_h_dim->curr_size = 16;
    // set_FunctionVar_0_0_dim_2();
}
inline int loop_OY_4_reset(){
    // loop_OY_4_iter = 0;
    main_0_conv2d_out_h_dim->global_idx -= 2*8;
    return 0;
    // update_FunctionVar_0_0_dim_2();
}
inline void loop_OY_4_update(){
    loop_OY_4_iter += 1;
    main_0_conv2d_out_h_dim->global_idx += 8;
    // update_FunctionVar_0_0_dim_2();
}
inline int loop_OY_4_end(){
    return loop_OY_4_iter >= 2? loop_OY_4_reset() : 1;
}
inline void loop_OY_3_set(){
    loop_OY_3_iter = 0;
    main_0_conv2d_out_h_dim->curr_size = 8;
    // set_FunctionVar_0_0_dim_2();
}
inline int loop_OY_3_reset(){
    // loop_OY_3_iter = 0;
    main_0_conv2d_out_h_dim->global_idx -= 2*4;
    return 0;
    // update_FunctionVar_0_0_dim_2();
}
inline void loop_OY_3_update(){
    loop_OY_3_iter += 1;
    main_0_conv2d_out_h_dim->global_idx += 4;
    // update_FunctionVar_0_0_dim_2();
}
inline int loop_OY_3_end(){
    return loop_OY_3_iter >= 2? loop_OY_3_reset() : 1;
}
inline void loop_OY_2_set(){
    loop_OY_2_iter = 0;
    main_0_conv2d_out_h_dim->curr_size = 4;
    // set_FunctionVar_0_0_dim_2();
}
inline int loop_OY_2_reset(){
    // loop_OY_2_iter = 0;
    main_0_conv2d_out_h_dim->global_idx -= 2*2;
    return 0;
    // update_FunctionVar_0_0_dim_2();
}
inline void loop_OY_2_update(){
    loop_OY_2_iter += 1;    
    main_0_conv2d_out_h_dim->global_idx += 2;
    // update_FunctionVar_0_0_dim_2();
}
inline int loop_OY_2_end(){
    return loop_OY_2_iter >= 2? loop_OY_2_reset() : 1;
}
inline void loop_OY_1_set(){
    loop_OY_1_iter = 0;
    main_0_conv2d_out_h_dim->curr_size = 2;
    // set_FunctionVar_0_0_dim_2();
}
inline int loop_OY_1_reset(){
    // loop_OY_1_iter = 0;
    main_0_conv2d_out_h_dim->global_idx -= 2;
    return 0;
    // update_FunctionVar_0_0_dim_2();
}
inline void loop_OY_1_update(){
    loop_OY_1_iter += 1;
    main_0_conv2d_out_h_dim->global_idx += 1;
    // update_FunctionVar_0_0_dim_2();
}
inline int loop_OY_1_end(){
    return loop_OY_1_iter >= 2? loop_OY_1_reset() : 1;
}
inline void loop_FX_set(){
    loop_FX_iter = 0;
    main_0_FunctionVar_0_1_dim_3_dim->curr_size = 3;
    // set_FunctionVar_0_0_dim_3();
}
inline int loop_FX_reset(){
    // loop_FX_iter = 0;
    main_0_FunctionVar_0_1_dim_3_dim->global_idx -= 3;
    return 0;
    // update_FunctionVar_0_0_dim_3();
}
inline void loop_FX_update(){
    loop_FX_iter += 1;
    main_0_FunctionVar_0_1_dim_3_dim->global_idx += 1;
    // update_FunctionVar_0_0_dim_3();
}
inline int loop_FX_end(){
    return loop_FX_iter >= 3 ? loop_FX_reset() : 1;
}
inline void loop_FY_set(){
    loop_FY_iter = 0;
    main_0_FunctionVar_0_1_dim_2_dim->curr_size = 3;
    // set_FunctionVar_0_0_dim_2();
}
inline int loop_FY_reset(){
    // loop_FY_iter = 0;
    main_0_FunctionVar_0_1_dim_2_dim->global_idx -= 3;
    return 0;
    // update_FunctionVar_0_0_dim_2();
}
inline void loop_FY_update(){
    loop_FY_iter += 1;
    main_0_FunctionVar_0_1_dim_2_dim->global_idx += 1;
    // update_FunctionVar_0_0_dim_2();
}
inline int loop_FY_end(){
    return loop_FY_iter >= 3 ? loop_FY_reset() : 1;
}
inline void loop_C_set(){
    loop_C_iter = 0;
    main_0_FunctionVar_0_0_dim_1_dim->curr_size = 3;
}
inline int loop_C_reset(){
    // loop_C_iter = 0;
    main_0_FunctionVar_0_0_dim_1_dim->global_idx -= 3;
    return 0;
}
inline void loop_C_update(){
    loop_C_iter += 1;
    main_0_FunctionVar_0_0_dim_1_dim->global_idx += 1;
}
inline int loop_C_end(){
    return loop_C_iter >= 3 ? loop_C_reset() : 1;
}


#endif