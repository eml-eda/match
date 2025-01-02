#ifndef __MATCH_CTX_H__
#define __MATCH_CTX_H__
#include <match/utils.h>

typedef enum{
    MATCH_OP_CONV2D,
    MATCH_OP_BIAS_ADD,
    MATCH_OP_ADD,
    MATCH_OP_MULTIPLY,
    MATCH_OP_RELU,
}MATCH_OPS_CODE;

typedef struct{
    int size;
    int dynamic;
    int curr_size;
    int global_idx;
}MatchDim;

typedef struct MatchDims_t{
    int num_dims;
    char** dims_names;
    MatchDim (*get_dim)(struct MatchDims_t *,const char*);
    MatchDim* dims;
}MatchDims;

typedef struct{
    MatchDim* dim;
    int size;
    int start_idx;
}MatchTensorTile;

typedef struct{
    MatchTensorTile** tiles;
    unsigned int base_pt;
    unsigned int* pts;
    int num_tiles;
    int curr_tile;
    int num_dims;
    int bits;
}MatchVarTensor;

typedef struct{
    MatchTensorTile** tiles;
    unsigned int base_pt;
    unsigned int* pts;
    int num_tiles;
    int curr_tile;
    int num_dims;
    int bits;
}MatchConstTensor;

typedef struct{
    MatchTensorTile** tiles;
    unsigned int base_pt;
    unsigned int* pts;
    int num_tiles;
    int curr_tile;
    int num_dims;
    int bits;
}MatchOutputTensor;

typedef struct MatchVars_t{
    // vars can be 3D,4D etc.
    // var types is used to signal this
    int num_vars;
    char** vars_names;
    MatchVarTensor* (*get_var)(struct MatchVars_t *,const char*);
    MatchVarTensor* tensors;
}MatchVars;

typedef struct MatchConsts_t{
    int num_consts;
    char** consts_names;
    MatchConstTensor* (*get_const)(struct MatchConsts_t *,const char*);
    MatchConstTensor* tensors;
}MatchConsts;

typedef struct MatchOutputs_t{
    int num_outputs;
    char** outputs_names;
    MatchOutputTensor* (*get_out)(struct MatchOutputs_t *,const char*);
    MatchOutputTensor* tensors;
}MatchOutputs;

typedef struct{
    int idx;
    int strides[2];
    int dilation[2];
    int padding[4];
    int kernel_size[2];
    int depthwise;
    int groups;
}MatchConv2DAttrs;

typedef struct{
    int idx;
    int axis;
}MatchBiasAddAttrs;

typedef struct{
    int idx;
}MatchReLUAttrs;

typedef struct{
    int op_code;
    // MatchConv2DAttrs or others...
    void* attrs;
}MatchOp;

typedef struct MatchOps_t{
    int num_ops;
    char** ops_names;
    MatchOp* (*get_op)(struct MatchOps_t *,const char*);
    MatchOp* ops;
}MatchOps;

typedef struct{
    void* ctx_extension;

    MatchVars* vars;
    
    MatchConsts* consts;

    MatchOutputs* outputs;
    
    MatchOps* ops;

    MatchDims* dims;

    int pattern_family;
    int pattern_name;
}MatchCtx;

MatchVarTensor* default_match_ctx_get_var(struct MatchVars_t *self,const char *name);

MatchConstTensor* default_match_ctx_get_const(struct MatchConsts_t *self,const char *name);

MatchOutputTensor* default_match_ctx_get_out(struct MatchOutputs_t *self,const char *name);

MatchOp* default_match_ctx_get_op(struct MatchOps_t *self,const char *name);

MatchDim* default_match_ctx_get_dim(struct MatchDims_t *self,const char *name);

#endif