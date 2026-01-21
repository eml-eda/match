#ifndef __MATCH_CTX_H__
#define __MATCH_CTX_H__

#include <match/ops.h>
#include <match/utils.h>

// Enums

typedef enum {
    MATCH_VAR_TENSOR,
    MATCH_CONST_TENSOR,
    MATCH_OUT_TENSOR,
} MATCH_TENSOR_TYPE;

typedef enum {
    MATCH_SW_LOAD_TENSOR,
    MATCH_SW_STORE_TENSOR,
} MATCH_MEM_OPS_TYPE;

// Tensor Dimensions

typedef struct {
    int size;
    int dynamic;
    int curr_size;
    int curr_max_size;
    int global_idx;
    float idx_remainder;
} MatchDim;

typedef struct MatchDims_t {
    int num_dims;
    char** dims_names;
    MatchDim (*get_dim)(struct MatchDims_t*, const char*);
    int (*get_dim_idx)(struct MatchDims_t*, const char*);
    MatchDim* dims;
} MatchDims;

// Tensors

typedef struct {
    MatchDim* dim;
    int size;
    float idx_remainder;
    int max_size;
    int start_idx;
    int curr_idx;
} MatchTensorTile;

typedef struct {
    MatchTensorTile* tiles;
    void* base_pt;
    unsigned int* pts;
    void* pt;
    int num_tiles;
    int tensor_type;
    int curr_tile;
    int num_dims;
    int bits;
} MatchTensor;

typedef struct MatchTensors_t {
    int num_tensors;
    char** tensors_names;
    MatchTensor* (*get_tensor)(struct MatchTensors_t*, const char*);
    int (*get_tensor_idx)(struct MatchTensors_t*, const char*);
    MatchTensor* tensors;
} MatchTensors;

// Operations

typedef struct {
    int op_code;
    // MatchConv2DAttrs or others...
    void* attrs;
} MatchOp;

typedef struct MatchOps_t {
    int num_ops;
    char** ops_names;
    MatchOp* (*get_op)(struct MatchOps_t*, const char*);
    int (*get_op_idx)(struct MatchOps_t*, const char*);
    MatchOp* ops;
} MatchOps;

// Context

typedef struct {
    void* ctx_extension;

    MatchTensors* tensors;

    MatchOps* ops;

    MatchDims* dims;

    int exec_module;
    int pattern_name;
} MatchCtx;

typedef struct match_runtime_ctx_t {
    int status;
} match_runtime_ctx;

// Methods

MatchTensor* default_match_ctx_get_tensor(struct MatchTensors_t* self, const char* name);

int default_match_ctx_get_tensor_idx(struct MatchTensors_t* self, const char* name);

MatchOp* default_match_ctx_get_op(struct MatchOps_t* self, const char* name);

int default_match_ctx_get_op_idx(struct MatchOps_t* self, const char* name);

MatchDim* default_match_ctx_get_dim(struct MatchDims_t* self, const char* name);

int default_match_ctx_get_dim_idx(struct MatchDims_t* self, const char* name);

// Tile Utils

inline int match_get_pad_x_of_tile(MatchTensorTile* tile) {
    return -tile->curr_idx > 0 ? -tile->curr_idx : 0;
}

inline int match_get_pad_y_of_tile(MatchTensorTile* tile) {
    // TODO: check which version is correct
    // upstream padding calculation
    int remaining = tile->dim->size - (tile->dim->curr_max_size + tile->curr_idx);
    return remaining < 0 ? -remaining : 0;
    // odl padding calculation
    // return tile->dim->size-(tile->max_size+tile->curr_idx)<0?-(tile->dim->size-(tile->max_size+tile->curr_idx)):0;
}

#endif