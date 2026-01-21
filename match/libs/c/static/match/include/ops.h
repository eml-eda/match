#ifndef __MATCH_OPS_H__
#define __MATCH_OPS_H__

typedef enum {
    MATCH_OP_DENSE = 0,
    MATCH_OP_RELU = 1,
    MATCH_OP_ADD = 2,
    MATCH_OP_BIAS_ADD = 3,
    MATCH_OP_CONV3D = 4,
    MATCH_OP_CONV2D = 5,
    MATCH_OP_CONV1D = 6,
    MATCH_OP_CAST = 7,
    MATCH_OP_CLIP = 8,
    MATCH_OP_RIGHT_SHIFT = 9,
    MATCH_OP_MULTIPLY = 10,
    MATCH_OP_CONV2D_TRANSPOSE = 11,
    MATCH_OP_SQRT = 12,
    MATCH_OP_DIVIDE = 13,
    MATCH_OP_REPEAT = 14,
    MATCH_OP_RESHAPE = 15,
    MATCH_OP_SUM = 16,
    MATCH_OP_INSTANCE_NORM = 17,
    MATCH_OP_SUBTRACT = 18,
    MATCH_OP_RSQRT = 19,
    MATCH_OP_BATCH_MATMUL = 20,
} MATCH_OPS_CODE;

// Attributes
typedef struct {
    int idx;
    int strides[3];
    int dilation[3];
    int padding[6];
    int kernel_size[3];
    int depthwise;
    int groups;
    const char* data_layout;
    const char* kernel_layout;
} MatchConv3DAttrs;

typedef struct {
    int idx;
    int strides[2];
    int dilation[2];
    int padding[4];
    int kernel_size[2];
    int depthwise;
    int groups;
    const char* data_layout;
    const char* kernel_layout;
} MatchConv2DAttrs;

typedef struct {
    int idx;
    int strides[2];
    int dilation[2];
    int padding[4];
    int output_padding[2];
    int kernel_size[2];
    int depthwise;
    int groups;
    const char* data_layout;
    const char* kernel_layout;
} MatchConv2DTransposeAttrs;

typedef struct {
    int idx;
    int inp_features;
    int out_features;
} MatchDenseAttrs;

typedef struct {
    int idx;
    int right_shift;
} MatchRightShiftAttrs;

typedef struct {
    int idx;
    int clip_min;
    int clip_max;
} MatchClipAttrs;

typedef struct {
    int idx;
} MatchCastAttrs;

typedef struct {
    int idx;
    int axis;
    int bias;
}MatchBiasAddAttrs;

typedef struct {
    int idx;
    int axis;
    int multiplier;
}MatchMultiplyAttrs;

typedef struct {
    int idx;
    int axis;
    int adder;
}MatchAddAttrs;

typedef struct {
    int idx;
    int axis;    // axis along which subtraction constant/broadcast applies
    int subtractor; // immediate scalar if const 0-dim
} MatchSubtractAttrs;

typedef struct {
    int idx;
} MatchReLUAttrs;

typedef struct {
    int idx;
    int strides[1];
    int dilation[1];
    int padding[2];
    int kernel_size[1];
    int depthwise;
    int groups;
    const char* data_layout;
    const char* kernel_layout;
} MatchConv1DAttrs;

typedef struct {
    int idx;
} MatchSqrtAttrs;

typedef struct {
    int idx; // rsqrt has no additional attributes
} MatchRsqrtAttrs;

typedef struct {
    int idx;
    int axis; // -1 means last axis
} MatchDivideAttrs;

typedef struct {
    int idx;
    int repeats; // number of repeats
    int axis;    // axis along which to repeat
} MatchRepeatAttrs;

typedef struct {
    int idx;
    int axis_count;   // number of valid axes in axes[]
    int axes[4];      // supports up to 4 reduction axes
    int keepdims;     // boolean
} MatchSumAttrs;

typedef struct {
    int idx;
    int newshape_rank;  // number of dimensions in newshape
    int newshape[8];     // supports reshape up to rank 8
} MatchReshapeAttrs;

typedef struct {
    int idx;
    float epsilon;
    float momentum;
} MatchInstanceNormAttrs;

typedef struct {
    // (B, M, N) @ (B, N, K) = (B, M, K)
    int idx;
    int dim_b; 
    int dim_m;
    int dim_n;
    int dim_k;
} MatchBatchMatMulAttrs;

#endif