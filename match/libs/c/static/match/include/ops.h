#ifndef __MATCH_OPS_H__
#define __MATCH_OPS_H__

typedef enum{
    MATCH_OP_CONV2D = 0,
    MATCH_OP_BIAS_ADD = 1,
    MATCH_OP_ADD = 2,
    MATCH_OP_MULTIPLY = 3,
    MATCH_OP_RELU = 4,
    MATCH_OP_CLIP = 5,
    MATCH_OP_DENSE = 6,
    MATCH_OP_RIGHT_SHIFT = 7,
    MATCH_OP_CAST = 8,
    MATCH_OP_CONV1D = 9,
}MATCH_OPS_CODE;

#endif