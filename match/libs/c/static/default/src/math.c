#include <math.h>

_Float16 my_exp(_Float16 x_f16) {
    float x_f32 = (float)x_f16;
    float y_f32 = expf(x_f32);
    return (_Float16)y_f32;
}

_Float16 my_pow(_Float16 base_fp16, _Float16 exponent_fp16) {
    float base_f32 = (float)base_fp16;
    float exponent_f32 = (float)exponent_fp16;
    float result_f32 = powf(base_f32, exponent_f32);
    return (_Float16)result_f32;
}

_Float16 my_sqrt(_Float16 x_f16) {
    float x_f32 = (float)x_f16;
    float y_f32 = sqrtf(x_f32);
    return (_Float16)y_f32;
}