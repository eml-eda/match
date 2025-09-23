#include <math.h>

_Float16 my_exp(_Float16 x_f16) {
    float x_f32 = (float)x_f16;
    float y_f32 = expf(x_f32);
    return (_Float16)y_f32;
}