#ifdef __pulp_cluster__
#ifndef __PULP_KERNELS_FP16_DEFINES_H__
#define __PULP_KERNELS_FP16_DEFINES_H__

#include <pulp.h>

#define barrier         synch_barrier
#define nthreads        (get_core_num())
#define tid             (rt_core_id())

#define min(a,b)        ( (a) < (b) ? (a) : (b) )
#define max(a,b)        ( (a) > (b) ? (a) : (b) )
#define abs(a)          ( (a) < 0 ? -(a) : (a) )

#define log2(x)         __builtin_pulp_fl1(x)

typedef float16 fp16;


// Conv2D Feature map indexers
static inline int idx_NHWC(int y, int x, int c, int dim_y, int dim_x, int dim_c) {
    return (y * dim_x + x) * dim_c + c;
}
static inline int idx_NCHW(int y, int x, int c, int dim_y, int dim_x, int dim_c) {
    return (c * dim_y + y) * dim_x + x;
}

//  Conv2D Filter indexers
static inline int idx_HWIO(int fy, int fx, int ic, int oc, int dim_fy, int dim_fx, int dim_ic, int dim_oc) {
    return ((fy * dim_fx + fx) * dim_ic + ic) * dim_oc + oc;
}
static inline int idx_OIHW(int fy, int fx, int ic, int oc, int dim_fy, int dim_fx, int dim_ic, int dim_oc) {
    return ((oc * dim_ic + ic) * dim_fy + fy) * dim_fx + fx;
}
static inline int idx_OHWI(int fy, int fx, int ic, int oc, int dim_fy, int dim_fx, int dim_ic, int dim_oc) {
    return ((oc * dim_fy + fy) * dim_fx + fx) * dim_ic + ic;
}

// Dense weight indexers
static inline int idx_OI(int o, int i, int dim_o, int dim_i) { // NC in TVM
    return o * dim_i + i;
}
static inline int idx_IO(int i, int o, int dim_i, int dim_o) { // CN in TVM
    return i * dim_o + o;
}

#endif // __PULP_KERNELS_FP16_DEFINES_H__
#endif // __pulp_cluster__