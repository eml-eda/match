#ifdef __pulp_cluster__
#ifndef __PULP_NN_FP16_DEFINES_H__
#define __PULP_NN_FP16_DEFINES_H__

#include <pulp.h>

#define nthreads        (get_core_num())
#define tid             (rt_core_id())

#define min(a,b)        ( (a) < (b) ? (a) : (b) )
#define max(a,b)        ( (a) > (b) ? (a) : (b) )
#define abs(a)          ( (a) < 0 ? -(a) : (a) )

#define log2(x)         __builtin_pulp_fl1(x)

typedef float16 fp16;

#endif // __PULP_NN_FP16_DEFINES_H__
#endif // __pulp_cluster__