#ifdef __pulp_cluster__
#ifndef __PULP_KERNELS_FP16_VECTOR_H__
#define __PULP_KERNELS_FP16_VECTOR_H__

typedef fp16 v2f16 __attribute__((vector_size (4)));

static inline v2f16 vfpack(fp16 a, fp16 b) {
  v2f16 result = (v2f16) {0,0};
  asm ("pv.pack.h %0, %1, %2" : "=f" (result): "f" (a), "f" (b) : );
  return result;
}

static inline fp16 vfdotp(v2f16 a, v2f16 b) {
  fp16 result;
  //asm ("vfdotp.h %0, %1, %2" : "=f" (result): "f" (a), "f" (b) : );
  return result;
}

static inline v2f16 vfadd(v2f16 a, v2f16 b) {
  v2f16 result;
  asm ("vfadd.h %0, %1, %2" : "=f" (result): "f" (a), "f" (b) : );
  return result;
}

#endif // __PULP_KERNELS_FP16_VECTOR_H__
#endif // __pulp_cluster__

