
#ifdef __spatz__

#include "spatz_kernels/spatz_fp16_defines.h"
#include "spatz_kernels/spatz_fp16_kernels.h"


fp16 spatz_fp16_fdotp(
    const fp16* a, 
    const fp16* b, 
    unsigned int avl
) {
    const unsigned int orig_avl = avl;
    unsigned int vl;

    fp16 red;

    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vmv.s.x v0, zero");

    // Stripmine and accumulate a partial reduced vector
    do {
        // Set the vl
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        // Load chunk a and b
        asm volatile("vle16.v v8,  (%0)" ::"r"(a));
        asm volatile("vle16.v v16, (%0)" ::"r"(b));

        // Multiply and accumulate
        if (avl == orig_avl) {
            asm volatile("vfmul.vv v24, v8, v16");
        } else {
            asm volatile("vfmacc.vv v24, v8, v16");
        }

        // Bump pointers
        a += vl;
        b += vl;
        avl -= vl;
    } while (avl > 0);

    // Reduce and return
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(orig_avl));
    asm volatile("vfredusum.vs v0, v24, v0");
    asm volatile("vfmv.f.s %0, v0" : "=f"(red));
    return red;
}


#endif // __spatz__