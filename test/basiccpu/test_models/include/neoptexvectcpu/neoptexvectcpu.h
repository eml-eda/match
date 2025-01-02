#ifndef __MATCH_EXEC_MODULE_neoptexvectcpu_H__
#define __MATCH_EXEC_MODULE_neoptexvectcpu_H__

typedef enum{
    mem_computation
    ,NEOPTEX_L1_CACHE
    ,NEOPTEX_L2_CACHE
}neoptexvectcpu_memories;

typedef enum{
    default_pattern
    ,vec_dense
    ,vec_conv
}neoptexvectcpu_patterns;






#endif