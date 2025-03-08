#ifndef _CLUSTER_COMP_H
#define _CLUSTER_COMP_H
#include <match_kernel.h>
#include <match_target_params.h>

typedef struct cluster_kernel_t
{
    common_kernel* common_kernel;
    unsigned int im2col_pt;
    unsigned int pwtbuf_pt;
}cluster_kernel;

void cluster_init_other_kernel_params(cluster_kernel* kernel);

void cluster_kernel_function_wrapper(cluster_kernel* kernel);
#endif