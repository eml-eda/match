#ifndef __PULP_CLUSTER_MATCH_DEVICE_H__
#define __PULP_CLUSTER_MATCH_DEVICE_H__

#include <pmsis.h>
#include <match/ctx.h>
#ifdef GAP_SDK
#include <GAP9.h>
#else
#include <pulp_platform.h>
#endif

#include <pulp_cluster/buffers.h>

#ifdef GAP_SDK
#define L1_SCRATCHPAD_SIZE 101*1024
#else
#define L1_SCRATCHPAD_SIZE 38*1024
#endif
// #define CLUSTER_LIB_DEBUG
#define USE_ODL_KERNEL 1
#define USE_ODL_KERNEL_PW 1

#endif // __PULP_CLUSTER_MATCH_DEVICE_H__