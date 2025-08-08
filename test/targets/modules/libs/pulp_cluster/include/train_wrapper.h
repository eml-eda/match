#ifndef __PULP_CLUSTER_TRAIN_WRAPPER_H__
#define __PULP_CLUSTER_TRAIN_WRAPPER_H__

#include <pulp_cluster/match_dev.h>
#include <pulp_train/pulp_conv2d_fp32.h> // FIXME: this should not be here. quick fix for the typedef
#include <pulp_train/pulp_conv_dw_fp32.h> // FIXME: this should not be here. quick fix for the typedef
#include <pulp_train/pulp_conv_pw_fp32.h>
#include <pulp_train/pulp_train_utils_fp32.h>

void pulp_train_conv2d_fp32_wrapper(void* args);
void pulp_train_conv2ddw_fp32_wrapper(void* args);

void pulp_train_conv2d_bw_fp32_wrapper(void* args);

#endif // __PULP_CLUSTER_TRAIN_WRAPPER_H__