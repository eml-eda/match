#ifndef __PULP_CLUSTER_NN_WRAPPER_H__
#define __PULP_CLUSTER_NN_WRAPPER_H__

#include <pulp_cluster/match_dev.h>
#include <pulp_nn/pulp_nn_kernels.h>

void pulp_nn_dense_wrapper(void* args);
void pulp_nn_dense_out_int_wrapper(void* args);
void pulp_nn_dw_conv2d_less_4_wrapper(void* args);
void pulp_nn_dw_conv2d_wrapper(void* args);
void pulp_nn_pw_conv2d_wrapper(void* args);
void pulp_nn_hoparallel_conv2d_wrapper(void* args);
void pulp_nn_add_wrapper(void* args);
void pulp_nn_conv3d_wrapper(void* args);

#endif // __PULP_CLUSTER_NN_WRAPPER_H__