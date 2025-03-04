#ifndef __PULP_CLUSTER_H__
#define __PULP_CLUSTER_H__


#include <pmsis.h>
extern struct pi_device cluster_dev;
extern struct pi_cluster_task cluster_task;

void pulp_cluster_init();

void pulp_cluster_close();

#endif //GAP9_CLUSTER_H