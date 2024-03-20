#ifndef _MATCH_SYNC_H
#define _MATCH_SYNC_H

#include <match_kernel.h>

void match_curr_computation(unsigned int task_id,common_kernel* common_kernel);

void match_prev_computation(unsigned int task_id,common_kernel* common_kernel);

void match_async_transfers(unsigned int task_id,common_kernel* common_kernel);

void match_sync_multilevel_transfer(unsigned int task_id,common_kernel* common_kernel);

#endif