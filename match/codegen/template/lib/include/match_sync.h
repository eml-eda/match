#ifndef _MATCH_SYNC_H
#define _MATCH_SYNC_H

void match_curr_computation(common_kernel* common_kernel);

void match_prev_computation(common_kernel* common_kernel);

void match_async_transfers(common_kernel* common_kernel);

void match_sync_multilevel_transfer(common_kernel* common_kernel);

#endif