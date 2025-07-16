#ifndef __MATCH_MEMORY_H__
#define __MATCH_MEMORY_H__
#define __MATCH_MEMORY_MAX_WORKSPACES__ 8
#define __MATCH_MEMORY_MAX_ALLOCS__ 16

void match_set_match_mem_pt(void* match_mem_pt);

void match_alloc_workspace(int offset, int size);

void match_free_workspace(void);

void* match_try_alloc_from_match_mem(int size);

int match_try_free_from_match_mem(void* ptr);

#endif