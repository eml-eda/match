#include <match/memory.h>

volatile void* match_mem = 0x0;
volatile int match_num_workspaces = 0;
volatile int match_num_allocs = 0;
volatile int match_workspaces_offsets[__MATCH_MEMORY_MAX_WORKSPACES__];
volatile int match_workspaces_sizes[__MATCH_MEMORY_MAX_WORKSPACES__];
volatile int match_workspaces_allocs[__MATCH_MEMORY_MAX_ALLOCS__];

void match_set_match_mem_pt(void* match_mem_pt){
    match_mem = match_mem_pt;
}

void match_alloc_workspace(int offset, int size){
    match_workspaces_offsets[match_num_workspaces] = offset;
    match_workspaces_sizes[match_num_workspaces] = size;
    match_num_workspaces++;
    if(match_num_workspaces >= __MATCH_MEMORY_MAX_WORKSPACES__){
        // printf("[MATCH] Maximum number of workspaces reached (%d). Cannot allocate more.\n", __MATCH_MEMORY_MAX_WORKSPACES__);
        exit(-1);
    }
    return;
}

void match_free_workspace(void){
    match_num_workspaces = 0;
    match_num_allocs = 0;
}

void* match_try_alloc_from_match_mem(int size){
    if(match_mem == 0x0){
        // printf("[MATCH] match_mem is not set. Cannot allocate from match memory.\n");
        return 0x0;
    }
    
    // Check if there is enough space in match_mem
    for(int i = 0; i < match_num_workspaces; i++){
        if(match_workspaces_sizes[i] >= size){
            void* ptr = match_mem + match_workspaces_offsets[i];
            match_workspaces_offsets[i] += size; // Move the offset forward
            match_workspaces_sizes[i] -= size; // Reduce the size of the workspace
            match_workspaces_allocs[match_num_allocs] = ptr; // Store the pointer in the allocs array
            match_num_allocs++;
            return ptr;
        }
    }
    
    return 0x0;
}

int match_try_free_from_match_mem(void* ptr){
    if(match_mem == 0x0){
        // printf("[MATCH] match_mem is not set. Cannot free to match memory.\n");
        return 0x0;
    }
    
    // Find the workspace that contains the pointer
    for(int i = match_num_allocs-1; i >= 0; i--){
        if(ptr == match_workspaces_allocs[i]){
            // Reset the workspace
            // match_workspaces_offsets[i] -= size;
            // match_workspaces_sizes[i] += size;
            match_num_allocs--;
            return 0;
        }
    }
    
    // printf("[MATCH] Pointer %p does not belong to any workspace.\n", ptr);
    return -1;
}