#ifndef __MATCH_TYPES_H__
#define __MATCH_TYPES_H__

#include <stddef.h>
#include <stdint.h>


typedef struct dma_transfer {
    void* src;
    void* dst;
    size_t size;
} dma_transfer_t;


typedef struct node_stats {
    uint32_t total_cycles;
    uint32_t compute_cycles;
    uint32_t load_cycles;
    uint32_t store_cycles;
    uint32_t load_bytes;
    uint32_t store_bytes;
    uint32_t device;
} node_stats_t;

#endif