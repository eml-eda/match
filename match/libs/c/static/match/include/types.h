#ifndef __MATCH_TYPES_H__
#define __MATCH_TYPES_H__

#include <stdint.h>
#include <stdlib.h>


typedef struct dma_transfer {
    void* src;
    void* dst;
    size_t size;
} dma_transfer_t;

#endif