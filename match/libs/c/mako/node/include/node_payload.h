#ifndef __MATCH_NODE_PAYLOAD_${node_fullname}_H__
#define __MATCH_NODE_PAYLOAD_${node_fullname}_H__

#include "match/types.h"

// @name_begin@ ${node_fullname} @name_end@

// @data_sections_begin@
// @data_sections_end@

dma_transfer_t ${node_fullname}_binary_sections[] = {
// @dma_sections_begin@
// @dma_sections_end@
    {NULL,NULL,0}
};

volatile void* ${node_fullname}_boot_addr = NULL;
volatile void* ${node_fullname}_args_addr = NULL;

#endif