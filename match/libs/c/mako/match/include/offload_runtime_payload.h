#ifndef __MATCH_OFFLOAD_PAYLOAD_${exec_module.name}_H__
#define __MATCH_OFFLOAD_PAYLOAD_${exec_module.name}_H__

#include "match/types.h"

// @name_begin@ ${exec_module.name} @name_end@

// @data_sections_begin@
// @data_sections_end@

dma_transfer_t ${exec_module.name}_binary_sections[] = {
// @dma_sections_begin@
// @dma_sections_end@
    {NULL,NULL,0}
};

volatile void* ${exec_module.name}_boot_addr = NULL;

#endif // __MATCH_OFFLOAD_PAYLOAD_${exec_module.name}_H__