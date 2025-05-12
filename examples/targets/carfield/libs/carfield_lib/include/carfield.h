#ifndef CAR_LIB_CARFIELD_H
#define CAR_LIB_CARFIELD_H

#include <match/ctx.h>

#include <carfield_lib/dma.h>

#define L1_SCRATCHPAD_SIZE 32768

// #define CLUSTER_LIB_DEBUG

void reset_cluster();

void offload_to_pulp_cluster(void* boot_addr);

void carfield_init();

void carfield_shutdown();

void handle_host_dma_transfer(
    void* src,
    void* dst,
    size_t size
);

extern const uint8_t __l2_common_start[];
extern const uint8_t __l2_common_end[];

#define offload_args ((volatile uint32_t*)__l2_common_start)

#endif // CAR_LIB_CARFIELD_H