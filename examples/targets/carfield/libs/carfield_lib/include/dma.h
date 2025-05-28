#ifndef CAR_LIB_DMA_H
#define CAR_LIB_DMA_H

#include <stdint.h>

#ifdef __pulp_cluster__
#include "pulp.h"
#endif

#define DMA_DIR_L2_TO_L1 0
#define DMA_DIR_L1_TO_L2 1

typedef struct dma_transfer_cfg {
  uint32_t ext;
  uint32_t loc;
  int stride_2d;
  int number_of_2d_copies;
  int stride_1d;
  int number_of_1d_copies;
  int length_1d_copy;
  int hwc_to_chw;
  int dir; // 0 l1->l2, 1 l2->l1
} dma_transfer_cfg_t;

typedef int dma_transfer_id_t;

void dma_transfer_1d_async(dma_transfer_cfg_t conf);
void dma_transfer_2d_async(dma_transfer_cfg_t conf);
void dma_transfer_3d_async(dma_transfer_cfg_t conf);
void dma_transfer_async(dma_transfer_cfg_t conf);
void dma_transfer_hwc_to_chw(dma_transfer_cfg_t conf);

dma_transfer_id_t dma_transfer_create();
void dma_transfer_free(dma_transfer_id_t transfer);
void dma_transfer_wait(dma_transfer_id_t transfer);

void dma_mutex_init();
void dma_mutex_lock();
void dma_mutex_unlock();

#endif  // CAR_LIB_DMA_H
