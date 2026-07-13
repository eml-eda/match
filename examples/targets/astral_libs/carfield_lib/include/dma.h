#ifndef CAR_LIB_DMA_H
#define CAR_LIB_DMA_H

#include <stdint.h>

#define DMA_DIR_LOC2EXT 0
#define DMA_DIR_EXT2LOC 1

#define __PULP_NO_DMA__ 0
#define __PULP_USE_MCHAN_API__ 0

typedef struct { 
  unsigned int size_2d;
  unsigned int l1_length_2d;
  unsigned int l1_stride_2d;
  unsigned int ext_length_2d;
  unsigned int ext_stride_2d;
} transfer_2d;

void mchan_2d(
    uint8_t* ext_ptr, uint8_t* l1_ptr,
    transfer_2d transfer_params, int core_id, int ext2loc
);
void mchan_1d(
    uint8_t* ext_ptr, uint8_t* l1_ptr,
    unsigned int size, int core_id, int ext2loc
);


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

void pulp_cluster_transfer_1d(dma_transfer_cfg_t conf);
void pulp_cluster_transfer_2d(dma_transfer_cfg_t conf);
void pulp_cluster_transfer_3d(dma_transfer_cfg_t conf);

dma_transfer_id_t dma_transfer_create();
void dma_transfer_free(dma_transfer_id_t transfer);
void dma_transfer_wait(dma_transfer_id_t transfer);

void dma_mutex_init();
void dma_mutex_lock();
void dma_mutex_unlock();

#endif  // CAR_LIB_DMA_H
