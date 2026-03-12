#ifdef __pulp_cluster__

#include <carfield_lib/dma.h>

#include "pulp.h"

dma_transfer_id_t dma_transfer_create() { return plp_dma_counter_alloc(); }

void dma_transfer_free(dma_transfer_id_t transfer) {
  plp_dma_counter_free(transfer);
}

void dma_transfer_wait(dma_transfer_id_t transfer) {
  plp_dma_wait(transfer);
  // free in plp_dma_wait
}

void dma_transfer_hwc_to_chw(dma_transfer_cfg_t conf) {
  int start_fm_channel = 0, end_fm_channel = conf.length_1d_copy;
  void *loc = conf.loc + conf.number_of_1d_copies * conf.number_of_2d_copies *
                             start_fm_channel;
  void *ext = conf.ext + start_fm_channel;
  const int size_2d = conf.number_of_1d_copies * conf.number_of_2d_copies;

  for (int i = start_fm_channel; i < end_fm_channel; i++) {
    unsigned int dma_cmd = plp_dma_getCmd(conf.dir, size_2d, 1, 1, 1, 1);
    plp_dma_cmd_push_2d(dma_cmd, loc, ext, conf.stride_1d, 1);
    ext += 1; // next channel // TODO check if this is correct for other types different than uint8_t
    loc += conf.number_of_1d_copies * conf.number_of_2d_copies;
  }
}

void dma_transfer_1d_async(dma_transfer_cfg_t conf) {
  unsigned int dma_cmd =
      plp_dma_getCmd(conf.dir, conf.length_1d_copy, 0, 1, 1, 1);
  plp_dma_cmd_push(dma_cmd, conf.loc, conf.ext);
}

void dma_transfer_2d_async(dma_transfer_cfg_t conf) {
  const int size_2d = conf.number_of_1d_copies * conf.length_1d_copy;
  unsigned int dma_cmd = plp_dma_getCmd(conf.dir, size_2d, 1, 1, 1, 1);
  plp_dma_cmd_push_2d(dma_cmd, conf.loc, conf.ext, conf.stride_1d,
                      conf.length_1d_copy);
}

void dma_transfer_3d_async(dma_transfer_cfg_t conf) {
  const int size_2d = conf.number_of_1d_copies * conf.length_1d_copy;

  for (int i = 0; i < conf.number_of_2d_copies; i++) {
    pulp_cluster_transfer_2d(conf);
    conf.loc += size_2d;
    conf.ext += conf.stride_2d;
  }
}

void dma_transfer_async(dma_transfer_cfg_t conf) {
  if (conf.hwc_to_chw == 1) {
    dma_transfer_hwc_to_chw(conf);
  } else if (conf.number_of_2d_copies == 1 && conf.number_of_1d_copies == 1) {
    dma_transfer_1d_async(conf);
  } else if (conf.number_of_2d_copies == 1) {
    dma_transfer_2d_async(conf);
  } else {
    dma_transfer_3d_async(conf);
  }
}

void pulp_cluster_transfer_1d(dma_transfer_cfg_t conf) {
  #if __PULP_NO_DMA__
  int core_id = rt_core_id();
  int num_cores = get_core_num();
  if (conf.dir == DMA_DIR_EXT2LOC) {
      for(int i=core_id; i<conf.length_1d_copy; i+=num_cores)
          ((uint8_t*)conf.loc)[i] = ((uint8_t*)conf.ext)[i];
  } else {
      for(int i=core_id; i<conf.length_1d_copy; i+=num_cores)
          ((uint8_t*)conf.ext)[i] = ((uint8_t*)conf.loc)[i];
  }
  #else
    dma_transfer_1d_async(conf);
  #endif
}

void pulp_cluster_transfer_2d(dma_transfer_cfg_t conf) {
  #if __PULP_NO_DMA__
  int core_id = rt_core_id();
  int num_cores = get_core_num();
  uint8_t* loc_ptr = conf.loc;
  uint8_t* ext_ptr = conf.ext;
  uint8_t* start_ext_ptr = ext_ptr;
  unsigned int blkCnt = conf.length_1d_copy >> 2u;
  unsigned int lfover = conf.length_1d_copy & 0x3;
  if(conf.dir == DMA_DIR_EXT2LOC){
    for(int i=core_id; i<conf.number_of_1d_copies; i+=num_cores){
      for(int j=0; j<blkCnt; j++){
          *((int*)loc_ptr) = *((int*)ext_ptr);
          loc_ptr += 4;
          ext_ptr += 4;
      }
      while(lfover){
          *((uint8_t*)loc_ptr) = *((uint8_t*)ext_ptr);
          loc_ptr++;
          ext_ptr++;
          lfover--;
      }
      ext_ptr = start_ext_ptr + i*conf.stride_1d;
    }
  }else{
    for(int i=core_id; i<conf.number_of_1d_copies; i+=num_cores){
      for(int j=0; j<blkCnt; j++){
          *((int*)ext_ptr) = *((int*)loc_ptr);
          loc_ptr += 4;
          ext_ptr += 4;
      }
      while(lfover){
          *((uint8_t*)ext_ptr) = *((uint8_t*)loc_ptr);
          loc_ptr++;
          ext_ptr++;
          lfover--;
      }
      ext_ptr = start_ext_ptr + i*conf.stride_1d;
    }
  }
  #else
    dma_transfer_2d_async(conf);
  #endif
}

void pulp_cluster_transfer_3d(dma_transfer_cfg_t conf) {
  dma_transfer_3d_async(conf);
}

static uint32_t dma_mutex;

void dma_mutex_init() {
  dma_mutex = eu_mutex_addr(0);
}

void dma_mutex_lock() {
  eu_mutex_lock(dma_mutex);
}

void dma_mutex_unlock() {
  eu_mutex_unlock(dma_mutex);
}

#endif // __pulp_cluster__