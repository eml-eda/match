#ifdef __spatz__

#include <carfield_lib/dma.h>

#include "snrt.h"


// plp_dma_getCmd(int ext2loc, unsigned int size, int is2D, int trigEvt, int trigIrq, int broadcast);

// snrt_dma_start_2d(void *dst, const void *src, size_t size, size_t dst_stride, size_t src_stride, size_t repeat)

// snrt_dma_start_1d(void *dst, const void *src, size_t size);

// plp_dma_cmd_push_2d(unsigned int cmd, unsigned int locAddr, mchan_ext_t extAddr, unsigned int stride, unsigned int length);


dma_transfer_id_t dma_transfer_create() { return 0; }

void dma_transfer_free(dma_transfer_id_t transfer) {
  ;
}

void dma_transfer_wait(dma_transfer_id_t transfer) {
  snrt_dma_wait_all();
}

void dma_transfer_hwc_to_chw(dma_transfer_cfg_t conf) {

  int start_fm_channel = 0, end_fm_channel = conf.length_1d_copy;
  void *loc = conf.loc + conf.number_of_1d_copies * conf.number_of_2d_copies *
                             start_fm_channel;
  void *ext = conf.ext + start_fm_channel;
  const int size_2d = conf.number_of_1d_copies * conf.number_of_2d_copies;

  for (int i = start_fm_channel; i < end_fm_channel; i++) {

    if (conf.dir == DMA_DIR_L2_TO_L1) {
      snrt_dma_start_2d(loc, ext, size_2d, conf.stride_1d, 1, 1);
    } else {
      snrt_dma_start_2d(ext, loc, size_2d, conf.stride_1d, 1, 1);
    }

    ext += 1; // next channel  TODOOOOO this is probably not ok if sizeof(type) != 1
    loc += conf.number_of_1d_copies * conf.number_of_2d_copies;
  }
}

void dma_transfer_1d_async(dma_transfer_cfg_t conf) {
  if (conf.dir == DMA_DIR_L2_TO_L1) {
    snrt_dma_start_1d(conf.loc, conf.ext, conf.length_1d_copy);
  } else {
    snrt_dma_start_1d(conf.ext, conf.loc, conf.length_1d_copy);
  }
}

void dma_transfer_2d_async(dma_transfer_cfg_t conf) {
  if (conf.dir == DMA_DIR_L2_TO_L1) {
    snrt_dma_start_2d(conf.loc, conf.ext, conf.length_1d_copy, conf.length_1d_copy, conf.stride_1d, conf.number_of_1d_copies);
  } else {
    snrt_dma_start_2d(conf.ext, conf.loc, conf.length_1d_copy, conf.length_1d_copy, conf.stride_1d, conf.number_of_1d_copies);
  }
}

void dma_transfer_3d_async(dma_transfer_cfg_t conf) {
  const int size_2d = conf.number_of_1d_copies * conf.length_1d_copy;

  for (int i = 0; i < conf.number_of_2d_copies; i++) {
    dma_transfer_2d_async(conf);
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


#endif // __spatz__