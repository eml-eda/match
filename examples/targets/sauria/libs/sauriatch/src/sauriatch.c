#include <sauriatch/sauriatch.h>

static sauria_t sauria;
static int __sauria_node_count__ = 0;

void __init_sauria__(MatchCtx* ctx, void (*__inner_node_fn__)(void *), unsigned int **args) {
    if(!__sauria_node_count__) {
        sauria.base_addr = mmio_region_from_addr((uintptr_t)SAURIA_PERIPH_START_ADDRESS);
    }
    __sauria_node_count__++;
    __inner_node_fn__(args);
}

/* ------------------------------------------------------------------ */
/*  Primitive: unico punto che tocca il DMA della piattaforma          */
/* ------------------------------------------------------------------ */
void sauria_1d_dma_transfer(void* src, void* dst, size_t size) {
    sys_dma_blk_memcpy((uintptr_t)dst,
                       (uintptr_t)src,
                       size,
                       DMA_CONF_DECOUPLE_NONE);
}

/* is_load = 1 -> L2 -> L1, is_load = 0 -> L1 -> L2 */
static inline void sauria_dma_1d(uint8_t* l2, uint8_t* l1, size_t bytes, int is_load) {
    #ifdef __DEBUG_DMA__
    mini_printf("            1D transfer: bytes=%d\r\n", bytes);
    #endif
    if (bytes == 0) return;              // size-1 con size=0 andrebbe in underflow
    if (is_load) sauria_1d_dma_transfer(l2, l1, bytes);
    else         sauria_1d_dma_transfer(l1, l2, bytes);
}

/* Equivalente di pulp_cluster_transfer_2d:
 * n righe da len byte, in L2 separate da l2_stride, in L1 contigue */
static void sauria_dma_2d(uint8_t* l2, uint8_t* l1,
                          int n, size_t len, size_t l2_stride, int is_load) {
    #ifdef __DEBUG_DMA__
    mini_printf("            2D transfer: n=%d, len=%d, l2_stride=%d\r\n", n, len, l2_stride);
    #endif
    if (l2_stride == len) {              // in realtà è contiguo: un solo transfer
        sauria_dma_1d(l2, l1, (size_t)n * len, is_load);
        return;
    }
    for (int i = 0; i < n; i++)
        sauria_dma_1d(l2 + i * l2_stride, l1 + i * len, len, is_load);
}

/* Equivalente di pulp_cluster_transfer_3d */
static void sauria_dma_3d(uint8_t* l2, uint8_t* l1,
                          int n2, int n1, size_t len,
                          size_t l2_stride_1d, size_t l2_stride_2d, int is_load) {
    #ifdef __DEBUG_DMA__
    mini_printf("            3D transfer: n2=%d, n1=%d, len=%d, l2_stride_1d=%d, l2_stride_2d=%d\r\n",
                n2, n1, len, l2_stride_1d, l2_stride_2d);
    #endif
    for (int j = 0; j < n2; j++)
        sauria_dma_2d(l2 + j * l2_stride_2d, l1 + (size_t)j * n1 * len,
                      n1, len, l2_stride_1d, is_load);
}

/* Equivalente di dma_transfer_hwc_to_chw (solo load L2 -> L1).
 * Con soli transfer 1D la trasposizione richiede un transfer per elemento. */
static void sauria_dma_hwc_to_chw(uint8_t* l2, uint8_t* l1,
                                  int H, int W, int C,
                                  size_t l2_stride_h, size_t l2_stride_w, size_t elem) {
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                sauria_dma_1d(l2 + h * l2_stride_h + w * l2_stride_w + c * elem,
                              l1 + (((size_t)c * H + h) * W + w) * elem,
                              elem, 1);
}

int sauria_dma_transfer_complete(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem
){
    #ifdef __DEBUG_DMA__
    mini_printf("[SAURIA DMA]: tensor=%p, tensor_l2_pt=%p, tensor_l1_pt=%p, match_transfer_type=%d, ext_mem=%d, int_mem=%d\r\n",
                (void*)tensor, (void*)tensor_l2_pt, (void*)tensor_l1_pt,
                match_transfer_type, ext_mem, int_mem);
    mini_printf("[SAURIA DMA]: tensor->num_dims=%d, tensor->tensor_type=%d, tensor->bits=%d\r\n",
                tensor->num_dims, tensor->tensor_type, tensor->bits);
    for(int idx=0; idx<tensor->num_dims; idx++) 
        mini_printf(" [L2: %d L1: %d]", tensor->tiles[0*tensor->num_dims+idx].size, tensor->tiles[1*tensor->num_dims+idx].size);
    mini_printf("\r\n");
    #endif
    asm volatile("fence rw,rw":::"memory");
    // if (ext_mem != __dram_mem_idx__ || int_mem != __scratchpad_mem_idx__)
    //     exit(1);
    if (tensor->num_dims > 5)
        return -1;
    if (!tensor->num_dims) return 0;

    uint8_t* l2 = (uint8_t*)tensor_l2_pt;
    uint8_t* l1 = (uint8_t*)tensor_l1_pt;
    const int    is_load = (match_transfer_type == MATCH_SW_LOAD_TENSOR);
    const size_t elem    = tensor->bits / 8;
    int transferred_bytes = 0;
    int match_tensor_type = tensor->tensor_type;
    int __scratchpad_mem_idx__ = 1;
    int __dram_mem_idx__ = 0;

    switch (tensor->num_dims) {
        case 1: {
            int bytes = L1_SZ(1,0) * elem;
            sauria_dma_1d(l2, l1, bytes, is_load);
            transferred_bytes = bytes;
            break;
        }
        case 2: {
            int is_1d = L2_SZ(2,1) == L1_SZ(2,1);
            int bytes = L1_SZ(2,0) * L1_SZ(2,1) * elem;
            if (is_1d)
                sauria_dma_1d(l2, l1, bytes, is_load);
            else
                sauria_dma_2d(l2, l1, L1_SZ(2,0),
                              L1_SZ(2,1) * elem,
                              L2_SZ(2,1) * elem, is_load);
            transferred_bytes = bytes;
            break;
        }
        case 3: {
            int is_1d = L2_SZ(3,1) == L1_SZ(3,1) && L2_SZ(3,2) == L1_SZ(3,2);
            int is_2d = L2_SZ(3,2) == L1_SZ(3,2);
            int bytes = L1_SZ(3,0) * L1_SZ(3,1) * L1_SZ(3,2) * elem;
            if (is_1d)
                sauria_dma_1d(l2, l1, bytes, is_load);
            else if (is_2d)
                sauria_dma_2d(l2, l1, L1_SZ(3,0),
                              L1_SZ(3,1) * L1_SZ(3,2) * elem,
                              L2_SZ(3,1) * L2_SZ(3,2) * elem, is_load);
            else
                sauria_dma_3d(l2, l1, L1_SZ(3,0), L1_SZ(3,1),
                              L1_SZ(3,2) * elem,
                              L2_SZ(3,2) * elem,
                              L2_SZ(3,1) * L2_SZ(3,2) * elem, is_load);
            transferred_bytes = bytes;
            break;
        }
        case 4: {
            int is_hwc_to_chw = 0; // TODO: implementare la logica per determinare se è necessario un trasferimento HWC->CHW
            int is_1d = L2_SZ(4,1) == L1_SZ(4,1) && L2_SZ(4,2) == L1_SZ(4,2) && L2_SZ(4,3) == L1_SZ(4,3);
            int is_2d = L2_SZ(4,2) == L1_SZ(4,2) && L2_SZ(4,3) == L1_SZ(4,3);
            int bytes = 0;
            if (is_hwc_to_chw) {
                bytes = L1_SZ(4,1) * L1_SZ(4,2) * L1_SZ(4,3) * elem;
                sauria_dma_hwc_to_chw(l2, l1,
                                      L1_SZ(4,1), L1_SZ(4,2), L1_SZ(4,3),
                                      L2_SZ(4,2) * L2_SZ(4,3) * elem,
                                      L2_SZ(4,3) * elem, elem);
            } else if (is_1d) {
                bytes = L1_SZ(4,0) * L1_SZ(4,1) * L1_SZ(4,2) * L1_SZ(4,3) * elem;
                sauria_dma_1d(l2, l1, bytes, is_load);
            } else if (is_2d) {
                bytes = L1_SZ(4,0) * L1_SZ(4,1) * L1_SZ(4,2) * L1_SZ(4,3) * elem;
                sauria_dma_2d(l2, l1, L1_SZ(4,0),
                              L1_SZ(4,1) * L1_SZ(4,2) * L1_SZ(4,3) * elem,
                              L2_SZ(4,1) * L2_SZ(4,2) * L2_SZ(4,3) * elem, is_load);
            } else {
                size_t l2_vol = L2_SZ(4,1) * L2_SZ(4,2) * L2_SZ(4,3) * elem;
                size_t l1_vol = L1_SZ(4,1) * L1_SZ(4,2) * L1_SZ(4,3) * elem;
                for (int idx = 0; idx < L1_SZ(4,0); idx++)
                    sauria_dma_3d(l2 + idx * l2_vol, l1 + idx * l1_vol,
                                  L1_SZ(4,1), L1_SZ(4,2),
                                  L1_SZ(4,3) * elem,
                                  L2_SZ(4,3) * elem,
                                  L2_SZ(4,2) * L2_SZ(4,3) * elem, is_load);
                bytes = l1_vol * L1_SZ(4,0);
            }
            transferred_bytes = bytes;
            break;
        }
        case 5: {
            int bytes = 0;
            int is_hwc_to_chw = 0; // TODO: implementare la logica per determinare se è necessario un trasferimento HWC->CHW
            if (tensor->tiles[__dram_mem_idx__*5+1].dim == tensor->tiles[__scratchpad_mem_idx__*5+4].dim) {
                int is_1d = L2_SZ(5,1) == L1_SZ(5,1) && L2_SZ(5,2) == L1_SZ(5,2) && L2_SZ(5,3) == L1_SZ(5,3);
                int is_2d = L2_SZ(5,2) == L1_SZ(5,2) && L2_SZ(5,3) == L1_SZ(5,3);
                if (is_1d) {
                    bytes = L1_SZ(5,0) * L1_SZ(5,1) * L1_SZ(5,2) * L1_SZ(5,3) * L1_SZ(5,4) * elem;
                    sauria_dma_1d(l2, l1, bytes, is_load);
                } else if (is_2d) {
                    bytes = L1_SZ(5,0) * L1_SZ(5,1) * L1_SZ(5,2) * L1_SZ(5,3) * L1_SZ(5,4) * elem;
                    sauria_dma_2d(l2, l1, L1_SZ(5,0),
                                  L1_SZ(5,1) * L1_SZ(5,2) * L1_SZ(5,3) * elem,
                                  L2_SZ(5,1) * L2_SZ(5,2) * L2_SZ(5,3) * elem, is_load);
                } else {
                    size_t l2_vol = L2_SZ(5,1) * L2_SZ(5,2) * L2_SZ(5,3) * elem;
                    size_t l1_vol = L1_SZ(5,1) * L1_SZ(5,2) * L1_SZ(5,3) * elem;
                    for (int idx = 0; idx < L1_SZ(5,0); idx++)
                        sauria_dma_3d(l2 + idx * l2_vol, l1 + idx * l1_vol,
                                      L1_SZ(5,1), L1_SZ(5,2),
                                      L1_SZ(5,3) * elem,
                                      L2_SZ(5,3) * elem,
                                      L2_SZ(5,2) * L2_SZ(5,3) * elem, is_load);
                    // mantiene il conteggio dell'originale
                    bytes = L1_SZ(5,1) * L1_SZ(5,2) * L1_SZ(5,3) * L1_SZ(5,4) * L1_SZ(5,0) * elem
                            * L1_SZ(5,0);
                }
            } else {
                int is_1d = L2_SZ(5,4) == L1_SZ(5,4) && L2_SZ(5,3) == L1_SZ(5,3) &&
                            L2_SZ(5,2) == L1_SZ(5,2) && L2_SZ(5,1) == L1_SZ(5,1) &&
                            L2_SZ(5,0) == L1_SZ(5,0);
                int is_2d = L2_SZ(5,4) == L1_SZ(5,4) && L2_SZ(5,3) == L1_SZ(5,3) &&
                            L2_SZ(5,2) == L1_SZ(5,2) && L2_SZ(5,1) == L1_SZ(5,1);
                int is_2d_dim_0_one = L2_SZ(5,4) == L1_SZ(5,4) && L2_SZ(5,3) == L1_SZ(5,3) &&
                                      L2_SZ(5,2) == L1_SZ(5,2) && L2_SZ(5,0) == 1;
                bytes = L1_SZ(5,0) * L1_SZ(5,1) * L1_SZ(5,2) * L1_SZ(5,3) * L1_SZ(5,4) * elem;

                if (is_1d) {
                    sauria_dma_1d(l2, l1, bytes, is_load);
                } else if (is_2d) {
                    sauria_dma_2d(l2, l1, L1_SZ(5,0),
                                  L1_SZ(5,1) * L1_SZ(5,2) * L1_SZ(5,3) * L1_SZ(5,4) * elem,
                                  L2_SZ(5,1) * L2_SZ(5,2) * L2_SZ(5,3) * L2_SZ(5,4) * elem, is_load);
                } else if (is_2d_dim_0_one) {
                    sauria_dma_2d(l2, l1, L1_SZ(5,1),
                                  L1_SZ(5,2) * L1_SZ(5,3) * L1_SZ(5,4) * elem,
                                  L2_SZ(5,2) * L2_SZ(5,3) * L2_SZ(5,4) * elem, is_load);
                } else {
                    size_t l2_s0 = L2_SZ(5,1) * L2_SZ(5,2) * L2_SZ(5,3) * L2_SZ(5,4) * elem;
                    size_t l2_s1 =              L2_SZ(5,2) * L2_SZ(5,3) * L2_SZ(5,4) * elem;
                    size_t l1_s0 = L1_SZ(5,1) * L1_SZ(5,2) * L1_SZ(5,3) * L1_SZ(5,4) * elem;
                    size_t l1_s1 =              L1_SZ(5,2) * L1_SZ(5,3) * L1_SZ(5,4) * elem;
                    for (int i0 = 0; i0 < L1_SZ(5,0); i0++)
                        for (int i1 = 0; i1 < L1_SZ(5,1); i1++)
                            sauria_dma_3d(l2 + i0 * l2_s0 + i1 * l2_s1,
                                          l1 + i0 * l1_s0 + i1 * l1_s1,
                                          L1_SZ(5,2), L1_SZ(5,3),
                                          L1_SZ(5,4) * elem,
                                          L2_SZ(5,4) * elem,
                                          L2_SZ(5,3) * L2_SZ(5,4) * elem, is_load);
                }
            }
            transferred_bytes = bytes;
            break;
        }
    }

    return transferred_bytes;
}

static void sauria_copy_tensor_recursive(
    uint8_t* dst_base,
    uint8_t* src_base,
    const size_t* src_strides,
    const size_t* dst_strides,
    const size_t* src_sizes,
    const size_t* dst_sizes,
    const size_t* copy_sizes,
    int dim,
    int num_dims
) {
    if (src_sizes[dim] == dst_sizes[dim]) {
        int transfer_dim = dim;
        while (transfer_dim + 1 < num_dims &&
               src_sizes[transfer_dim + 1] == dst_sizes[transfer_dim + 1] &&
               src_strides[transfer_dim] ==
                   src_strides[transfer_dim + 1] * src_sizes[transfer_dim + 1] &&
               dst_strides[transfer_dim] ==
                   dst_strides[transfer_dim + 1] * dst_sizes[transfer_dim + 1]) {
            transfer_dim++;
        }

        if (transfer_dim == num_dims - 1) {
            sauria_1d_dma_transfer(
                src_base,
                dst_base,
                copy_sizes[dim] * dst_strides[dim]
            );
            return;
        }
    }

    for (size_t idx = 0; idx < copy_sizes[dim]; idx++) {
        uint8_t* src = src_base + idx * src_strides[dim];
        uint8_t* dst = dst_base + idx * dst_strides[dim];

        if (dim == num_dims - 1) {
            sauria_1d_dma_transfer(src, dst, copy_sizes[dim] * dst_strides[dim]);
        } else {
            sauria_copy_tensor_recursive(
                dst,
                src,
                src_strides,
                dst_strides,
                src_sizes,
                dst_sizes,
                copy_sizes,
                dim + 1,
                num_dims
            );
        }
    }
}

void __sauria_wait_dma_transfer__(MatchCtx* ctx){
    return;
}

int sauria_dma_transfer_org(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem 
) {
    (void)ctx;

    asm volatile("fence rw,rw" ::: "memory");

    if (!tensor || !tensor->num_dims) {
        return 0;
    }

    if (tensor->num_dims > 5) {
        return -1;
    }

    size_t bytes_per_element = (size_t)(tensor->bits / 8);
    size_t l1_sizes[5] = {0};
    size_t l2_sizes[5] = {0};
    size_t l1_strides[5] = {0};
    size_t l2_strides[5] = {0};

    for (int idx = 0; idx < tensor->num_dims; idx++) {
        l1_sizes[idx] = (size_t)tensor->tiles[__sauria_scratchpad_ifmap_bank__ * tensor->num_dims + idx].size;
        l2_sizes[idx] = (size_t)tensor->tiles[__astral_dram__ * tensor->num_dims + idx].size;
    }

    l1_strides[tensor->num_dims - 1] = bytes_per_element;
    l2_strides[tensor->num_dims - 1] = bytes_per_element;
    for (int idx = tensor->num_dims - 2; idx >= 0; idx--) {
        l1_strides[idx] = l1_strides[idx + 1] * l1_sizes[idx + 1];
        l2_strides[idx] = l2_strides[idx + 1] * l2_sizes[idx + 1];
    }

    size_t transferred_bytes = l1_strides[0] * l1_sizes[0];

    uint8_t* src = match_transfer_type == MATCH_SW_LOAD_TENSOR
        ? (uint8_t*)tensor_l2_pt
        : (uint8_t*)tensor_l1_pt;
    uint8_t* dst = match_transfer_type == MATCH_SW_LOAD_TENSOR
        ? (uint8_t*)tensor_l1_pt
        : (uint8_t*)tensor_l2_pt;

    bool same_shape = true;
    for (int idx = 0; idx < tensor->num_dims; idx++) {
        if (l1_sizes[idx] != l2_sizes[idx]) {
            same_shape = false;
            break;
        }
    }

    if (same_shape) {
        sauria_1d_dma_transfer(src, dst, transferred_bytes);
    } else {
        sauria_copy_tensor_recursive(
            dst,
            src,
            match_transfer_type == MATCH_SW_LOAD_TENSOR ? l2_strides : l1_strides,
            match_transfer_type == MATCH_SW_LOAD_TENSOR ? l1_strides : l2_strides,
            match_transfer_type == MATCH_SW_LOAD_TENSOR ? l2_sizes : l1_sizes,
            match_transfer_type == MATCH_SW_LOAD_TENSOR ? l1_sizes : l2_sizes,
            l1_sizes,
            0,
            tensor->num_dims
        );
    }

    asm volatile("fence rw,rw" ::: "memory");

    return (int)transferred_bytes;
}

int sauria_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem 
)
{
    // return sauria_dma_transfer_org(ctx, tensor, tensor_l2_pt, tensor_l1_pt, match_transfer_type, ext_mem, int_mem);
    return sauria_dma_transfer_complete(ctx, tensor, tensor_l2_pt, tensor_l1_pt, match_transfer_type, ext_mem, int_mem);
}

void* __sauria_scratchpad_get_weights_bank__(MatchCtx* ctx) {
    return (void*)SAURIA_START_ADDRESS + SAURIA_SRAMB_OFFSET;
}

void* __sauria_scratchpad_get_ifmap_bank__(MatchCtx* ctx) {
    return (void*)SAURIA_START_ADDRESS + SAURIA_SRAMA_OFFSET;
}

void* __sauria_scratchpad_get_output_bank__(MatchCtx* ctx) {
    return (void*)SAURIA_START_ADDRESS + SAURIA_SRAMC_OFFSET;
}


static inline int ceil_int(int numerator, int denominator)
{
    return numerator / denominator + (numerator % denominator != 0);
}

/* One SAURIA run. SAURIA has no base-address registers: every SRAM address is a sum
 * of counters that all start from 0. On SRAMA:
 *     addr = til_y + til_x + ch + y + lwoffs[row] + tap
 * where "tap" is one of the set bits of the 64-bit dilation pattern inside the x window,
 * and y / ch are two plain affine loops. The reduction is taps (inner) -> y -> ch (outer),
 * so the weights in SRAMB must be stored in that order, with out_c innermost.
 * Outputs: out_w on the array rows (lwoffs = j*stride_w), out_c on the columns, and
 * out_w tiles / out_h / out_d on the activation tiling counters (til_x, til_y):
 *     out_d == 1   -> til_x = out_w tiles, til_y = out_h   (plain CONV2D)
 *     x_tiles == 1 -> til_x = out_h,       til_y = out_d
 *     out_h == 1   -> til_x = out_w tiles, til_y = out_d
 * so contexts always produce rows in order and SRAMC is out_c x out_d x out_h x out_w. */
typedef struct {
    uint64_t dil_pat;           // tap mask of one x sweep, bit DILP_W-1 = offset 0
    uint16_t taps;              // number of set bits in dil_pat
    uint16_t span;              // last tap offset + 1
    uint32_t ylim, ystep;       // middle reduction loop
    uint32_t chlim, chstep;     // outer reduction loop
    uint16_t out_c, out_d, out_h, out_w;
    uint16_t stride_w;          // input bytes between consecutive out_w
    uint32_t oh_step, od_step;  // input bytes between consecutive out_h / out_d
    bool preload;
} sauria_conv_cfg_t;

static void sauria_cfg_add_tap(sauria_conv_cfg_t* cfg, uint32_t offset) {
    cfg->dil_pat |= 1ULL << (DILP_W - 1 - offset);
    cfg->taps++;
    if (offset + 1 > cfg->span) cfg->span = offset + 1;
}

/* Every SAURIA config field is a fixed-width counter: a value that doesn't fit
 * gets silently truncated by the driver, so check before programming it. */
static int sauria_check_field(const char* name, uint32_t value, uint32_t bits) {
    if (value < (1u << bits)) return 0;
    #ifdef __DEBUG_COMPUTE__
    mini_printf("[SAURIA][KER] ERROR: %s=%d does not fit in %d bits\r\n", name, value, bits);
    #endif
    return 1;
}

static int sauria_largest_divisor(int n, int max) {
    if (n <= max) return n;
    for (int v = max; v > 1; v--)
        if (n % v == 0) return v;
    return 1;
}

static int execute_sauria_conv(const sauria_conv_cfg_t* cfg) {
    const int sauria_x = 8; // Number of PE units in the x direction (out channels)
    const int sauria_y = 8; // Number of PE units in the y direction (out_w positions)
    uint16_t out_c = cfg->out_c, out_d = cfg->out_d, out_h = cfg->out_h, out_w = cfg->out_w;
    uint16_t stride_w = cfg->stride_w;

    int x_used = sauria_largest_divisor(out_c, sauria_x);
    int y_used = sauria_largest_divisor(out_w, sauria_y);
    uint32_t x_tiles = ceil_int(out_w, y_used);
    uint32_t k_tiles = out_c / x_used;
    uint32_t out_dhw = (uint32_t)out_d * out_h * out_w;

    if (cfg->span == 0 || cfg->span > DILP_W) {
        #ifdef __DEBUG_COMPUTE__
        mini_printf("[SAURIA][KER] ERROR: tap span %d outside [1, %d]\r\n", cfg->span, DILP_W);
        #endif
        return -1;
    }
    if (out_d > 1 && x_tiles > 1 && out_h > 1) {
        #ifdef __DEBUG_COMPUTE__
        mini_printf("[SAURIA][KER] ERROR: out_d=%d needs out_w<=%d or out_h==1 (got out_w=%d out_h=%d)\r\n",
                    out_d, sauria_y, out_w, out_h);
        #endif
        return -1;
    }

    // Main FSM
    uint32_t n_macs = (uint32_t)cfg->taps * (cfg->ylim / cfg->ystep) * (cfg->chlim / cfg->chstep);
    uint32_t incntlim = n_macs - 1;                 // MACs per output
    uint32_t act_reps = k_tiles;                    // ifmap tiling iterations
    uint32_t wei_reps = x_tiles * out_h * out_d;    // weight tiling iterations (spatial contexts)

    // Ifmap feeder. The aligned-activations case of the reference driver needs
    // y_used % SRAMA_N == 0, impossible with y_used <= 8 < SRAMA_N, so xlim is always the window.
    uint32_t xlim = (1 + (y_used - 1) * stride_w) + cfg->span + 1 - (cfg->span % 2) + SRAMA_N;
    uint32_t xstep = SRAMA_N;
    uint32_t til_xlim = x_tiles * y_used * stride_w;
    uint32_t til_xstep = y_used * stride_w;
    uint32_t til_ylim = (uint32_t)out_h * cfg->oh_step;
    uint32_t til_ystep = cfg->oh_step;
    if (out_d > 1) {
        if (x_tiles == 1) {             // out_h moves to til_x, freeing til_y
            til_xlim = til_ylim;
            til_xstep = til_ystep;
        }
        til_ylim = (uint32_t)out_d * cfg->od_step;
        til_ystep = cfg->od_step;
    }

    uint8_t rows_active = 0;
    uint8_t lwoffs[8] = {0};            // Local word offset for Rows [0:7]
    for (int j = 0; j < y_used; j++) {
        rows_active |= (1 << (sauria_y - 1 - j));
        lwoffs[j] = j * stride_w;
    }

    // Weight feeder
    uint32_t wlim = (uint32_t)out_c * n_macs;
    uint16_t wstep = out_c;
    bool waligned = ((out_c % SRAMB_N) == 0) && (x_used == SRAMB_N);
    uint16_t klim = waligned ? 1 : SRAMB_N + 1;
    uint16_t kstep = SRAMB_N;
    uint16_t til_klim = out_c;
    uint16_t til_kstep = x_used;

    uint8_t cols_active = 0;            // Pattern of active columns
    for (int i = 0; i < x_used; i++)
        cols_active |= (1 << (sauria_x - 1 - i));

    // Partial sums manager
    uint32_t ncontexts = wei_reps * k_tiles;
    uint16_t cxlim = y_used + SRAMC_N;
    uint16_t cxstep = SRAMC_N;
    uint32_t cklim = out_dhw * x_used;
    uint32_t ckstep = out_dhw;
    uint32_t til_cylim = out_dhw;
    uint16_t til_cystep = y_used;
    uint32_t til_cklim = out_dhw * out_c;
    uint32_t til_ckstep = out_dhw * x_used;
    uint8_t inactive_cols = sauria_x - x_used;

    // Field widths from sauria_regs.h
    int overflow = 0;
    overflow |= sauria_check_field("incntlim", incntlim, 16);
    overflow |= sauria_check_field("act_reps", act_reps, 14);
    overflow |= sauria_check_field("wei_reps", wei_reps, 14);
    overflow |= sauria_check_field("xlim", xlim, 8);    // feeder shift index: PARAMS_W=8 bits in the reference RTL
    overflow |= sauria_check_field("ylim", cfg->ylim, 16);
    overflow |= sauria_check_field("chlim", cfg->chlim, 16);
    overflow |= sauria_check_field("til_xlim", til_xlim, 16);
    overflow |= sauria_check_field("til_ylim", til_ylim, 16);
    overflow |= sauria_check_field("wlim", wlim, 16);
    overflow |= sauria_check_field("ncontexts", ncontexts, 14);
    overflow |= sauria_check_field("cklim", cklim, 14);
    overflow |= sauria_check_field("til_cylim", til_cylim, 14);
    overflow |= sauria_check_field("til_cklim", til_cklim, 14);
    if (overflow) return -1;

    #ifdef __DEBUG_COMPUTE__
    mini_printf("[SAURIA][KER] execute_sauria_conv configuration:\r\n");
    mini_printf("x used: %d, y used: %d\r\n", x_used, y_used);
    mini_printf("  Incntlim: %d, Act Reps: %d, Wei Reps: %d\r\n", incntlim, act_reps, wei_reps);
    mini_printf("  xlim: %d, xstep: %d, ylim: %d, ystep: %d\r\n", xlim, xstep, cfg->ylim, cfg->ystep);
    mini_printf("  Chlim: %d, Chstep: %d, Til xlim: %d, Til xstep: %d, Til ylim: %d, Til ystep: %d\r\n", cfg->chlim, cfg->chstep, til_xlim, til_xstep, til_ylim, til_ystep);
    mini_printf("  Dil Pat: %x %x, Rows Active: %d\r\n", (uint32_t)(cfg->dil_pat >> 32), (uint32_t)cfg->dil_pat, rows_active);
    mini_printf("  Lwoffs: ");
    for (int i = 0; i < sauria_y; i++) {
        mini_printf("%d ", lwoffs[i]);
    }
    mini_printf("\r\n");
    mini_printf("  Wlim: %d, Wstep: %d, Waligned: %d, Klim: %d, Kstep: %d, Til Klim: %d, Til Kstep: %d\r\n", wlim, wstep, waligned, klim, kstep, til_klim, til_kstep);
    mini_printf("  Cols Active: %d, Inactive Cols: %d\r\n", cols_active, inactive_cols);
    mini_printf("  Ncontexts: %d, Cxlim: %d, Cxstep: %d, Cklim: %d, Ckstep: %d, Til Cylim: %d, Til Cystep: %d, Til Cklim: %d, Til Ckstep: %d\r\n", ncontexts, cxlim, cxstep, cklim, ckstep, til_cylim, til_cystep, til_cklim, til_ckstep);
    #endif

    sauria_global_ien(&sauria, true);
    sauria_done_ien(&sauria, true);
    sauria_reg0x200(&sauria, incntlim, act_reps, wei_reps);
    sauria_reg0x204(&sauria, wei_reps, 0 /*neg_thres*/, 0xFFFF /*res_mask */, 255 /*appr_mask*/);
    sauria_reg0x208(&sauria, 255 /*appr_mask*/, false /*sram_deeepsleep*/, false /*sram_powergate*/);
    sauria_reg0x400(&sauria, xlim, xstep);
    sauria_reg0x404(&sauria, cfg->ylim, cfg->ystep);
    sauria_reg0x408(&sauria, cfg->chlim, cfg->chstep);
    sauria_reg0x40C(&sauria, til_xlim, til_xstep);
    sauria_reg0x410(&sauria, til_ylim, til_ystep);
    sauria_reg0x414(&sauria, cfg->dil_pat);
    sauria_reg0x418(&sauria, cfg->dil_pat);
    sauria_reg0x41C(&sauria, rows_active, lwoffs[0], lwoffs[1], lwoffs[2]);
    sauria_reg0x420(&sauria, lwoffs[3], lwoffs[4], lwoffs[5], lwoffs[6]);
    sauria_reg0x424(&sauria, lwoffs[7]);
    sauria_reg0x600(&sauria, wlim, wstep);
    sauria_reg0x604(&sauria, klim, kstep);
    sauria_reg0x608(&sauria, til_klim, til_kstep);
    sauria_reg0x60C(&sauria, cols_active, waligned);
    sauria_reg0x800(&sauria, ncontexts, cxlim, cxstep);
    sauria_reg0x804(&sauria, cxstep, cklim, ckstep);
    sauria_reg0x808(&sauria, ckstep, til_cylim, til_cystep);
    sauria_reg0x80C(&sauria, til_cystep, til_cklim, til_ckstep, inactive_cols);
    sauria_reg0x810(&sauria, inactive_cols, cfg->preload);

    #ifdef __DEBUG_COMPUTE__
    mini_printf("[SAURIA][KER] Starting SAURIA...\r\n");
    #endif
    __asm__ volatile ("fence iorw, rw" ::: "memory");
    sauria_start(&sauria, true, false, false, false, false, false, false);
    #ifdef __DEBUG_COMPUTE__
    mini_printf("[SAURIA][KER] SAURIA started. Waiting for completion...\r\n");
    #endif
    while (!sauria_done(&sauria)) {
    }
    /* Each SRAM is double-buffered: host and SAURIA always see opposite halves, and the
     * select flips on every start (config_regs.sv). So start hands SAURIA the halves the
     * host just wrote, but its results land in the SRAMC half the host can't see.
     * Flip everything back: the host sees the outputs, SRAMA/B are again the halves it
     * wrote, and the next start gives SAURIA these outputs back (for preload). */
    sauria_start(&sauria, false, false, true /* mem_switch */, false, false, false, false);
    __asm__ volatile ("fence iorw, rw" ::: "memory");
    #ifdef __DEBUG_COMPUTE__
    mini_printf("[SAURIA][KER] SAURIA done.\r\n");
    #endif
    return 0;
}

/* CONV2D: input C x H x W, weights C x KH x KW x K (IHWO), output K x OH x OW.
 * Taps = kw, y = kh, ch = c. */
void execute_sauria_conv2d(
    uint16_t inp_c, uint16_t inp_h, uint16_t inp_w,
    uint16_t fil_h, uint16_t fil_w,
    uint16_t out_c, uint16_t out_h, uint16_t out_w,
    uint16_t stride_x, uint16_t dil_x, bool preload
) {
    sauria_conv_cfg_t cfg = {0};
    for (int kw = 0; kw < fil_w; kw++)
        sauria_cfg_add_tap(&cfg, kw * dil_x);
    cfg.ystep = (uint32_t)inp_w * dil_x;
    cfg.ylim = cfg.ystep * fil_h;
    cfg.chstep = (uint32_t)inp_w * inp_h;
    cfg.chlim = cfg.chstep * inp_c;
    cfg.out_c = out_c; cfg.out_d = 1; cfg.out_h = out_h; cfg.out_w = out_w;
    cfg.stride_w = stride_x;
    cfg.oh_step = (uint32_t)inp_w * stride_x;
    cfg.od_step = 0;
    cfg.preload = preload;
    execute_sauria_conv(&cfg);
}

void sauria_conv2d_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;

    void *input  = tensors[0].pt;
    void *weight = tensors[1].pt;
    void *output = tensors[num_tensors - 1].pt;
    // sauria_set_ifmap_ptr(NULL, (uint32_t)(uintptr_t)input);
    // sauria_set_weight_ptr(NULL, (uint32_t)(uintptr_t)weight);

    int base_tile_idx = __sauria_scratchpad_ifmap_bank__ * 4;
    uint16_t inp_c = (uint16_t)tensors[0].tiles[base_tile_idx + 1].size;
    uint16_t inp_h  = (uint16_t)tensors[0].tiles[base_tile_idx + 2].size;
    uint16_t inp_w  = (uint16_t)tensors[0].tiles[base_tile_idx + 3].size;
    uint16_t out_c  = (uint16_t)tensors[num_tensors - 1].tiles[base_tile_idx + 1].size;
    uint16_t out_h  = (uint16_t)tensors[num_tensors - 1].tiles[base_tile_idx + 2].size;
    uint16_t out_w  = (uint16_t)tensors[num_tensors - 1].tiles[base_tile_idx + 3].size;

    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    uint16_t fil_h      = (uint16_t)conv_attrs->kernel_size[0];
    uint16_t fil_w      = (uint16_t)conv_attrs->kernel_size[1];
    uint16_t stride_x   = (uint16_t)conv_attrs->strides[1];
    uint16_t dil_x      = (conv_attrs->dilation != NULL) ? (uint16_t)conv_attrs->dilation[1] : 1;

    #ifdef __DEBUG_COMPUTE__
    mini_printf("[SAURIA][KER] sauria_conv2d_wrapper:\r\n");
    mini_printf("  Inp Tile (C, H, W): (%d, %d, %d)\r\n",
               inp_c, inp_h, inp_w);
    mini_printf("  Kernel (H,W)  : (%d, %d)",
               fil_h, fil_w);
    mini_printf(" | Out Tile (C, H, W) : (%d, %d, %d)\r\n",
               out_c, out_h, out_w);
    mini_printf("  Stride x : %d", stride_x);
    mini_printf(" | Dilation x : %d\r\n", dil_x);
    mini_printf("  Inp fm ptr : %p", (void*)(uintptr_t)input);
    mini_printf(" | Weights ptr : %p", (void*)(uintptr_t)weight);
    mini_printf(" | Output ptr  : %p\r\n", (void*)(uintptr_t)output);

    mini_printf("  First 10 input values :");
    for (int i = 0; i < 10; i++) {
        mini_printf(" %d", ((uint8_t*)input)[i]);
    }
    mini_printf("\r\n");
    mini_printf("  First 10 weight values :");
    for (int i = 0; i < 10; i++) {
        mini_printf(" %d", ((int8_t*)weight)[i]);
    }
    mini_printf("\r\n");
    #endif

    execute_sauria_conv2d(
        inp_c, inp_h, inp_w,
        fil_h, fil_w,
        out_c, out_h, out_w,
        stride_x, dil_x,
        false  // preload is false for Conv2D
    );

    #ifdef __DEBUG_COMPUTE__
    mini_printf("  First 10 output values (@ %p):", (int32_t*)(uintptr_t)tensors[num_tensors - 1].pt);
    for (int i = 0; i < 10; i++) {
        volatile const int32_t *sauria_output =
            (volatile const int32_t *)(uintptr_t)(SAURIA_START_ADDRESS + SAURIA_SRAMC_OFFSET);
        mini_printf(" %d", ((int*)sauria_output)[i]);
    }
    mini_printf("\r\n");
    #endif


}

/* ------------------------------------------------------------------ */
/*  CONV3D                                                             */
/* ------------------------------------------------------------------ */
/* Tensors as MATCH puts them in the SRAMs:
 *   SRAMA: input   C x D  x H  x W       (NCDHW, uint8)
 *   SRAMB: weights C x KD x KH x KW x K  (IDHWO, int8, from Sauria.update_constants)
 *   SRAMC: output  K x OD x OH x OW      (NCDHW, int32)
 * With NCDHW the input address of every MAC splits into independent terms:
 *     c*D*H*W + kd*dd*H*W + (kh*dh*W + kw*dw) + od*sd*H*W + oh*sh*W + ow*sw
 *       ch       y            taps               til_y       til_x    rows
 * so the whole CONV3D is a single SAURIA run (sauria_conv3d_direct): the 64-bit tap
 * pattern holds the 2D kh x kw window, kd goes on the y loop and c on the ch loop.
 * The weight stream order (ch -> y -> taps -> k) is then exactly IDHWO.
 * When the tap window doesn't fit DILP_W or out_d can't go on the tiling counters,
 * sauria_conv3d_per_kd runs one CONV2D per kd (and per od), moving data in the SRAMs. */
typedef struct {
    uint8_t *input, *weight, *output;
    uint16_t inp_c, inp_d, inp_h, inp_w;
    uint16_t out_c, out_d, out_h, out_w;
    uint16_t fil_d, fil_h, fil_w;
    uint16_t stride_d, stride_h, stride_w;
    uint16_t dil_d, dil_h, dil_w;
} sauria_conv3d_t;

/* Must match the bank sizes declared in sauria.py (__SAURIA_*_BANK_SIZE__, in kB) */
#define SAURIA_BANK_BYTES (32 * 1024)

/* execute_sauria_conv can put out_d on til_y only if out_h or the out_w tiles leave til_x free */
static bool sauria_out_d_in_hw(uint16_t out_d, uint16_t out_h, uint16_t out_w) {
    int y_used = sauria_largest_divisor(out_w, 8 /* sauria_y */);
    return out_d == 1 || out_h == 1 || ceil_int(out_w, y_used) == 1;
}

/* Span of the kh x kw tap window on SRAMA, 0 if it can't be a single tap pattern:
 * it must fit DILP_W and the taps must increase in (kh, kw) order like the weights. */
static uint32_t sauria_conv3d_hw_span(const sauria_conv3d_t* params_sauria_conv3d) {
    uint32_t span = (uint32_t)(params_sauria_conv3d->fil_h - 1) * params_sauria_conv3d->dil_h * params_sauria_conv3d->inp_w + (uint32_t)(params_sauria_conv3d->fil_w - 1) * params_sauria_conv3d->dil_w + 1;
    if (span > DILP_W) return 0;
    if (params_sauria_conv3d->fil_h > 1 && (uint32_t)(params_sauria_conv3d->fil_w - 1) * params_sauria_conv3d->dil_w >= (uint32_t)params_sauria_conv3d->dil_h * params_sauria_conv3d->inp_w) return 0;
    return span;
}

static int sauria_conv3d_direct(const sauria_conv3d_t* params_sauria_conv3d) {
    uint32_t plane = (uint32_t)params_sauria_conv3d->inp_h * params_sauria_conv3d->inp_w;
    uint32_t vol = plane * params_sauria_conv3d->inp_d;

    sauria_conv_cfg_t cfg = {0};
    for (int kh = 0; kh < params_sauria_conv3d->fil_h; kh++)
        for (int kw = 0; kw < params_sauria_conv3d->fil_w; kw++)
            sauria_cfg_add_tap(&cfg, (uint32_t)kh * params_sauria_conv3d->dil_h * params_sauria_conv3d->inp_w + (uint32_t)kw * params_sauria_conv3d->dil_w);
    cfg.ystep = plane * params_sauria_conv3d->dil_d;
    cfg.ylim = cfg.ystep * params_sauria_conv3d->fil_d;
    cfg.chstep = vol;
    cfg.chlim = vol * params_sauria_conv3d->inp_c;
    cfg.out_c = params_sauria_conv3d->out_c; cfg.out_d = params_sauria_conv3d->out_d; cfg.out_h = params_sauria_conv3d->out_h; cfg.out_w = params_sauria_conv3d->out_w;
    cfg.stride_w = params_sauria_conv3d->stride_w;
    cfg.oh_step = (uint32_t)params_sauria_conv3d->inp_w * params_sauria_conv3d->stride_h;
    cfg.od_step = plane * params_sauria_conv3d->stride_d;
    cfg.preload = false;
    #ifdef __DEBUG_COMPUTE__
    mini_printf("[SAURIA][KER] conv3d direct: taps=%d span=%d\r\n", cfg.taps, cfg.span);
    #endif
    return execute_sauria_conv(&cfg);
}

/* L2 copies used by the fallback to keep the SRAM tiles intact: MATCH may reuse an
 * ifmap/weight tile in the next node iteration without reloading it, so every change
 * done below is undone. */
static uint8_t sauria_ifmap_backup[SAURIA_BANK_BYTES] __attribute__((aligned(64)));
static uint8_t sauria_weights_backup[SAURIA_BANK_BYTES] __attribute__((aligned(64)));
static uint8_t sauria_output_stage[SAURIA_BANK_BYTES] __attribute__((aligned(64)));

static inline void sauria_local_copy(void* dst, const void* src, size_t bytes) {
    if (!bytes) return;
    asm volatile("fence rw,rw" ::: "memory");
    sauria_1d_dma_transfer((void*)src, dst, bytes);
    asm volatile("fence rw,rw" ::: "memory");
}

/* Pass kd is a CONV2D (taps = kw, y = kh, ch = c with step D*H*W) over input depths
 * od*stride_d + kd*dil_d, accumulated in SRAMC with preload for kd > 0.
 * SAURIA addresses always start from 0, so before each pass:
 *   - weights slice kd (C x KH x KW x K, strided in IDHWO) is gathered at the start of SRAMB,
 *   - the input is shifted down by kd*dil_d*H*W bytes (re-read from an L2 backup).
 * out_d goes on the activation tiling counters when possible; otherwise out_d is an
 * extra software loop (with the input also shifted by od*stride_d*H*W) and every K x OH x OW
 * result is gathered in L2 and copied back to SRAMC as K x OD x OH x OW at the end. */
static int sauria_conv3d_per_kd(const sauria_conv3d_t* params_sauria_conv3d) {
    uint8_t *input = params_sauria_conv3d->input, *weight = params_sauria_conv3d->weight, *output = params_sauria_conv3d->output;
    uint16_t fil_d = params_sauria_conv3d->fil_d, out_c = params_sauria_conv3d->out_c, out_d = params_sauria_conv3d->out_d;

    uint32_t plane = (uint32_t)params_sauria_conv3d->inp_h * params_sauria_conv3d->inp_w;
    uint32_t vol = plane * params_sauria_conv3d->inp_d;
    uint32_t inp_bytes = vol * params_sauria_conv3d->inp_c;                                    // uint8
    uint32_t w_khkwk = (uint32_t)params_sauria_conv3d->fil_h * params_sauria_conv3d->fil_w * out_c;               // int8, one (c, kd)
    uint32_t w_bytes = w_khkwk * fil_d * params_sauria_conv3d->inp_c;                          // int8, whole tile
    uint32_t out_blk_bytes = (uint32_t)params_sauria_conv3d->out_h * params_sauria_conv3d->out_w * sizeof(int32_t); // one (k, od) plane
    bool od_in_hw = sauria_out_d_in_hw(out_d, params_sauria_conv3d->out_h, params_sauria_conv3d->out_w);
    uint16_t sw_od = od_in_hw ? 1 : out_d;          // software od iterations
    bool shifts_input = fil_d > 1 || sw_od > 1;

    if (inp_bytes > sizeof(sauria_ifmap_backup) ||
        (fil_d > 1 && w_bytes > sizeof(sauria_weights_backup)) ||
        (sw_od > 1 && out_blk_bytes * out_c * out_d > sizeof(sauria_output_stage))) {
        #ifdef __DEBUG_COMPUTE__
        mini_printf("[SAURIA][KER] ERROR: conv3d tile larger than the L2 backup buffers\r\n");
        #endif
        return -1;
    }

    // One CONV2D per pass: taps = kw, y = kh, ch = c
    sauria_conv_cfg_t cfg = {0};
    for (int kw = 0; kw < params_sauria_conv3d->fil_w; kw++)
        sauria_cfg_add_tap(&cfg, kw * params_sauria_conv3d->dil_w);
    cfg.ystep = (uint32_t)params_sauria_conv3d->inp_w * params_sauria_conv3d->dil_h;
    cfg.ylim = cfg.ystep * params_sauria_conv3d->fil_h;
    cfg.chstep = vol;
    cfg.chlim = inp_bytes;
    cfg.out_c = out_c; cfg.out_d = od_in_hw ? out_d : 1; cfg.out_h = params_sauria_conv3d->out_h; cfg.out_w = params_sauria_conv3d->out_w;
    cfg.stride_w = params_sauria_conv3d->stride_w;
    cfg.oh_step = (uint32_t)params_sauria_conv3d->inp_w * params_sauria_conv3d->stride_h;
    cfg.od_step = plane * params_sauria_conv3d->stride_d;

    if (shifts_input)
        sauria_local_copy(sauria_ifmap_backup, input, inp_bytes);
    if (fil_d > 1)
        sauria_local_copy(sauria_weights_backup, weight, w_bytes);

    uint32_t cur_shift = 0;                         // input bytes currently shifted out of SRAMA
    int cur_kd = -1;                                // weight slice gathered at the start of SRAMB
    bool weights_changed = false;
    int err = 0;

    for (uint16_t od = 0; od < sw_od && !err; od++) {
        for (uint16_t kd = 0; kd < fil_d; kd++) {
            uint32_t shift = ((uint32_t)od * params_sauria_conv3d->stride_d + (uint32_t)kd * params_sauria_conv3d->dil_d) * plane;
            if (shift != cur_shift) {
                sauria_local_copy(input, sauria_ifmap_backup + shift, inp_bytes - shift);
                cur_shift = shift;
            }
            if (fil_d > 1 && kd != cur_kd) {
                // IDHWO -> C x KH x KW x K of slice kd
                for (uint16_t c = 0; c < params_sauria_conv3d->inp_c; c++)
                    sauria_local_copy(weight + (uint32_t)c * w_khkwk,
                                      sauria_weights_backup + ((uint32_t)c * fil_d + kd) * w_khkwk,
                                      w_khkwk);
                cur_kd = kd;
                weights_changed = true;
            }

            cfg.preload = kd > 0;                   // accumulate on the previous kd passes
            #ifdef __DEBUG_COMPUTE__
            mini_printf("[SAURIA][KER] conv3d pass od=%d kd=%d shift=%d\r\n", od, kd, shift);
            #endif
            if (execute_sauria_conv(&cfg)) {
                err = 1;
                break;
            }
        }
        // K x OH x OW of this od -> K x OD x OH x OW in L2
        if (!err && sw_od > 1)
            for (uint16_t k = 0; k < out_c; k++)
                sauria_local_copy(sauria_output_stage + ((uint32_t)k * out_d + od) * out_blk_bytes,
                                  output + (uint32_t)k * out_blk_bytes, out_blk_bytes);
    }

    // Leave the ifmap / weights tiles as MATCH loaded them
    if (cur_shift)
        sauria_local_copy(input, sauria_ifmap_backup, inp_bytes);
    if (weights_changed)
        sauria_local_copy(weight, sauria_weights_backup, w_bytes);

    if (err) return -1;
    if (sw_od > 1)
        sauria_local_copy(output, sauria_output_stage, out_blk_bytes * out_c * out_d);
    return 0;
}

/* Reads the CONV3D tile from the context, checks it and runs it on SAURIA.
 * On success the int32 K x OD x OH x OW result is in SRAMC and *params_out describes it. */
static int sauria_conv3d_run(MatchCtx* ctx, sauria_conv3d_t* params_out) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    MatchTensor* out_t = &tensors[num_tensors - 1];
    MatchConv3DAttrs* conv_attrs = (MatchConv3DAttrs*)ctx->ops->ops[0].attrs;
    int base_tile_idx = __sauria_scratchpad_ifmap_bank__ * 5;

    sauria_conv3d_t params_sauria_conv3d;
    params_sauria_conv3d.input  = (uint8_t*)tensors[0].pt;
    params_sauria_conv3d.weight = (uint8_t*)tensors[1].pt;
    params_sauria_conv3d.output = (uint8_t*)out_t->pt;
    params_sauria_conv3d.inp_c = (uint16_t)tensors[0].tiles[base_tile_idx + 1].size;
    params_sauria_conv3d.inp_d = (uint16_t)tensors[0].tiles[base_tile_idx + 2].size;
    params_sauria_conv3d.inp_h = (uint16_t)tensors[0].tiles[base_tile_idx + 3].size;
    params_sauria_conv3d.inp_w = (uint16_t)tensors[0].tiles[base_tile_idx + 4].size;
    params_sauria_conv3d.out_c = (uint16_t)out_t->tiles[base_tile_idx + 1].size;
    params_sauria_conv3d.out_d = (uint16_t)out_t->tiles[base_tile_idx + 2].size;
    params_sauria_conv3d.out_h = (uint16_t)out_t->tiles[base_tile_idx + 3].size;
    params_sauria_conv3d.out_w = (uint16_t)out_t->tiles[base_tile_idx + 4].size;
    params_sauria_conv3d.fil_d    = (uint16_t)conv_attrs->kernel_size[0];
    params_sauria_conv3d.fil_h    = (uint16_t)conv_attrs->kernel_size[1];
    params_sauria_conv3d.fil_w    = (uint16_t)conv_attrs->kernel_size[2];
    params_sauria_conv3d.stride_d = (uint16_t)conv_attrs->strides[0];
    params_sauria_conv3d.stride_h = (uint16_t)conv_attrs->strides[1];
    params_sauria_conv3d.stride_w = (uint16_t)conv_attrs->strides[2];
    params_sauria_conv3d.dil_d    = (conv_attrs->dilation != NULL) ? (uint16_t)conv_attrs->dilation[0] : 1;
    params_sauria_conv3d.dil_h    = (conv_attrs->dilation != NULL) ? (uint16_t)conv_attrs->dilation[1] : 1;
    params_sauria_conv3d.dil_w    = (conv_attrs->dilation != NULL) ? (uint16_t)conv_attrs->dilation[2] : 1;

    #ifdef __DEBUG_COMPUTE__
    mini_printf("[SAURIA][KER] sauria_conv3d_wrapper:\r\n");
    mini_printf("  Inp Tile (C, D, H, W): (%d, %d, %d, %d)", params_sauria_conv3d.inp_c, params_sauria_conv3d.inp_d, params_sauria_conv3d.inp_h, params_sauria_conv3d.inp_w);
    mini_printf(" | Kernel (D, H, W): (%d, %d, %d)", params_sauria_conv3d.fil_d, params_sauria_conv3d.fil_h, params_sauria_conv3d.fil_w);
    mini_printf(" | Out Tile (C, D, H, W): (%d, %d, %d, %d)\r\n", params_sauria_conv3d.out_c, params_sauria_conv3d.out_d, params_sauria_conv3d.out_h, params_sauria_conv3d.out_w);
    mini_printf("  Strides (D, H, W): (%d, %d, %d) | Dilations (D, H, W): (%d, %d, %d)\r\n",
                params_sauria_conv3d.stride_d, params_sauria_conv3d.stride_h, params_sauria_conv3d.stride_w, params_sauria_conv3d.dil_d, params_sauria_conv3d.dil_h, params_sauria_conv3d.dil_w);
    mini_printf("  Inp fm ptr : %p | Weights ptr : %p | Output ptr  : %p\r\n",
                (void*)params_sauria_conv3d.input, (void*)params_sauria_conv3d.weight, (void*)params_sauria_conv3d.output);
    #endif
    // SAURIA has no base-address registers: the tiles must sit at the start of their SRAM
    if (params_sauria_conv3d.input  != (uint8_t*)(SAURIA_START_ADDRESS + SAURIA_SRAMA_OFFSET) ||
        params_sauria_conv3d.weight != (uint8_t*)(SAURIA_START_ADDRESS + SAURIA_SRAMB_OFFSET) ||
        params_sauria_conv3d.output != (uint8_t*)(SAURIA_START_ADDRESS + SAURIA_SRAMC_OFFSET)) {
        #ifdef __DEBUG_COMPUTE__
        mini_printf("[SAURIA][KER] ERROR: conv3d tiles are not at the start of the SRAMs\r\n");
        #endif
        return -1;
    }
    // SAURIA has no padding: the input tile must already contain every needed element
    if (params_sauria_conv3d.inp_d < (params_sauria_conv3d.out_d - 1) * params_sauria_conv3d.stride_d + (params_sauria_conv3d.fil_d - 1) * params_sauria_conv3d.dil_d + 1 ||
        params_sauria_conv3d.inp_h < (params_sauria_conv3d.out_h - 1) * params_sauria_conv3d.stride_h + (params_sauria_conv3d.fil_h - 1) * params_sauria_conv3d.dil_h + 1 ||
        params_sauria_conv3d.inp_w < (params_sauria_conv3d.out_w - 1) * params_sauria_conv3d.stride_w + (params_sauria_conv3d.fil_w - 1) * params_sauria_conv3d.dil_w + 1) {
        #ifdef __DEBUG_COMPUTE__
        mini_printf("[SAURIA][KER] ERROR: input tile too small for the output tile (padding?)\r\n");
        #endif
        return -1;
    }

    int err;
    if (sauria_conv3d_hw_span(&params_sauria_conv3d) && sauria_out_d_in_hw(params_sauria_conv3d.out_d, params_sauria_conv3d.out_h, params_sauria_conv3d.out_w)) {
        #ifdef __DEBUG_COMPUTE__
        mini_printf("  Mode: direct (single SAURIA run)\r\n");
        #endif
        err = sauria_conv3d_direct(&params_sauria_conv3d);
    } else {
        #ifdef __DEBUG_COMPUTE__
        mini_printf("  Mode: per-kd fallback\r\n");
        #endif
        err = sauria_conv3d_per_kd(&params_sauria_conv3d);
    }
    if (err) {
        #ifdef __DEBUG_COMPUTE__
        mini_printf("[SAURIA][KER] ERROR: conv3d not executed\r\n");
        #endif
        return -1;
    }

    *params_out = params_sauria_conv3d;
    return 0;
}

static void sauria_print_output_head(const void* output) {
    mini_printf("  First 10 output values (@ %p):", output);
    volatile const int32_t *sauria_output = (volatile const int32_t *)output;
    for (int i = 0; i < 10; i++)
        mini_printf(" %d", sauria_output[i]);
    mini_printf("\r\n");
}

void sauria_conv3d_wrapper(MatchCtx* ctx) {
    sauria_conv3d_t params_sauria_conv3d;
    if (sauria_conv3d_run(ctx, &params_sauria_conv3d)) return;
    #ifdef __DEBUG_COMPUTE__
    sauria_print_output_head(params_sauria_conv3d.output);
    #endif
}

/* ------------------------------------------------------------------ */
/*  CONV3D + multiply / add / right_shift / clip                       */
/* ------------------------------------------------------------------ */
/* Pattern sauria_conv3d_fused_mul_add_shift_clip: SAURIA computes the int32 CONV3D in
 * SRAMC (K x OD x OH x OW), then the host requantizes it in place:
 *     out = clip(((acc * mul[k]) + add[k]) >> shift)
 * tensors: input, weights, mul (int32, per out channel), add (int32, per out channel), output.
 * ops: conv3d, [cast], multiply, add, right_shift, clip. */
void sauria_conv3d_fused_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    MatchOp* ops = ctx->ops->ops;
    int num_ops = ctx->ops->num_ops;

    sauria_conv3d_t params_sauria_conv3d;
    if (sauria_conv3d_run(ctx, &params_sauria_conv3d)) return;

    const int32_t* mul = (const int32_t*)tensors[2].pt;
    const int32_t* add = (const int32_t*)tensors[3].pt;
    const int shift = ((MatchRightShiftAttrs*)ops[num_ops - 2].attrs)->right_shift;
    // generic version, however its always 0 and 255
    // const int32_t clip_min = ((MatchClipAttrs*)ops[num_ops - 1].attrs)->clip_min;
    // const int32_t clip_max = ((MatchClipAttrs*)ops[num_ops - 1].attrs)->clip_max;
    const int32_t clip_min = 0;
    const int32_t clip_max = 255;

    int32_t* out = (int32_t*)params_sauria_conv3d.output;
    const uint32_t out_dhw = (uint32_t)params_sauria_conv3d.out_d * params_sauria_conv3d.out_h * params_sauria_conv3d.out_w;
    for (uint16_t k = 0; k < params_sauria_conv3d.out_c; k++) {
        const int32_t m = mul[k], a = add[k];
        int32_t* blk = out + (uint32_t)k * out_dhw;
        for (uint32_t i = 0; i < out_dhw; i++) {
            int32_t v = (blk[i] * m + a) >> shift;
            blk[i] = v < clip_min ? clip_min : (v > clip_max ? clip_max : v);
        }
    }
    asm volatile("fence rw,rw" ::: "memory");

    #ifdef __DEBUG_COMPUTE__
    sauria_print_output_head(params_sauria_conv3d.output);
    #endif
}

void __sauria_compute__(MatchCtx* ctx) {
    if(!__sauria_node_count__) {
        sauria.base_addr = mmio_region_from_addr((uintptr_t)SAURIA_PERIPH_START_ADDRESS);
        __sauria_node_count__++;
    }
    switch (ctx->ops->ops[0].op_code) {
        case MATCH_OP_CONV2D:
            sauria_conv2d_wrapper(ctx);
            break;
        case MATCH_OP_CONV3D:
            if (ctx->ops->ops[ctx->ops->num_ops - 1].op_code == MATCH_OP_CLIP)
                sauria_conv3d_fused_wrapper(ctx);
            else
                sauria_conv3d_wrapper(ctx);
            break;
        default:
            // sauria_config(ctx);

            // sauria_start(
            //     &sauria,
            //     false, // start //Starts a computation. Self-clearing
            //     false, // auto_restart //Auto-restart (unsupported atm)
            //     false, // mem_switch //Forces the double-buffering system to swap. Self-clearing
            //     false, // mem_keep_A //Disables bouble-buffering on SRAMA, to keep the same data on all computations
            //     false, // mem_keep_B //Disables bouble-buffering on SRAMB, to keep the same data on all computations
            //     false, // mem_keep_C //Disables bouble-buffering on SRAMC, to keep the same data on all computations
            //     false // soft_rst //Soft reset for all SAURIA FSMs
            // );
            // while (sauria_done(&sauria) == 0) {}
            break;
    }
}