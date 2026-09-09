#include <sauria/sauria.h>

static int32_t psums[C_c][C_h][C_w] __attribute__ ((aligned (4))) = {0};
static sauria_t sauria;
static int __sauria_node_count__ = 0;

void __init_sauria__(MatchCtx* ctx, void (*__inner_node_fn__)(void *), unsigned int **args) {
    if(!__sauria_node_count__) {
        sauria.base_addr = mmio_region_from_addr((uintptr_t)SAURIA_PERIPH_START_ADDRESS);
    }
    __sauria_node_count__++;
    __inner_node_fn__(args);
}

void sauria_1d_dma_transfer(void*src, void *dst, size_t size) {
    sys_dma_blk_memcpy( (uintptr_t)(void *)dst,
        (uintptr_t)(void *)src, 
        size - 1,
        DMA_CONF_DECOUPLE_NONE
    );
}

static void sauria_copy_tensor_recursive(
    uint8_t* dst_base,
    uint8_t* src_base,
    const size_t* src_strides,
    const size_t* dst_strides,
    const size_t* copy_sizes,
    int dim,
    int num_dims,
    size_t row_bytes
) {
    for (size_t idx = 0; idx < copy_sizes[dim]; idx++) {
        uint8_t* src = src_base + idx * src_strides[dim];
        uint8_t* dst = dst_base + idx * dst_strides[dim];

        if (dim == num_dims - 1) {
            sauria_1d_dma_transfer(src, dst, row_bytes);
        } else {
            sauria_copy_tensor_recursive(
                dst,
                src,
                src_strides,
                dst_strides,
                copy_sizes,
                dim + 1,
                num_dims,
                row_bytes
            );
        }
    }
}

int sauria_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem 
) {
    (void)ctx;

    asm volatile("fence rw,rw" ::: "memory");

    if (ext_mem != MEM_L2 || int_mem != MEM_L1_PULPD) {
        exit(1);
    }

    if (!tensor || !tensor->num_dims) {
        return 0;
    }

    if (tensor->num_dims > 5) {
        exit(1);
    }

    size_t bytes_per_element = (size_t)(tensor->bits / 8);
    size_t l1_sizes[5] = {0};
    size_t l2_sizes[5] = {0};
    size_t l1_strides[5] = {0};
    size_t l2_strides[5] = {0};

    for (int idx = 0; idx < tensor->num_dims; idx++) {
        l1_sizes[idx] = (size_t)tensor->tiles[MEM_L1_PULPD * tensor->num_dims + idx].size;
        l2_sizes[idx] = (size_t)tensor->tiles[MEM_L2 * tensor->num_dims + idx].size;
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
        size_t row_bytes = l1_sizes[tensor->num_dims - 1] * bytes_per_element;
        sauria_copy_tensor_recursive(
            dst,
            src,
            match_transfer_type == MATCH_SW_LOAD_TENSOR ? l2_strides : l1_strides,
            match_transfer_type == MATCH_SW_LOAD_TENSOR ? l1_strides : l2_strides,
            l1_sizes,
            0,
            tensor->num_dims,
            row_bytes
        );
    }

    asm volatile("fence rw,rw" ::: "memory");

    return (int)transferred_bytes;
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


// Assuming sauria hardware driver object instance exists
// extern sauria_t sauria;
void execute_sauria_conv2d(
    uint16_t A_h, uint16_t A_w, uint16_t AB_c,
    uint16_t P_h, uint16_t P_w,
    uint16_t B_h, uint16_t B_w,
    uint16_t C_c,
    uint16_t s, uint16_t d,
    uint16_t X, uint16_t Y,
    uint16_t SRAMA_N, uint16_t SRAMB_N, uint16_t SRAMC_N,
    uint16_t DILP_W
) {
    execute_sauria_conv3d(
        1, A_h, A_w, AB_c,
        0, P_h, P_w,
        1, B_h, B_w,
        C_c,
        s, d,
        X, Y,
        SRAMA_N, SRAMB_N, SRAMC_N,
        DILP_W
    );
}

void execute_sauria_conv3d(
    uint16_t A_d, uint16_t A_h, uint16_t A_w, uint16_t AB_c, // Input dimensions & channels
    uint16_t P_d, uint16_t P_h, uint16_t P_w,                 // External pre-padding applied to memory
    uint16_t B_d, uint16_t B_h, uint16_t B_w,                 // 3D Kernel sizes [Depth, Height, Width]
    uint16_t C_c,                                             // Output Channels
    uint16_t s, uint16_t d,                                   // Stride and Dilation
    uint16_t X, uint16_t Y,                                   // Array hardware dimensions
    uint16_t SRAMA_N, uint16_t SRAMB_N, uint16_t SRAMC_N,     // Memory bank alignment parameters
    uint16_t DILP_W                                           // Dilation pattern register width
) {
    // ------------------------------------------------------------
    // 1. DIMENSION SWAP DETECTION & SHAPE COMPUTATION
    // ------------------------------------------------------------
    // Calculate 3D output spatial dimensions
    uint16_t B_d_eff = (1 + (B_d - 1) * d);
    uint16_t B_h_eff = (1 + (B_h - 1) * d);
    uint16_t B_w_eff = (1 + (B_w - 1) * d);

    uint16_t A_d_padded = A_d + 2 * P_d;
    uint16_t A_h_padded = A_h + 2 * P_h;
    uint16_t A_w_padded = A_w + 2 * P_w;

    uint16_t C_d = (A_d_padded - B_d_eff) / s + 1;
    uint16_t C_h = (A_h_padded - B_h_eff) / s + 1;
    uint16_t C_w = (A_w_padded - B_w_eff) / s + 1;

    // Dimension Swap: If B_d > 1 and B_h == 1 (e.g. [3,1,1] or [2,1,1]),
    // swap D and H axes in execution so depth iteration count becomes 1.
    bool swap_dh = (B_d > 1 && B_h == 1);

    uint16_t B_d_exec = swap_dh ? B_h : B_d; // Becomes 1 when swapped
    uint16_t B_h_exec = swap_dh ? B_d : B_h; // Becomes 3 or 2 when swapped
    
    uint16_t C_d_exec = swap_dh ? C_h : C_d;
    uint16_t C_h_exec = swap_dh ? C_d : C_h;

    uint16_t A_h_exec_padded = swap_dh ? A_d_padded : A_h_padded;

    uint16_t B_h_exec_eff = (1 + (B_h_exec - 1) * d);

    // ------------------------------------------------------------
    // 2. TILING & REPETITION COUNTERS
    // ------------------------------------------------------------
    uint16_t incntlim = (B_d_exec * B_h_exec * B_w * AB_c - 1); // MAC counter limit per pass

    int X_used;
    if (C_c < X) {
        X_used = C_c;
    } else if (C_c == X) {
        X_used = X;
    } else {
        for (int x = X; x > 0; x--) {
            if (C_c % x == 0) {
                X_used = x;
                break;
            }
        }
    }
    uint16_t act_reps = (int)(ceil((double)C_c / (double)X_used));

    int Y_used;
    if (C_w < Y) {
        Y_used = C_w;
    } else if (C_w == Y) {
        Y_used = Y;
    } else {
        for (int y = Y; y > 0; y--) {
            if (C_w % y == 0) {
                Y_used = y;
                break;
            }
        }
    }
    
    // Weight repetitions scaled by the execution depth kernel size
    uint16_t wei_reps = (int)(ceil((double)C_w / (double)Y_used) * ceil((double)C_h_exec / 1.0) * B_d_exec);

    // Hardware masks and configurations
    bool global_ien = true;
    bool done_ien = true;
    uint8_t neg_thres = 0;
    uint16_t res_mask = 0xFFFF;
    uint8_t appr_mask = 255;
    bool sram_deepsleep = false;
    bool sram_powergate = false;

    // ------------------------------------------------------------
    // 3. IFMAP FEEDER CONFIGURATION
    // ------------------------------------------------------------
    uint16_t xlim;
    if ((s == 1) && (d == 1) && (B_h_exec == 1) && (B_w == 1) && ((AB_c % SRAMA_N) == 0) && ((Y_used % SRAMA_N) == 0)) {
        xlim = Y_used;
    } else {
        xlim = ((1 + (Y_used - 1) * s) + B_w_eff + 1 - (B_w_eff % 2) + SRAMA_N);
    }

    uint16_t xstep = SRAMA_N;
    uint16_t ylim = (A_w_padded * B_h_exec_eff);
    uint16_t ystep = (A_w_padded * d);
    
    uint16_t chlim = (A_w_padded * A_h_exec_padded * AB_c);
    uint16_t chstep = (A_w_padded * A_h_exec_padded);

    uint16_t til_xlim = (int)((ceil((double)C_w / (double)Y_used)) * Y_used * s);
    uint16_t til_xstep = (Y_used * s);
    uint16_t til_ylim = (int)((ceil(C_h_exec / 1.0)) * A_w_padded * 1 * s);
    uint16_t til_ystep = (A_w_padded * 1 * s);

    // Address stride between adjacent 2D execution depth planes in SRAM
    uint32_t depth_plane_stride = (A_w_padded * A_h_exec_padded * AB_c);

    // Generate dilation pattern
    uint64_t Dil_pat = 0;
    for (int i = 0; i < DILP_W; i++) {
        if ((i % d == 0) && ((i / d) < B_w)) {
            Dil_pat |= (1ULL << (DILP_W - 1 - i));
        }
    }

    // Generate row active pattern
    uint8_t rows_active = 0;
    bool rows_active_arr[Y] = {false};
    for (int j = 0; j < Y; j++) {
        if (j < Y_used) {
            rows_active |= (1 << (Y - 1 - j));
            rows_active_arr[j] = true;
        }
    }

    uint8_t lwoffs[Y];
    for (int i = 0; i < Y; i++) {
        lwoffs[i] = rows_active_arr[i] ? i * s : 0;
    }

    // ------------------------------------------------------------
    // 4. WEIGHT FEEDER CONFIGURATION
    // ------------------------------------------------------------
    uint32_t wlim = (C_c * B_d_exec * B_w * B_h_exec * AB_c);
    uint16_t wstep = C_c;

    bool waligned = (((C_c % SRAMB_N) == 0) && (X_used == SRAMB_N));
    uint16_t klim = (!waligned) ? (SRAMB_N + 1) : 1;

    uint16_t kstep = SRAMB_N;
    uint16_t til_klim = C_c;
    uint16_t til_kstep = X_used;

    uint8_t cols_active = 0;
    bool cols_active_arr[X] = {false};
    for (int i = 0; i < X; i++) {
        if (i < X_used) {
            cols_active |= (1 << (X - 1 - i));
            cols_active_arr[i] = true;
        }
    }

    // ------------------------------------------------------------
    // 5. PS MANAGER CONFIGURATION
    // ------------------------------------------------------------
    uint16_t X_tiles = (int)(ceil((double)C_w / (double)Y_used));
    uint16_t Y_tiles = C_h_exec;
    uint16_t K_tiles = (C_c / X_used);

    uint32_t ncontexts = (uint32_t)X_tiles * Y_tiles * K_tiles * C_d_exec;
    
    uint16_t cxlim = (Y_used + SRAMC_N);
    uint16_t cxstep = SRAMC_N;
    uint16_t cklim = (C_w * C_h_exec * X_used);
    uint16_t ckstep = (C_w * C_h_exec);
    uint16_t til_cylim = (C_w * C_h_exec);
    uint16_t til_cystep = Y_used;
    uint16_t til_cklim = (C_w * C_h_exec * C_c);
    uint16_t til_ckstep = (C_w * C_h_exec * X_used);
    uint8_t inactive_cols = (X - X_used);

    // ------------------------------------------------------------
    // 6. ACCELERATOR EXECUTION LOOPS
    // ------------------------------------------------------------
    for (int od = 0; od < C_d_exec; od++) {
        for (int kd = 0; kd < B_d_exec; kd++) {
            
            // Calculate slice depth index in pre-padded feature map
            int input_depth_idx = od * s + kd * d;
            
            // Enable accumulator preloading only when slicing deep kernels (kd > 0)
            bool slice_preload_en = (kd > 0);

            // Compute memory offsets for current slice
            uint32_t ifmap_base_addr = input_depth_idx * depth_plane_stride;
            uint32_t weight_base_addr = kd * (B_h_exec * B_w * AB_c * C_c);

            // Program Memory Base Pointers
            sauria_set_ifmap_ptr(&sauria, ifmap_base_addr);
            sauria_set_weight_ptr(&sauria, weight_base_addr);

            // Write Control Registers
            sauria_global_ien(&sauria, global_ien);
            sauria_done_ien(&sauria, done_ien);
            sauria_reg0x200(&sauria, incntlim, act_reps, wei_reps);
            sauria_reg0x204(&sauria, wei_reps, neg_thres, res_mask, appr_mask);
            sauria_reg0x208(&sauria, appr_mask, sram_deepsleep, sram_powergate);
            sauria_reg0x400(&sauria, xlim, xstep);
            sauria_reg0x404(&sauria, ylim, ystep);
            sauria_reg0x408(&sauria, chlim, chstep);
            sauria_reg0x40C(&sauria, til_xlim, til_xstep);
            sauria_reg0x410(&sauria, til_ylim, til_ystep);
            sauria_reg0x414(&sauria, Dil_pat);
            sauria_reg0x418(&sauria, Dil_pat);
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
            
            // Set partial sum accumulator preload flag
            sauria_reg0x810(&sauria, inactive_cols, slice_preload_en);

            // Dispatch Hardware Execution
            sauria_start(&sauria);
            sauria_wait_done(&sauria);
        }
    }
}

void sauria_conv3d_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;

    // 1. Extract Tensor Memory Pointers
    void *input  = tensors[0].pt;
    void *weight = tensors[1].pt;

    // Program physical base pointers into SAURIA hardware driver
    sauria_set_ifmap_ptr(NULL, (uint32_t)(uintptr_t)input);
    sauria_set_weight_ptr(NULL, (uint32_t)(uintptr_t)weight);

    // 2. Extract Input Tensor Dimensions (NDHWC Layout)
    // Memory is pre-padded by host node -> sizes represent A_d_padded, A_h_padded, A_w_padded
    int base_tile_idx = MEM_L1_PULPD * 5; 
    uint16_t inp_height_padded = (uint16_t)tensors[0].tiles[base_tile_idx + 1].size;
    uint16_t inp_width_padded  = (uint16_t)tensors[0].tiles[base_tile_idx + 2].size;
    uint16_t inp_depth_padded  = (uint16_t)tensors[0].tiles[base_tile_idx + 3].size;
    uint16_t inp_ch            = (uint16_t)tensors[0].tiles[base_tile_idx + 4].size;

    // 3. Extract Output Tensor Dimensions
    int out_base_tile_idx = MEM_L1_PULPD * 5;
    uint16_t out_ch = (uint16_t)tensors[num_tensors - 1].tiles[out_base_tile_idx + 4].size;

    // 4. Zero Padding (Handled explicitly by preceding host node)
    uint16_t pad_d = 0;
    uint16_t pad_h = 0;
    uint16_t pad_w = 0;

    // 5. Extract Conv3D Layer Attributes
    MatchConv3DAttrs* conv_attrs = (MatchConv3DAttrs*)ctx->ops->ops[0].attrs;
    
    // kernel_size: [filter_depth, filter_height, filter_width]
    uint16_t filter_depth  = (uint16_t)conv_attrs->kernel_size[0];
    uint16_t filter_height = (uint16_t)conv_attrs->kernel_size[1];
    uint16_t filter_width  = (uint16_t)conv_attrs->kernel_size[2];

    // strides & dilations
    uint16_t stride_x = (uint16_t)conv_attrs->strides[2]; 
    uint16_t dilation_x = (conv_attrs->dilations != NULL) ? (uint16_t)conv_attrs->dilations[0] : 1;

    // 6. Hardcoded Accelerator Hardware Parameters
    uint16_t SAURIA_X = 8;        // PE Array Columns
    uint16_t SAURIA_Y = 8;        // PE Array Rows
    uint16_t SRAMA_N  = 4;        // Memory Bank Alignment Input
    uint16_t SRAMB_N  = 4;        // Memory Bank Alignment Weight
    uint16_t SRAMC_N  = 4;        // Memory Bank Alignment PSum
    uint16_t DILP_W   = 64;       // Dilation Pattern Register Width

#if DEBUG_CLUSTER_LIB
    smp_printf("[SAURIA][KER] sauria_conv3d_wrapper:\r\n");
    smp_printf("  Pre-Padded Inp Tile (D,H,W,C): (%d, %d, %d, %d)\r\n", 
               inp_depth_padded, inp_height_padded, inp_width_padded, inp_ch);
    smp_printf("  Kernel              (D,H,W)  : (%d, %d, %d)\r\n", 
               filter_depth, filter_height, filter_width);
    smp_printf("  Out Ch                       : %d\r\n", out_ch);
#endif

    // 7. Dispatch to SAURIA Execution Controller FSM
    execute_sauria_conv3d(
        inp_depth_padded, inp_height_padded, inp_width_padded, inp_ch,
        pad_d, pad_h, pad_w,
        filter_depth, filter_height, filter_width,
        out_ch,
        stride_x, dilation_x,
        SAURIA_X, SAURIA_Y,
        SRAMA_N, SRAMB_N, SRAMC_N,
        DILP_W
    );
}

void sauria_conv2d_wrapper(MatchCtx* ctx) {
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;

    void *input  = tensors[0].pt;
    void *weight = tensors[1].pt;

    sauria_set_ifmap_ptr(NULL, (uint32_t)(uintptr_t)input);
    sauria_set_weight_ptr(NULL, (uint32_t)(uintptr_t)weight);

    int base_tile_idx = MEM_L1_PULPD * 4;
    uint16_t inp_height_padded = (uint16_t)tensors[0].tiles[base_tile_idx + 1].size;
    uint16_t inp_width_padded  = (uint16_t)tensors[0].tiles[base_tile_idx + 2].size;
    uint16_t inp_ch            = (uint16_t)tensors[0].tiles[base_tile_idx + 3].size;
    uint16_t out_ch            = (uint16_t)tensors[num_tensors - 1].tiles[base_tile_idx + 3].size;

    MatchConv2DAttrs* conv_attrs = (MatchConv2DAttrs*)ctx->ops->ops[0].attrs;
    uint16_t filter_height = (uint16_t)conv_attrs->kernel_size[0];
    uint16_t filter_width  = (uint16_t)conv_attrs->kernel_size[1];
    uint16_t stride_x      = (uint16_t)conv_attrs->strides[1];
    uint16_t dilation_x    = (conv_attrs->dilations != NULL) ? (uint16_t)conv_attrs->dilations[1] : 1;

#if DEBUG_CLUSTER_LIB
    smp_printf("[SAURIA][KER] sauria_conv2d_wrapper:\r\n");
    smp_printf("  Pre-Padded Inp Tile (H,W,C): (%d, %d, %d)\r\n",
               inp_height_padded, inp_width_padded, inp_ch);
    smp_printf("  Kernel              (H,W)  : (%d, %d)\r\n",
               filter_height, filter_width);
    smp_printf("  Out Ch                     : %d\r\n", out_ch);
#endif

    execute_sauria_conv2d(
        inp_height_padded, inp_width_padded, inp_ch,
        0, 0,
        filter_height, filter_width,
        out_ch,
        stride_x, dilation_x,
        8, 8,
        4, 4, 4,
        64
    );
}

void sauria_config(MatchCtx* ctx) {

    //Main FSM
    bool global_ien = true; //Global interrupt enable
    bool done_ien = true; //Interrupt enable for SAURIA's done signal
    bool done_intr_status = false; //Interrupt status flag - Can be toggled by writing a 1
    
    uint16_t incntlim = (1*B_h*B_w*AB_c - 1); //Total number of MACs of the computation
    int X_used;
    if(C_c < X){
        X_used = C_c;
    }else if (C_c == X){
        X_used = X;
    }else{
        for(int x = X; x > 0; x--){
            if(C_c%x == 0){
                X_used = x;
                break;
            }
        }
    }
    uint16_t act_reps = (int)(ceil((double)C_c/(double)X_used)); //Number of ifmap tiling iterations

    int Y_used;
    if(C_w < Y){
        Y_used = C_w;
    }else if (C_w == Y){
        Y_used = Y;
    }else{
        for(int y = Y; y > 0; y--){
            if(C_w%y == 0){
                Y_used = y;
                break;
            }
        }
    }
    uint16_t wei_reps = (int)(ceil((double)C_w/(double)Y_used)*ceil((double)C_h/(double)1)); //Number of weight tiling iterations

    uint8_t neg_thres = 0; //Zero negligence threshold for the systolic array PEs

    uint16_t res_mask = (pow(2, 16) - 1) ; //Mask for the multiplier to select result precision
    /*2^16 - 1 = 65535 - > 1111 1111 1111 1111*/

    uint8_t appr_mask = 255; //Mask for the multiplier to select result approximation

    bool sram_deepsleep = false;
    bool sram_powergate = false;

    //Ifmap Feeder

    //Effective kernel size (dilation)
    uint16_t B_w_eff = (1 + (B_w - 1)*d);
    uint16_t B_h_eff = (1 + (B_h - 1)*d);

    uint16_t xlim; //X counter limit
    //Aligned activations condition => Only when 1x1 convolution and no d or ss is applied
    if ((s == 1) && (d == 1) && (B_h == 1) && (B_w == 1) && ((AB_c % SRAMA_N) == 0) && ((Y_used % SRAMA_N) == 0)){
        xlim = Y_used;
    }else{
        xlim = ((1 + (Y_used - 1)*s) + B_w_eff + 1 - (B_w_eff % 2) + SRAMA_N);
    }

    uint16_t xstep = SRAMA_N; //X counter step
    uint16_t ylim = (A_w_padded*B_h_eff); //Y counter limit
    uint16_t ystep = (A_w_padded*d); //Y counter step
    uint16_t chlim = (A_w_padded*A_h_padded*AB_c); //Channel counter limit
    uint16_t chstep = (A_w_padded*A_h_padded); //Channel counter step
    uint16_t til_xlim = (int)((ceil((double)C_w/(double)Y_used))*Y_used*s); //Tiling X counter limit
    uint16_t til_xstep = (Y_used*s); //Tiling X counter step
    uint16_t til_ylim = (int)((ceil(C_h/1))*A_w_padded*1*s); //Tiling Y counter limit
    uint16_t til_ystep = (A_w_padded*1*s); //Tiling Y counter step
    //Dilation pattern generation
    uint64_t  Dil_pat = 0; //64-bit dilation pattern describing the horizontal shape of the convolution kernel
    for (int i = 0; i < DILP_W; i++) {
        if ((i % d == 0) && ((i / d) < B_w)) {
            Dil_pat |= (1ULL << (DILP_W - 1 - i));  // Set the corresponding bit to 1
        }
    }

    // Generate row mask and array
    uint8_t rows_active = 0; //Pattern of active rows (each bit represents one row out of 8)
    bool rows_active_arr[Y];
    memset(rows_active_arr, 0, sizeof(rows_active_arr)); // Initialize array to false (0)
    for (int j = 0; j < Y; j++) {
        if (j < Y_used) {
            rows_active |= (1 << (Y - 1 - j));  // Set the corresponding bit to 1
            rows_active_arr[j] = true;          // Set the array element to 1 (true)
        }
    }

    //Local woffs
    uint8_t lwoffs[Y]; //Local word offset for Rows [0:7]
    for (int i = 0; i < Y; i++) {
        lwoffs[i] = rows_active_arr[i] ? i * s : 0;
    }

    //Weight Feeder

    uint16_t wlim = (C_c*B_w*B_h*AB_c); //Weight counter limit
    uint16_t wstep = C_c; //Weight counter step

    //Aligned weights condition => feeder optimization   
    bool waligned = (((C_c % SRAMB_N) == 0) && (X_used == SRAMB_N)); //Aligned weights flag - Indicate the weights are perfectly aligned in SRAM memory reads
    uint16_t klim; //Out-Channel counter limit
    if (!waligned) {
        klim = (SRAMB_N+1);
    }else{
        klim = 1;
    }

    uint16_t kstep = SRAMB_N; //Out-Channel counter step
    uint16_t til_klim = C_c; //Tiling Out-Channel counter limit
    uint16_t til_kstep = X_used; //Tiling Out-Channel counter step
    // Generate column mask and array
    uint8_t cols_active = 0; //Pattern of active columns (each bit represents one column out of 8)
    bool cols_active_arr[X];
    memset(cols_active_arr, 0, sizeof(cols_active_arr)); // Initialize array to false (0)
    for (int i = 0; i < X; i++) {
        if (i < X_used) {
            cols_active |= (1 << (X - 1 - i));  // Set the corresponding bit to 1
            cols_active_arr[i] = true;          // Set the array element to 1 (true)
        }
    }

    //PS Manager

    //Internal tiles for SAURIA execution
    uint16_t X_tiles = (int)(ceil((double)C_w/(double)Y_used));
    uint16_t Y_tiles = C_h;
    uint16_t K_tiles = (C_c/X_used);

    //Number of context switches
    uint16_t ncontexts = (X_tiles*Y_tiles*K_tiles); //Number of computation contexts (total tiling iterations)
    uint16_t cxlim = (Y_used + SRAMC_N); //X counter limit for partial sums
    uint16_t cxstep = SRAMC_N; //X counter step for partial sums
    uint16_t cklim = (C_w*C_h*X_used); //Out-Channel counter limit for partial sums
    uint16_t ckstep = (C_w*C_h); //Out-Channel counter step for partial sums
    uint16_t til_cylim = (C_w*C_h); //Tiling Y counter limit for partial sums
    uint16_t til_cystep = Y_used; //Tiling Y counter step for partial sums
    uint16_t til_cklim = (C_w*C_h*C_c); //Tiling Out-Channel counter limit for partial sums
    uint16_t til_ckstep = (C_w*C_h*X_used); //Tiling Out-Channel counter step for partial sums
    uint8_t inactive_cols = (X-X_used); //Number of inactive columns
    bool preload_en = false; //Preload enable - Set to 1 to preload the MAC Accumulators with the initial data on the SRAM


    // ------------------------------------------------------------
    // WRITE CONFIGURATION REGISTERS
    // ------------------------------------------------------------

    sauria_global_ien(&sauria, global_ien);
    sauria_done_ien(&sauria, done_ien);
    sauria_reg0x200(&sauria, incntlim, act_reps, wei_reps); //
    sauria_reg0x204(&sauria, wei_reps, neg_thres, res_mask, appr_mask); //
    sauria_reg0x208(&sauria, appr_mask, sram_deepsleep, sram_powergate);
    sauria_reg0x400(&sauria, xlim, xstep);
    sauria_reg0x404(&sauria, ylim, ystep);
    sauria_reg0x408(&sauria, chlim, chstep);
    sauria_reg0x40C(&sauria, til_xlim, til_xstep);
    sauria_reg0x410(&sauria, til_ylim, til_ystep);
    sauria_reg0x414(&sauria, Dil_pat);
    sauria_reg0x418(&sauria, Dil_pat);
    sauria_reg0x41C(&sauria, rows_active, lwoffs[0], lwoffs[1], lwoffs[2]);
    sauria_reg0x420(&sauria, lwoffs[3], lwoffs[4], lwoffs[5], lwoffs[6]);
    sauria_reg0x424(&sauria, lwoffs[7]);
    sauria_reg0x600(&sauria, wlim, wstep);
    sauria_reg0x604(&sauria, klim, kstep);
    sauria_reg0x608(&sauria, til_klim, til_kstep);
    sauria_reg0x60C(&sauria, cols_active, waligned);
    sauria_reg0x800(&sauria, ncontexts, cxlim, cxstep); //
    sauria_reg0x804(&sauria, cxstep, cklim, ckstep); //
    sauria_reg0x808(&sauria, ckstep, til_cylim, til_cystep); //
    sauria_reg0x80C(&sauria, til_cystep, til_cklim, til_ckstep, inactive_cols); //
    sauria_reg0x810(&sauria, inactive_cols, preload_en); //
}

void sauria_compute(MatchCtx* ctx) {
    switch (ctx->ops->ops[0].op_code) {
        case MATCH_OP_CONV2D:
            sauria_conv2d_wrapper(ctx);
            break;
        case MATCH_OP_CONV3D:
            sauria_conv3d_wrapper(ctx);
            break;
        default:
            sauria_config(ctx);

            sauria_start(
                &sauria,
                false, // start //Starts a computation. Self-clearing
                false, // auto_restart //Auto-restart (unsupported atm)
                false, // mem_switch //Forces the double-buffering system to swap. Self-clearing
                false, // mem_keep_A //Disables bouble-buffering on SRAMA, to keep the same data on all computations
                false, // mem_keep_B //Disables bouble-buffering on SRAMB, to keep the same data on all computations
                false, // mem_keep_C //Disables bouble-buffering on SRAMC, to keep the same data on all computations
                false // soft_rst //Soft reset for all SAURIA FSMs
            );
            while (sauria_done(&sauria) == 0) {}
            break;
    }
}