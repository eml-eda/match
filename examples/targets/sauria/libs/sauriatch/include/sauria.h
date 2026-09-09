#ifndef __MATCH_SAURIA_H__
#define __MATCH_SAURIA_H__

#include "sauria_regs.h"
#include "sauria.h"

#include "approx_output_tensor_0.h"  //Change with proper header file name

/*
// from sauria.h
#define X 8                                                          //SYSTOLIC ARRAY COLUMNS
#define Y 8                                                          //SYSTOLIC ARRAY ROWS
#define A_h 8                                                        //IFMAPS ROWS
#define A_w 8                                                        //IFMAPS COLUMNS
#define AB_c 4                                                       //INPUT CHANNELS
#define B_h 3                                                        //o ROWS
#define B_w 3                                                        //WEIGHTS COLUMNS
#define s 1                                                          //STRIDE
#define d 1                                                          //DILATION
#define p 1                                                          //PADDING (Sauria doesn't manage padding, so this is managed by the application)
#define A_h_padded (A_h + 2*p)                                       //IFMAPS ROWS PADDED (if p = 0 -> A_h_padded = A_h)
#define A_w_padded (A_w + 2*p)                                       //IFMAPS COLUMNS PADDED (if p = 0 -> A_w_padded = A_w)
#define C_h 8                                                        //PSUMS ROWS ((int)(floor((A_h + 2*p - B_h - (B_h-1)*(d-1))/s) + 1))
#define C_w 8                                                        //PSUMS COLUMNS ((int)(floor((A_w + 2*p - B_w - (B_w-1)*(d-1))/s) + 1))
#define C_c 4                                                        //OUTPUT CHANNELS 
#define SRAMA_W 128                                                  //SRAM A DATA WIDTH
#define IA_W 8                                                       //IFAMPS DATA WIDTH
#define SRAMA_N (SRAMA_W / IA_W)                                     //SRAM A WORDS PER ROW
#define SRAMB_W 256                                                  //SRAM B DATA WIDTH
#define IB_W 8                                                       //WEIGHTS DATA WIDTH
#define SRAMB_N (SRAMB_W / IB_W)                                     //SRAM B WORDS PER ROW
#define SRAMC_W 128                                                  //SRAM C DATA WIDTH
#define OC_W 32                                                      //PSUMS DATA WIDTH
#define SRAMC_N (SRAMC_W/OC_W)                                       //SRAM C WORDS PER ROW
#define DILP_W 64                                                    //DILATION PARAMETER WIDTH
*/                                              //DILATION PARAMETER WIDTH

#define SAURIA_PERIPH_START_ADDRESS 0x21004000
#define SAURIA_START_ADDRESS 0x52000000

void __init_sauria__(MatchCtx* ctx, void (*__inner_node_fn__)(void *), unsigned int **args);

void sauria_1d_dma_transfer(void*src, void *dst, size_t size);


void* __sauria_scratchpad_get_weights_bank__(MatchCtx* ctx);
void* __sauria_scratchpad_get_ifmap_bank__(MatchCtx* ctx);
void* __sauria_scratchpad_get_output_bank__(MatchCtx* ctx);

int sauria_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type,
    int ext_mem, int int_mem 
);

void sauria_conv2d_wrapper(MatchCtx* ctx);
void sauria_conv3d_wrapper(MatchCtx* ctx);

void execute_sauria_conv2d(
    uint16_t A_h, uint16_t A_w, uint16_t AB_c,
    uint16_t P_h, uint16_t P_w,
    uint16_t B_h, uint16_t B_w,
    uint16_t C_c,
    uint16_t s, uint16_t d,
    uint16_t X, uint16_t Y,
    uint16_t SRAMA_N, uint16_t SRAMB_N, uint16_t SRAMC_N,
    uint16_t DILP_W
);

void execute_sauria_conv3d(
    uint16_t A_d, uint16_t A_h, uint16_t A_w, uint16_t AB_c, // Input dimensions & channels
    uint16_t P_d, uint16_t P_h, uint16_t P_w,                 // External pre-padding applied to memory
    uint16_t B_d, uint16_t B_h, uint16_t B_w,                 // 3D Kernel sizes [Depth, Height, Width]
    uint16_t C_c,                                             // Output Channels
    uint16_t s, uint16_t d,                                   // Stride and Dilation
    uint16_t X, uint16_t Y,                                   // Array hardware dimensions
    uint16_t SRAMA_N, uint16_t SRAMB_N, uint16_t SRAMC_N,     // Memory bank alignment parameters
    uint16_t DILP_W                                           // Dilation pattern register width
);

void sauria_compute(MatchCtx* ctx);

#endif  /*__MATCH_SAURIA_H__*/