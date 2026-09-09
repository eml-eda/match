
#include "car_memory_map.h"
#include "io.h"
#include "dif/clint.h"
#include "dif/uart.h"
#include "params.h"
#include "regs/cheshire.h"
#include "regs/system_timer.h"
#include "util.h"
#include "car_util.h"
#include "printf.h"
#include "dif/dma.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "sauria_regs.h"
#include "sauria.h"

#include "input_tensor.h"
#include "weight_tensor.h"
#include "approx_output_tensor_0.h"  //Change with proper header file name

#define SAURIA_PERIPH_START_ADDRESS 0x21004000
#define SAURIA_START_ADDRESS 0x52000000

int32_t psums[C_c][C_h][C_w] __attribute__ ((aligned (4))) = {0};

sauria_t sauria;

int main(void) {

  // Init the HW
  car_init_start();

  //sauria_t sauria;
  sauria.base_addr = mmio_region_from_addr((uintptr_t)SAURIA_PERIPH_START_ADDRESS);

  int ifmap_h, ifmap_w, in_ch;
  int wei_h, wei_w, out_ch;
  int psums_h, psums_w;

  int cycles_sauria, cycles_cva6, cycles_sauria_cfg, cycles_dma; 

  writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_RESET_LO_OFFSET);
  writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_START_LO_OFFSET);
  // ------------------------------------------------------------
  // Send IFMAPS to SAURIA's IFMAPS SRAM through DMA
  // ------------------------------------------------------------

  int8_t *inputs_ptr = &input_tensor[0][0][0];
  uint32_t *srama_ptr = (uint32_t *)(SAURIA_START_ADDRESS+SAURIA_SRAMA_OFFSET);


  sys_dma_blk_memcpy( (uintptr_t)(void *)srama_ptr,
                      (uintptr_t)(void *)inputs_ptr, 
                      (AB_c*A_h_padded*A_w_padded) - 1,
                      DMA_CONF_DECOUPLE_NONE
                    );
    

  // ------------------------------------------------------------
  // Send Weights to SAURIA's Weights SRAM through DMA
  // ------------------------------------------------------------

  int c = 0;
  for (in_ch = 0; in_ch < AB_c; in_ch++) {
    for (wei_h = 0; wei_h < B_h; wei_h++) {
      for (wei_w = 0; wei_w < B_w; wei_w++) {

          int8_t *weights_ptr = &weight_tensor[0][in_ch][wei_h][wei_w];
          uint32_t *sramb_ptr = (uint32_t *)(SAURIA_START_ADDRESS+SAURIA_SRAMB_OFFSET+(c*C_c));
          
          sys_dma_blk_memcpy( (uintptr_t)(void *)sramb_ptr,
                              (uintptr_t)(void *)weights_ptr, 
                              (AB_c*B_h*B_w) - 1,
                              DMA_CONF_DECOUPLE_NONE
                            );
            
          c++;
      }
    }
  }

  writed(0, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CFG_LO_OFFSET);
  cycles_dma = readd(CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CNT_LO_OFFSET);

  writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_RESET_LO_OFFSET);
  writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_START_LO_OFFSET);
  // ------------------------------------------------------------
  // CONFIGURATION REGISTERS' PARAMETERS
  // ------------------------------------------------------------
  //Control & Status

  bool start = false; //Starts a computation. Self-clearing

  bool done = false; //High when SAURIA finishes a computation. Can be toggled by writing a 1

  bool idle = false; //SAURIA is idle

  bool ready = false; //SAURIA is ready to start a new computation

  bool auto_restart = false; //Auto-restart (unsupported atm)

  bool mem_switch = false; //Forces the double-buffering system to swap. Self-clearing

  bool mem_keep_A = false; //Disables bouble-buffering on SRAMA, to keep the same data on all computations

  bool mem_keep_B = false; //Disables bouble-buffering on SRAMB, to keep the same data on all computations

  bool mem_keep_C = false; //Disables bouble-buffering on SRAMC, to keep the same data on all computations

  bool soft_rst = false; //Soft reset for all SAURIA FSMs

  bool global_ien = true; //Global interrupt enable

  bool done_ien = true; //Interrupt enable for SAURIA's done signal

  bool done_intr_status = false; //Interrupt status flag - Can be toggled by writing a 1

  //Main FSM

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

  writed(0, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CFG_LO_OFFSET);
  cycles_sauria_cfg = readd(CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CNT_LO_OFFSET);

  
  writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_RESET_LO_OFFSET);
  writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_START_LO_OFFSET);
  // ------------------------------------------------------------
  // WRITE CONFIG REGISTER 0x0 AND START SAURIA'S COMPUTATION
  // ------------------------------------------------------------

  start = true;
  sauria_start(&sauria, start, auto_restart, mem_switch, mem_keep_A, mem_keep_B, mem_keep_C, soft_rst);
  
  while (sauria_done(&sauria) == 0) {}

  writed(0, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CFG_LO_OFFSET);
  cycles_sauria = readd(CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CNT_LO_OFFSET);


  writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_RESET_LO_OFFSET);
  writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_START_LO_OFFSET);

  int32_t psums[C_c][C_h][C_w] __attribute__ ((aligned (4))) = {0};

  // Perform the convolution
  for (int out_ch = 0; out_ch < C_c; out_ch++) {
      for (int i = 0; i <= A_h_padded - B_h; i++) {
          for (int j = 0; j <= A_w_padded - B_w; j++) {
              for (int in_ch = 0; in_ch < AB_c; in_ch++) {
                  for (int wi = 0; wi < B_h; wi++) {
                      for (int wj = 0; wj < B_w; wj++) {
                          psums[out_ch][i][j] += input_tensor[in_ch][i + wi][j + wj] * weight_tensor[out_ch][in_ch][wi][wj];
                      }
                  }
              }
          }
      }
  }

  writed(0, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CFG_LO_OFFSET);
  cycles_cva6 = readd(CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CNT_LO_OFFSET);

  uint32_t rtc_freq   = *reg32(&__base_regs, CHESHIRE_RTC_FREQ_REG_OFFSET);
  uint64_t reset_freq = clint_get_core_freq(rtc_freq, 2500);
  uart_init(&__base_uart, reset_freq, 115200);

  printf("%d\n", psums[0][0][0]);
  uart_write_flush(&__base_uart);
  printf("SAURIA cycles: %d\n", cycles_sauria);
  uart_write_flush(&__base_uart);
  printf("SAURIA config cycles: %d\n", cycles_sauria_cfg);
  uart_write_flush(&__base_uart);
  printf("DMA cycles: %d\n", cycles_dma);
  uart_write_flush(&__base_uart);
  printf("CVA6 cycles: %d\n", cycles_cva6);
  uart_write_flush(&__base_uart);

  uint32_t* sauria_start = (int*)(SAURIA_PERIPH_START_ADDRESS);

  sauria_start[0] = cycles_sauria;
  sauria_start[1] = cycles_sauria_cfg;
  sauria_start[2] = cycles_dma;
  sauria_start[3] = cycles_cva6;
  
  return EXIT_SUCCESS;

}  
