#ifndef __MATCHMAXWELL_H__
#define __MATCHMAXWELL_H__

#include <match_dimensions.h>
#include <match_kernel.h>


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include <function.h>
#include <core_v_mini_mcu.h>    //Comment if compiling outside heepnosis
#include <heepnosis.h>          //Comment if compilfing outside heepnosis
//#define MAXWELL_START_ADDRESS  0x0 //Uncomment if compiling outside heepnosis
//Register Addresses
#define AVG_KERNEL_REG  0x1088  //0b1000010001000
#define MAX_REG         0x1084 //0b1000010000100
#define BIAS_REG        0x1082 //0b1000010000010
#define CTRL_REG        0x1081 //0b1000010000001

//Control Register Bit Enabled
#define BUFF_REG_EN     1 << 0
#define DESCALER_EN     1 << 1
#define RELU_EN         1 << 2
#define STORE_MAX_EN    1 << 3
#define MAX_POOL_EN     1 << 4
#define AVG_POOL_EN     1 << 5
#define EQZ_MAX_REG_EN  1 << 6
#define RST_POOL_REG_EN 1 << 7
#define MAC_REG_LD_EN   1 << 8

// Maxwell Architecture Specifications
#define SB_NR           2
#define SB_SIZE         2048 // 2KB SRAM BANK
#define ILM_OFFSET      1024 //ILM Memory starts after 1024 addresses from Maxwell Start Address

unsigned int maxwell_load_activations(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem);

#endif