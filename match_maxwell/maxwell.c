#include <maxwell.h>

//Register Addresses
#define AVG_KERNEL_REG  0x1088  //0b1000010001000
#define MAX_REG         0x1084 //0b1000010000100
#define BIAS_REG        0x1082 //0b1000010000010
#define CTRL_REG        0x1081 //0b1000010000001

//Control Register Bit Enabled
#define BUFF_REG_EN     (1 << 0)
#define DESCALER_EN     (1 << 1)
#define RELU_EN         (1 << 2)
#define STORE_MAX_EN    (1 << 3)
#define MAX_POOL_EN     (1 << 4)
#define AVG_POOL_EN     (1 << 5)
#define EQZ_MAX_REG_EN  (1 << 6)
#define RST_POOL_REG_EN (1 << 7)
#define MAC_REG_LD_EN   (1 << 8)

//Defined on Heepnosis
#define MAXWELL_START_ADDRESS  0x0 

// // Maxwell Architecture Specifications
#define SB_NR           2
#define SB_SIZE         2048 // 2KB SRAM BANK
#define ILM_OFFSET      1024 //ILM Memory starts after 1024 addresses from Maxwell Start Address

//initialize Maxwell
//Set necessary parameters that describe Maxwell
void init_Maxwell(uint start_address){
    volatile uint32_t* maxwell = (volatile uint32_t *)start_address;
}

//initialize Kernel
//Set defining paramaeters that describe the application (Conv2D)
void define_application(uint IH, uint IW, uint IC, uint P, uint S, uint K, uint OC){
    // here for example, should I define these things as MACROs already, or is there some kind of structure that can hold this information
}

//define tiling parameters (variables needed for the loops)
void define_tiles(uint HOR_DIV, uint VER_DIV){
    uint NR_TILES        (HOR_DIV*VER_DIV)   //total number of tiles
    uint OTW             (OW/HOR_DIV)    //width of output tile
    uint OTH             (OH/VER_DIV)    //height of output tile
    uint ITW             ((OTW-1)*S + K - 2*P)    //width of input tile
    uint ITH             ((OTH-1)*S + K - 2*P)    //height of input tile
    uint INPUT_TILE_SIZE       (ITW*ITH) //size of input tile        
    uint OUTPUT_TILE_SIZE       (OTW*OTH) //size of input tile   
}

//store activation input in ILM
void store_input_ILM(uint start_address, uint32_t input, uint ILM_OFFSET){
    for(uint i=0; i<size_of(input); i++){
        start_address[ILM_OFFSET+i] = input[i];
    }
}

//enable different registers
void enable_regs(uint reg_addr, uint reg_value){
    maxwell[reg_addr] = reg_value;  //maxwell is the pointer defined in init Maxwell, how should I receive it
}

//map values from ILM to CM accordingly 
void map_ILM2CM(uint NR_TILES, uint HOR_DIV, uint OTH, uint OTW, uint IW, uint ITW, uint INPUT_TILE_SIZE, uint ILM_OFFSET){
    for(int i=0; i<NR_TILES; i++){  //loop through input tiles
        int tile_x = i%HOR_DIV; //x coordinate of the tile
        int tile_y = i/HOR_DIV; //y coordinate of the tile
        int tile_start = tile_y*(OTH*IW) + tile_x*OTW;   //first tile element
        for(int j=0; j<INPUT_TILE_SIZE; j++){    // loop through elements of tile
            int tile_row = j%ITW;   //row of element
            int tile_col = j/ITW;   //column of element
            uint32_t tile_element_idx = tile_start + (tile_col*IW) + tile_row;
            uint32_t buffer = maxwell[ILM_OFFSET + tile_element_idx]; //Do RD in ILM        //MAXWELL FROM INIT
            uint32_t address = ((1 << 12) + (1 << 11) + (i << 10) + j);    //FORMAT  NM - 1 - SA - CM Addres: NM=1, 1=Load CM cmd, SA=TILE_NR refers to SRAM BANK nr, CM Address - address to correct word in CM. All shifted by 2 for Heepnosis
            maxwell[address] = buffer * i; //Do WR in CM    
        }
    }
}


//do calculation and store from CM to ILM accordingly
void calc_map_CM2ILM(uint OUTPUT_TILE_SIZE, uint OTW, uint K, uint ITW, uint SB_NR, uint HOR_DIV, uint ILM_OFFSET, uint IH, uint IW, uint IC, uint OW, uint32_t* weights){
    for(int i=0; i<OUTPUT_TILE_SIZE; i++){  //loop through all elements of an output tile
        int tile_row = i%OTW;   //row of element
        int tile_col = i/OTW;   //column of element
        int tile_start = ITW*tile_col + tile_row;
        for(int j=0; j<(K*K); j++){    // loop through all elements of kernel
            int kernel_x = j%K;   //x coordinate of kernel element
            int kernel_y = j/K;   //y coordinate of kernel element
            uint32_t CM_addr_idx = tile_start + (ITW)*kernel_y + kernel_x;
            // Computations
            /**********@TODO****************/
            uint32_t address = ((1 << 12) + (1 << 10) + CM_addr_idx);    //FORMAT  NM - Cmd - CM address: NM=1, Cmd = 01, CM Address depending on tile and element; Correct Address of CM that needs to be loaded. All shifted by 2 for Heepnosis
            uint32_t imo_reg =maxwell[address];     //Load CM element in IMO Reg
            mac_operation(weights[j], 8, (uint32_t*)maxwell);
        }
        for(int k=0; k<SB_NR; k++){
            int k_x = k%HOR_DIV;
            int k_y = k/HOR_DIV;
            uint32_t address = ((1 << 12) + (1 << 8) + k);    //FORMAT  NM - Cmd - 7unused bits - SA: NM=1, Cmd = 0001, Unused = 0000000, SA = tile number; Read output register by doing a read in the correct subarray. All shifted by 2 for Heepnosis
            uint32_t buffer = maxwell[address]; //Do RD in CM
            maxwell[ILM_OFFSET + (IH*IW*IC)+OW*tile_col + tile_row + k_x*OTW + k_y*OW*OTH] = buffer; //@TODO CHECK THE ADDRESS NOT SURE  
        }
    }
}
