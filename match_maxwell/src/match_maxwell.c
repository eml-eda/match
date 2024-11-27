#include <match_maxwell.h>

volatile uint32_t* maxwell = (volatile uint32_t *)MAXWELL_START_ADDRESS;


/*
OTH -> Output Tile Height
OTW -> Output Tile Width
*/
unsigned int maxwell_load_activations(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    // NR OF TILES FOR INP CHANNEL
    int SIZE_TILE_INP_CHANNEL = dim->size_C[COMPUTE_MEMORY];
    int INP_CH_NR_TILES = dim->size_C[INTER_LAYER_MEMORY]/dim->size_C[COMPUTE_MEMORY];
    dim->size_IX[COMPUTE_MEMORY] --> 16 dim->size_IX[INTER_LAYER_MEMORY] --> 32 
    16/MAXWELL_NUM_BANKS 
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
    return ext_pt;
}

void maxwell_compute_tile_and_store_output(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    common_kernel* kernel = kernel->common_kernel;
    int o_tile_size = kernel->ox * kernel->oy;
    int HOR_DIV ;
    int OW ;
    for(int i=0; i<o_tile_size; i++){  //loop through all elements of an output tile
        int tile_row = i%kernel->ox;   //row of element
        int tile_col = i/kernel->ox;   //column of element
        int tile_start = kernel->ix*tile_col + tile_row;
        for(int j=0; j<(kernel->fx*kernel->fy); j++){    // loop through all elements of kernel
            int kernel_x = j%kernel->fx;   //x coordinate of kernel element
            int kernel_y = j/kernel->fx;   //y coordinate of kernel element
            uint32_t CM_addr_idx = tile_start + (kernel->ix)*kernel_y + kernel_x;
            // Computations
            /**********@TODO****************/
            uint32_t address = ((1 << 12) + (1 << 10) + CM_addr_idx);    //FORMAT  NM - Cmd - CM address: NM=1, Cmd = 01, CM Address depending on tile and element; Correct Address of CM that needs to be loaded. All shifted by 2 for Heepnosis
            uint32_t imo_reg =maxwell[address];     //Load CM element in IMO Reg
            mac_operation(weights[j], 8, (uint32_t*)maxwell);
        }
        for(int k=0; k<MAXWELL_NUM_BANKS; k++){
            int k_x = k%HOR_DIV;
            int k_y = k/HOR_DIV;
            uint32_t address = ((1 << 12) + (1 << 8) + k);    //FORMAT  NM - Cmd - 7unused bits - SA: NM=1, Cmd = 0001, Unused = 0000000, SA = tile number; Read output register by doing a read in the correct subarray. All shifted by 2 for Heepnosis
            uint32_t buffer = maxwell[address]; //Do RD in CM
            maxwell[ILM_OFFSET + (kernel->iy*kernel->ix*kernel->c_i)+kernel->ix*tile_col + tile_row + k_x*OTW + k_y*OW*OTH] = buffer; //@TODO CHECK THE ADDRESS NOT SURE  
        }
    }
}
