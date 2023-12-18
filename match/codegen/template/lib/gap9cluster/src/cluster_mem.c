#include <match_target_params.h>
// flags and pattern info
patterns_cluster current_pattern;
bool flag_dw=false;
// memory pts
static unsigned int l1_memory=0x0;
static unsigned int l1_O_off[2]={0,0};
static unsigned int l1_I_off[2]={0,0};
static unsigned int l1_X_off[2]={0,0};
static unsigned int l1_Y_off[2]={0,0};
static unsigned int l1_W_off[2]={0,0};
static unsigned int db_O=0;
static unsigned int db_I=0;
static unsigned int db_X=0;
static unsigned int db_Y=0;
static unsigned int db_W=0;
unsigned int l1_bias_off=0x0;
unsigned int l1_im2col_off=0x0;
unsigned int l1_pwt_off=0x0;
DmaTransfer transfer;

unsigned int memalloc_O(int size,int memorylevel,int operator){
    db_O++;
    return l1_memory+l1_O_off[db_O%2];
}
unsigned int memalloc_I(){
    db_I++;
    return l1_memory+l1_I_off[db_I%2];
}
unsigned int memalloc_X(){
    db_X++;
    return l1_memory+l1_X_off[db_X%2];
}
unsigned int memalloc_Y(){
    db_Y++;
    return l1_memory+l1_Y_off[db_Y%2];
}
unsigned int memalloc_W(){
    db_W++;
    return l1_memory+l1_W_off[db_W%2];
}

void init_platform(void (*inner_function)(unsigned int* args_inner_function),unsigned int* args){
    pi_cluster_task(&cluster_task,inner_function,args);
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    return 0;
}

void startup_memory_and_pattern(int* first_op_sizes,bool first_op_db,dimensionI* dim_I,
                                int* second_op_sizes,bool second_op_db,dimensionW* dim_W,
                                int* third_op_sizes,bool third_op_db,dimensionO* dim_O,
                                int* paddings,int* strides,patterns_cluster pattern){
    current_pattern=pattern;
    if(current_pattern==gap9cluster_conv2d && dim_W->size_C[l2_mem]==1 && dim_W->size_K[l2_mem]!=1) dw_flag=true;
    if(current_pattern!=gap9cluster_add){
        l1_I_off[0]=0;l1_I_off[1]=first_op_sizes[1]*first_op_db;
        l1_W_off[0]=(1+first_op_db)*first_op_sizes[1];l1_W_off[1]=l1_W_off[0]+second_op_sizes[1]*second_op_db;
        l1_bias_off=(1+first_op_db)*first_op_sizes[1]+(1+second_op_db)*second_op_sizes[1]+(1+third_op_db)*third_op_sizes[1];
    }
    else{
        l1_X_off[0]=0;l1_X_off[1]=first_op_sizes[1]*first_op_db;
        l1_Y_off[0]=(1+first_op_db)*first_op_sizes[1];l1_Y_off[1]=l1_Y_off[0]+second_op_sizes[1]*second_op_db;
        l1_bias_off=0x0;
    }
    l1_O_off[0]=(1+first_op_db)*first_op_sizes[1]+(1+second_op_db)*second_op_sizes[1];l1_O_off[1]=l1_O_off[0]+third_op_sizes[1]*third_op_db;
    l1_memory=pi_cl_l1_malloc(NULL, 12*1024*8);
    l1_im2col_off=l1_O_off[0]+third_op_sizes[1]*(third_op_db+1);
    if(current_pattern!=gap9cluster_add)    l1_im2col_off+=dim_O->size_K[l2_mem]*4;
    int im2coldim=NUM_CORES*(dim_W->size_FX[l2_mem]*dim_W->size_FY[l2_mem]*(dim_I->size_IY[l1_mem]+paddings[0]+paddings[2])+dim_W->size_FX[l2_mem]*dim_W->size_FY[l2_mem]);
    l1_pwt_off=l1_im2col_off+im2coldim;
    pi_team_config_offload(NUM_CORES);
    transfer = dma_transfer_create();
}

void shutdown_mem(){
    pi_cl_l1_free(NULL, l1_memory, 12*1024*8);
}


unsigned int mem_transfer_O(dimensionO* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    return memalloc_O();
}

void copy_out_computation_(dimensionO* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    dma_transfer_async((DmaTransferConf) {
            .ext = ext_pt,
            .loc = int_pt,
            .number_of_2d_copies = dim->size_OY[src_memory_level],
            .number_of_1d_copies = dim->size_OX[src_memory_level],
            .length_1d_copy = dim->size_K[src_memory_level],
            .hwc_to_chw = 0,
            .stride_2d = dim->size_K[dst_memory_level]*dim->size_OX[dst_memory_level],
            .stride_1d = dim->size_K[dst_memory_level],
            .dir = 0
    });
    return;
}

void copy_out_curr_computation(dimensionO* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    if(current_pattern==gap9cluster_add)    copy_out_computation_(dim,int_pt,ext_pt,int_mem,ext_mem);
    return;
}

void copy_out_prev_computation(dimensionO* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    if(current_pattern!=gap9cluster_add)    copy_out_computation_(dim,int_pt,ext_pt,int_mem,ext_mem);
    return;
}

unsigned int mem_transfer_I(dimensionI* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    unsigned int dst=memalloc_I();
    unsigned int src=ext_pt;
    dma_transfer_async((DmaTransferConf) {
        .ext = src,
        .loc = dst,
        .number_of_2d_copies = (dim->size_IY[dst_memory_level]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y),
        .number_of_1d_copies = (dim->size_IX[dst_memory_level]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y),
        .length_1d_copy = dim->size_C[dst_memory_level],
        .hwc_to_chw = dw_flag,
        .stride_2d = dim->size_C[src_memory_level]*dim->size_IX[src_memory_level],
        .stride_1d = dim->size_C[src_memory_level],
        .dir = 1
    });
    return dst;
}

unsigned int mem_transfer_X(dimensionX* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    unsigned int dst=memalloc_X();
    unsigned int src=ext_pt;
    dma_transfer_async((DmaTransferConf) {
        .ext = src,
        .loc = dst,
        .number_of_2d_copies = (dim->size_IY[dst_memory_level]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y),
        .number_of_1d_copies = (dim->size_IX[dst_memory_level]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y),
        .length_1d_copy = dim->size_C[dst_memory_level],
        .hwc_to_chw = 0,
        .stride_2d = dim->size_C[src_memory_level]*dim->size_IX[src_memory_level],
        .stride_1d = dim->size_C[src_memory_level],
        .dir = 1
    });
    return dst;
}

unsigned int mem_transfer_Y(dimensionY* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    unsigned int dst=memalloc_Y();
    unsigned int src=ext_pt;
    dma_transfer_async((DmaTransferConf) {
        .ext = src,
        .loc = dst,
        .number_of_2d_copies = (dim->size_IY[dst_memory_level]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y),
        .number_of_1d_copies = (dim->size_IX[dst_memory_level]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y),
        .length_1d_copy = dim->size_C[dst_memory_level],
        .hwc_to_chw = 0,
        .stride_2d = dim->size_C[src_memory_level]*dim->size_IX[src_memory_level],
        .stride_1d = dim->size_C[src_memory_level],
        .dir = 1
    });
    return dst;
}

unsigned int mem_transfer_W(dimensionW* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    unsigned int dst=memalloc_W();
    if(!flag_dw)
        dma_transfer_async((DmaTransferConf) {
            .ext = ext_pt,
            .loc = dst,
            .number_of_2d_copies = dim->size_K[dst_memory_level],
            .number_of_1d_copies = dim->size_FY[dst_memory_level]*dim->size_FX[dst_memory_level],
            .length_1d_copy = dim->size_C[dst_memory_level],
            .hwc_to_chw = 0,
            .stride_2d = dim->size_C[src_memory_level]*dim->size_FY[src_memory_level]*dim->size_FX[src_memory_level],
            .stride_1d = dim->size_C[src_memory_level],
            .dir = 1
        });
    else
        dma_transfer_async((DmaTransferConf) {
            .ext = ext_pt,
            .loc = dst,
            .number_of_2d_copies = 1,
            .number_of_1d_copies = 1,
            .length_1d_copy = dim->size_K[dst_memory_level]*dim->size_FY[dst_memory_level]*dim->size_FX[dst_memory_level],
            .hwc_to_chw = 0,
            .stride_2d = dim->size_FY[src_memory_level]*dim->size_FX[src_memory_level],
            .stride_1d = 1,
            .dir = 1
        });
    return dst;
}

void wait_any_transfer(){
    dma_transfer_wait(transfer);
}

void wait_any_computation(){
    pi_team_offload_wait();
}

void wait_prev_computation(){
    if(current_pattern!=gap9cluster_add) return wait_any_computation();
}

void wait_curr_computation(){
    if(current_pattern==gap9cluster_add) return wait_any_computation();
}

unsigned int pointer_offset_O(tile_indexes_O* tile_idxs,int mem_level){
    return 0;
}

unsigned int pointer_offset_I(tile_indexes_I* tile_idxs,int mem_level){
    return 0;
}

unsigned int pointer_offset_X(tile_indexes_X* tile_idxs,int mem_level){
    return 0;
}

unsigned int pointer_offset_Y(tile_indexes_Y* tile_idxs,int mem_level){
    return 0;
}

unsigned int pointer_offset_W(tile_indexes_W* tile_idxs,int mem_level){
    return 0;
}