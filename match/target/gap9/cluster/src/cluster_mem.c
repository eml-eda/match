#include <cluster_mem.h>
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

static unsigned int memalloc_O(){
    db_O++;
    return l1_memory+l1_O_off[db_O%2];
}
static unsigned int memalloc_I(){
    db_I++;
    return l1_memory+l1_I_off[db_I%2];
}
static unsigned int memalloc_X(){
    db_X++;
    return l1_memory+l1_X_off[db_X%2];
}
static unsigned int memalloc_Y(){
    db_Y++;
    return l1_memory+l1_Y_off[db_Y%2];
}
static unsigned int memalloc_W(){
    db_W++;
    return l1_memory+l1_W_off[db_W%2];
}

void cluster_init_platform(void (inner_function)(unsigned int* args_inner_function),unsigned int* args,common_kernel* common_kernel){
    #ifdef PROFILE_LAYERS
    stop_g_perf_counter();
    start_g_perf_counter();
    #endif

    pi_cluster_task(&cluster_task,inner_function,args);
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

    #ifdef PROFILE_LAYERS
    int32_t cycles=stop_g_perf_counter();
    printf(",%d",cycles);
    start_g_perf_counter();
    #endif
}

void cluster_init_l1_memory(){
    l1_memory=pi_cl_l1_malloc(NULL, 90*1024);
}

void cluster_startup_memory(common_kernel* common_kernel,int* first_op_sizes,unsigned char first_op_db,dimension_I* dim_I,
                                int* second_op_sizes,unsigned char second_op_db,dimension_W* dim_W,
                                int* third_op_sizes,unsigned char third_op_db,dimension_O* dim_O,
                                int* paddings,int* strides){
    if(common_kernel->specific_pattern!=elemwise_add){
        l1_I_off[0]=0;l1_I_off[1]=first_op_sizes[1]*first_op_db*common_kernel->prec_I/8;
        l1_W_off[0]=(1+first_op_db)*first_op_sizes[1]*common_kernel->prec_I/8;l1_W_off[1]=l1_W_off[0]+second_op_sizes[1]*second_op_db*common_kernel->prec_I/8;
        l1_bias_off=(1+first_op_db)*first_op_sizes[1]*common_kernel->prec_I/8+(1+second_op_db)*second_op_sizes[1]*common_kernel->prec_W/8+(1+third_op_db)*third_op_sizes[1]*common_kernel->prec_O/8;
    }
    else{
        l1_X_off[0]=0;l1_X_off[1]=first_op_sizes[1]*first_op_db;
        l1_Y_off[0]=(1+first_op_db)*first_op_sizes[1];l1_Y_off[1]=l1_Y_off[0]+second_op_sizes[1]*second_op_db;
        l1_bias_off=0x0;
    }
    l1_O_off[0]=(1+first_op_db)*first_op_sizes[1]*common_kernel->prec_I/8+(1+second_op_db)*second_op_sizes[1]*common_kernel->prec_W/8;l1_O_off[1]=l1_O_off[0]+third_op_sizes[1]*third_op_db*common_kernel->prec_O/8;
    cluster_init_l1_memory();
    l1_im2col_off=l1_O_off[0]+third_op_sizes[1]*(third_op_db+1)*common_kernel->prec_O/8;
    if(common_kernel->specific_pattern!=elemwise_add)    l1_im2col_off+=dim_O->size_K[l2_mem]*4;
    if(common_kernel->pattern_name==dense_bnorm_requant || common_kernel->pattern_name==conv2d_bnorm_requant)   l1_im2col_off+=dim_O->size_K[l2_mem]*4;
    int im2coldim=1*(dim_W->size_FX[l2_mem]*dim_W->size_FY[l2_mem]*(dim_I->size_IY[l1_mem]+paddings[0]+paddings[2])+dim_W->size_FX[l2_mem]*dim_W->size_FY[l2_mem]);
    l1_pwt_off=l1_im2col_off+im2coldim;
    //printf("L1 memory at %d offsets: I {%d,%d} W {%d,%d} O {%d,%d} bias %d im2col %d\n",l1_memory,l1_I_off[0],l1_I_off[1],l1_W_off[0],l1_W_off[1],
    //l1_O_off[0],l1_O_off[1],l1_bias_off,l1_im2col_off);
    pi_team_config_offload(NUM_CORES);
    transfer = dma_transfer_create();
}

void cluster_shutdown_mem(common_kernel* common_kernel){
    dma_transfer_free(transfer);
    pi_cl_l1_free(NULL, l1_memory, 90*1024);
}


unsigned int cluster_mem_transfer_O(common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    //printf("Mem transfer O: K %d OY %d OX %d from %d to %d int mem idx %d\n",dim->size_K[int_mem],dim->size_OY[int_mem],
    //dim->size_OX[int_mem],ext_pt,0,int_mem);
    return memalloc_O();
}

void copy_out_computation_(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    //printf("Copy out int %d\n",int_pt-l1_memory);
    //printf("Dim O K [int %d,ext %d] OY [int %d,ext %d] OX [int %d,ext %d]\n",dim->size_K[int_mem],dim->size_K[ext_mem],
    //dim->size_OY[int_mem],dim->size_OY[ext_mem],dim->size_OX[int_mem],dim->size_OX[ext_mem]);
    dma_transfer_async((DmaTransferConf) {
            .ext = ext_pt,
            .loc = int_pt,
            .number_of_2d_copies = dim->size_OY[int_mem],
            .number_of_1d_copies = dim->size_OX[int_mem],
            .length_1d_copy = dim->size_K[int_mem]*common_kernel->prec_O/8,
            .hwc_to_chw = 0,
            .stride_2d = dim->size_K[ext_mem]*dim->size_OX[ext_mem]*common_kernel->prec_O/8,
            .stride_1d = dim->size_K[ext_mem]*common_kernel->prec_O/8,
            .dir = 0
    });
    return;
}

void cluster_copy_out_curr_computation(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    if(common_kernel->specific_pattern==elemwise_add)    copy_out_computation_(common_kernel,dim,int_pt,ext_pt,int_mem,ext_mem);
    return;
}

void cluster_copy_out_prev_computation(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    if(common_kernel->specific_pattern!=elemwise_add)    copy_out_computation_(common_kernel,dim,int_pt,ext_pt,int_mem,ext_mem);
    return;
}

unsigned int cluster_mem_transfer_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    unsigned int dst=memalloc_I();
    //printf("Mem transfer I: C %d IY %d IX %d from %d to %d int mem idx %d\n",dim->size_C[int_mem],dim->size_IY[int_mem],
    //dim->size_IX[int_mem],ext_pt,dst-l1_memory,int_mem);
    unsigned int src=ext_pt;
    dma_transfer_async((DmaTransferConf) {
        .ext = src,
        .loc = dst,
        .number_of_2d_copies = (dim->size_IY[int_mem]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y),
        .number_of_1d_copies = (dim->size_IX[int_mem]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y),
        .length_1d_copy = dim->size_C[int_mem],
        .hwc_to_chw = (common_kernel->specific_pattern==depthwise_conv2d || common_kernel->specific_pattern==depthwise_conv2d_less_4),
        .stride_2d = dim->size_C[ext_mem]*dim->size_IX[ext_mem],
        .stride_1d = dim->size_C[ext_mem],
        .dir = 1
    });
    return dst;
}

unsigned int cluster_mem_transfer_X(common_kernel* common_kernel,dimension_X* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    unsigned int dst=memalloc_X();
    unsigned int src=ext_pt;
    dma_transfer_async((DmaTransferConf) {
        .ext = src,
        .loc = dst,
        .number_of_2d_copies = dim->size_IY[int_mem],
        .number_of_1d_copies = dim->size_IX[int_mem],
        .length_1d_copy = dim->size_C[int_mem],
        .hwc_to_chw = 0,
        .stride_2d = dim->size_C[ext_mem]*dim->size_IX[ext_mem],
        .stride_1d = dim->size_C[ext_mem],
        .dir = 1
    });
    return dst;
}

unsigned int cluster_mem_transfer_Y(common_kernel* common_kernel,dimension_Y* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    unsigned int dst=memalloc_Y();
    unsigned int src=ext_pt;
    dma_transfer_async((DmaTransferConf) {
        .ext = src,
        .loc = dst,
        .number_of_2d_copies = dim->size_IY[int_mem],
        .number_of_1d_copies = dim->size_IX[int_mem],
        .length_1d_copy = dim->size_C[int_mem],
        .hwc_to_chw = 0,
        .stride_2d = dim->size_C[ext_mem]*dim->size_IX[ext_mem],
        .stride_1d = dim->size_C[ext_mem],
        .dir = 1
    });
    return dst;
}

unsigned int cluster_mem_transfer_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    unsigned int dst=memalloc_W();
    //printf("Mem transfer W: K %d C %d FY %d FX %d from %d to %d int mem idx %d\n",dim->size_K[int_mem],dim->size_C[int_mem],
    //dim->size_FY[int_mem],dim->size_FX[int_mem],ext_pt,dst-l1_memory,int_mem);
    if(!(common_kernel->specific_pattern==depthwise_conv2d || common_kernel->specific_pattern==depthwise_conv2d_less_4))
        dma_transfer_async((DmaTransferConf) {
            .ext = ext_pt,
            .loc = dst,
            .number_of_2d_copies = dim->size_K[int_mem],
            .number_of_1d_copies = dim->size_FY[int_mem]*dim->size_FX[int_mem],
            .length_1d_copy = dim->size_C[int_mem],
            .hwc_to_chw = 0,
            .stride_2d = dim->size_C[ext_mem]*dim->size_FY[ext_mem]*dim->size_FX[ext_mem],
            .stride_1d = dim->size_C[ext_mem],
            .dir = 1
        });
    else
        dma_transfer_async((DmaTransferConf) {
            .ext = ext_pt,
            .loc = dst,
            .number_of_2d_copies = 1,
            .number_of_1d_copies = 1,
            .length_1d_copy = dim->size_K[int_mem]*dim->size_FY[int_mem]*dim->size_FX[int_mem],
            .hwc_to_chw = 0,
            .stride_2d = dim->size_FY[ext_mem]*dim->size_FX[ext_mem],
            .stride_1d = 1,
            .dir = 1
        });
    return dst;
}

void cluster_wait_any_transfer(common_kernel* common_kernel){
    dma_transfer_wait(transfer);
    transfer = dma_transfer_create();
}

void wait_any_computation(){
    pi_team_offload_wait();
}

void cluster_wait_prev_computation(common_kernel* common_kernel){
    if(common_kernel->specific_pattern!=elemwise_add) return wait_any_computation();
}

void cluster_wait_curr_computation(common_kernel* common_kernel){
    if(common_kernel->specific_pattern==elemwise_add) return wait_any_computation();
}

void cluster_pattern_constant_loading(cluster_kernel* kernel,unsigned int iter,tile_indexes_W* abs_tile_idx,
                                    tile_indexes_W* relative_tile_idx,void* weights_and_constant_buf){
    if(kernel->common_kernel->specific_pattern!=elemwise_add){
        int batchnorm_act=kernel->common_kernel->pattern_name==dense_bnorm_requant || kernel->common_kernel->pattern_name==conv2d_bnorm_requant;
        if(iter==0){
            unsigned int size_weights=kernel->common_kernel->dim_W->size_K[l2_mem]*kernel->common_kernel->dim_W->size_C[l2_mem]*
            kernel->common_kernel->dim_W->size_FY[l2_mem]*kernel->common_kernel->dim_W->size_FX[l2_mem];
            dma_transfer_1d_async((DmaTransferConf) {
                .ext = ((unsigned int) weights_and_constant_buf)+size_weights,
                .loc = l1_bias_off+l1_memory,
                .length_1d_copy = 4*kernel->common_kernel->dim_W->size_K[l2_mem]*(batchnorm_act+1),
                .dir = 1
            });
            if(!batchnorm_act)
                kernel->common_kernel->bias_pt=l1_bias_off+l1_memory;
            else{
                kernel->common_kernel->batchnorm_mul=l1_bias_off+l1_memory;
                kernel->common_kernel->batchnorm_add=l1_bias_off+l1_memory+4*kernel->common_kernel->dim_W->size_K[l2_mem];
            }
        }
        else{
            if(!batchnorm_act)
                kernel->common_kernel->bias_pt=l1_bias_off+l1_memory+4*abs_tile_idx->tile_K;
            else{
                kernel->common_kernel->batchnorm_mul=l1_bias_off+l1_memory+4*abs_tile_idx->tile_K;
                kernel->common_kernel->batchnorm_add=l1_bias_off+l1_memory+4*kernel->common_kernel->dim_W->size_K[l2_mem]+4*abs_tile_idx->tile_K;
            }
        }
    }
}

unsigned int get_im2col_pt(){
    return l1_memory+l1_im2col_off;
}

unsigned int get_pwtbuf(){
    return l1_memory+l1_pwt_off;
}

unsigned int cluster_get_l1_memory_addr(){
    return l1_memory;
}