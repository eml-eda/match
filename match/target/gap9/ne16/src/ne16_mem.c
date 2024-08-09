#include <ne16_mem.h>
#include <gap9_cluster.h>
// memory pts
static unsigned int l1_O_off[DB_BUFFER_SIZE]={0,0};
static unsigned int l1_I_off[DB_BUFFER_SIZE]={0,0};
static unsigned int l1_W_off[DB_BUFFER_SIZE]={0,0};

static unsigned int lock_loader_task=0;
static unsigned int l1_bias_off=0x0;
static DmaTransfer input_transfers,output_transfers,transfers;
static int input_dma_active=0,output_dma_active=0;
static void* ne16_callback;
static common_kernel* ne16_common_kernel;

static unsigned int memalloc_O(unsigned int task_id){
    return cluster_get_l1_memory_addr()+l1_O_off[get_nnx_db_O(task_id)];
}
static unsigned int memalloc_I(unsigned int task_id){
    return cluster_get_l1_memory_addr()+l1_I_off[get_nnx_db_I(task_id)];
}
static unsigned int memalloc_W(unsigned int task_id){
    return cluster_get_l1_memory_addr()+l1_W_off[get_nnx_db_W(task_id)];
}

void ne16_set_task_id(common_kernel* kernel){
    #ifdef MATCH_NE16_BUFFERED
    kernel->task_id=pi_core_id();
    #else
    kernel->task_id=SINGLE_CORE_TASK;
    #endif
}

void ne16_init_platform_(
    void* args){
    unsigned int *real_args = (unsigned int *) args;
    unsigned int input_I_pt = (unsigned int) real_args[0];
    unsigned int output_pt = (unsigned int) real_args[1];
    unsigned int args_ne16[2];
    args_ne16[0] = (unsigned int) input_I_pt;
    args_ne16[1] = (unsigned int) output_pt;
    #ifdef MATCH_NE16_BUFFERED
    dma_mutex_init();
    #endif
    match_ne16_set_nnx_dev();
    ne16_nnx_init(match_ne16_get_nnx_dev(),(ne16_pulp_conf_t){
        .max_stall = 8
    });
    #ifdef MATCH_NE16_BUFFERED
    int err = 0;

    if (err = monitor_init(&get_nnx_monitor()->input, DB_BUFFER_SIZE)) {
        //printf("Input monitor initialization failed with status %d.\n", err);
        return;
    }
    if (err = monitor_init(&get_nnx_monitor()->output, DB_BUFFER_SIZE)) {
        //printf("Output monitor initialization failed with status %d.\n", err);
        monitor_term(get_nnx_monitor()->input);
        return;
    }
    #endif
    cluster_init_l1_memory();
    hwpe_soft_clear(&(match_ne16_get_nnx_dev()->hwpe_dev));
    #ifdef DEBUG_GVSOC
    nnx_activate_gvsoc_logging(GVSOC_LOG_LEVEL_CONFIG, GVSOC_LOGGING_FORMAT_DECIMAL);
    #endif
    // Init nnx tasks
    for (int i = 0; i < DB_BUFFER_SIZE; i++) {
        ne16_task_init(match_ne16_get_nnx_task(i));
        ne16_task_set_op_to_conv(match_ne16_get_nnx_task(i),
                ne16_common_kernel->dim_W->size_FY[l2_mem],
                ne16_common_kernel->specific_pattern==depthwise_conv2d,
                ne16_common_kernel->stride_y);
        ne16_task_set_bits(match_ne16_get_nnx_task(i),
                ne16_common_kernel->prec_I,ne16_common_kernel->prec_O,
                ne16_common_kernel->prec_W);
        ne16_task_set_norm_quant(match_ne16_get_nnx_task(i),
                (ne16_quant_t) {
                    .shift_amount = ne16_common_kernel->right_shift,
                    .function = ne16_common_kernel->activation_function?quantFunctionRelu:quantFunctionIdentity,
                    .flag_rounding = ne16TaskFlagFalse
                }, (ne16_norm_t) {
                    .mode  = normMode32Bit,
                    .flag_bias  = ne16TaskFlagTrue,
                    .flag_shift = ne16TaskFlagFalse
                });
        ne16_task_set_weight_offset(match_ne16_get_nnx_task(i), weightOffsetModeLayerWise, -128);
    }

    // Fork
    #ifdef MATCH_NE16_BUFFERED
    //pi_cl_team_barrier();
    pi_cl_team_fork(NE16_TASKS, (void *)ne16_callback, (void*)args_ne16);
    //pi_cl_team_barrier();
    #else
    ((void (*)(unsigned int *))ne16_callback)((void*)args_ne16);
    #endif
    // Terminate
    cluster_free_mem();
    ne16_nnx_term(match_ne16_get_nnx_dev());
    #ifdef MATCH_NE16_BUFFERED
    monitor_term(get_nnx_monitor()->input);
    monitor_term(get_nnx_monitor()->output);
    #endif
    return;
}

void __attribute__ ((noinline)) ne16_init_platform(void (inner_function)(unsigned int* args_inner_function),unsigned int* args,common_kernel* common_kernel){
    #ifdef MATCH_GAP9_PROFILE_LAYERS
    stop_g_perf_counter();
    start_g_perf_counter();
    #endif

    ne16_common_kernel=common_kernel;
    ne16_callback=inner_function;
    //void* cl_ne16_args[4];
    //cl_ne16_args[0]=args[0];cl_ne16_args[1]=args[1];cl_ne16_args[2]=inner_function;cl_ne16_args[3]=common_kernel;
    
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cluster_task,ne16_init_platform_,args));
    
    #ifdef MATCH_GAP9_PROFILE_LAYERS
    int32_t cycles=stop_g_perf_counter();
    printf(",%d",cycles);
    start_g_perf_counter();
    #endif
}
void ne16_startup_memory(common_kernel* common_kernel,int* first_op_sizes,unsigned char first_op_db,dimension_I* dim_I,
                                int* second_op_sizes,unsigned char second_op_db,dimension_W* dim_W,
                                int* third_op_sizes,unsigned char third_op_db,dimension_O* dim_O,
                                int* paddings,int* strides){
    l1_I_off[0]=0;l1_I_off[1]=first_op_sizes[1]*first_op_db;
    l1_W_off[0]=(1+first_op_db)*first_op_sizes[1];l1_W_off[1]=l1_W_off[0]+second_op_sizes[1]*second_op_db;
    l1_bias_off=(1+first_op_db)*first_op_sizes[1]+(1+second_op_db)*second_op_sizes[1]+(1+third_op_db)*third_op_sizes[1];

    l1_O_off[0]=(1+first_op_db)*first_op_sizes[1]+(1+second_op_db)*second_op_sizes[1];l1_O_off[1]=l1_O_off[0]+third_op_sizes[1]*third_op_db;
    
    #ifdef MATCH_LOG_GAP9_VERBOSE
    printf("L1 off I [ %d %d ] W [ %d %d ] O [ %d %d ] Bias %d\n",l1_I_off[0],l1_I_off[1],l1_W_off[0],l1_W_off[1],l1_O_off[0],l1_O_off[1],l1_bias_off);
    #endif
    
    #ifndef MATCH_NE16_BUFFERED
    transfers = dma_transfer_create();
    #endif
}

void ne16_shutdown_mem(common_kernel* common_kernel){
    return;
}

unsigned int ne16_mem_transfer_O(common_kernel* common_kernel,dimension_O* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    
    #ifdef MATCH_LOG_GAP9_VERBOSE
    printf("Mem transfer O: K %d OY %d OX %d from %d to %d int mem idx %d\n",dim->size_K[int_mem],dim->size_OY[int_mem],
    dim->size_OX[int_mem],ext_pt,0,int_mem);
    #endif

    inc_nnx_db_O(common_kernel->task_id);
    return memalloc_O(common_kernel->task_id);
}

void ne16_copy_out_curr_computation(common_kernel* common_kernel,dimension_O* dim,unsigned int int_pt,unsigned int ext_pt,
                                    int int_mem,int ext_mem){
    if(common_kernel->task_id==STORER_TASK || common_kernel->task_id==SINGLE_CORE_TASK){
        #ifdef MATCH_NE16_BUFFERED
        dma_mutex_lock();
        output_transfers=dma_transfer_create();
        #endif
        // no tiling so do 1D transfer
        if(dim->size_OY[ext_mem]==dim->size_OY[int_mem] && dim->size_OX[ext_mem]==dim->size_OX[int_mem]
        && dim->size_K[ext_mem]==dim->size_K[int_mem])
            dma_transfer_1d_async((DmaTransferConf) {
                .ext = ext_pt,
                .loc = int_pt,
                .length_1d_copy = dim->size_OY[int_mem]*dim->size_OX[int_mem]*dim->size_K[int_mem]*common_kernel->prec_O/8,
                .dir = 0
            });
        // inner dimensions not tiled, do 2D transfers
        else if(dim->size_OX[ext_mem]==dim->size_OX[int_mem] && dim->size_K[ext_mem]==dim->size_K[int_mem])
            dma_transfer_2d_async((DmaTransferConf) {
                .ext = ext_pt,
                .loc = int_pt,
                .number_of_1d_copies = dim->size_OY[int_mem],
                .length_1d_copy = dim->size_OX[int_mem]*dim->size_K[int_mem]*common_kernel->prec_O/8,
                .stride_1d = dim->size_OX[ext_mem]*dim->size_K[ext_mem]*common_kernel->prec_O/8,
                .dir = 0
            });
        // unlucky
        else
            dma_transfer_3d_async((DmaTransferConf) {
                .ext = ext_pt,
                .loc = int_pt,
                .number_of_2d_copies = dim->size_OY[int_mem],
                .number_of_1d_copies = dim->size_OX[int_mem],
                .length_1d_copy = dim->size_K[int_mem]*common_kernel->prec_O/8,
                .stride_2d = dim->size_K[ext_mem]*dim->size_OX[ext_mem]*common_kernel->prec_O/8,
                .stride_1d = dim->size_K[ext_mem]*common_kernel->prec_O/8,
                .dir = 0
            });
        #ifdef MATCH_NE16_BUFFERED
        dma_mutex_unlock();
        #endif
    }
    return;
}

unsigned int ne16_mem_transfer_I(common_kernel* common_kernel,dimension_I* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    inc_nnx_db_I(common_kernel->task_id);
    unsigned int dst=memalloc_I(common_kernel->task_id);
    if(common_kernel->task_id==LOADER_TASK || common_kernel->task_id==SINGLE_CORE_TASK){
        
        #ifdef MATCH_LOG_GAP9_VERBOSE
        printf("Dst I %d src %d task id %d\n",dst,ext_pt,common_kernel->task_id);
        #endif

        #ifdef MATCH_NE16_BUFFERED
        if(!input_dma_active){
            monitor_produce_begin(get_nnx_monitor()->input);
            dma_mutex_lock();
            input_transfers=dma_transfer_create();
            input_dma_active++;
        }
        #endif
        unsigned int src=ext_pt;

        // not dw so input channels not tiled! then if height and width are not tiled we can do 1D transfer
        if(dim->size_IY[ext_mem]==dim->size_IY[int_mem] && dim->size_IX[ext_mem]==dim->size_IX[int_mem]
        && dim->size_C[ext_mem]==dim->size_C[int_mem])
            dma_transfer_1d_async((DmaTransferConf) {
                .ext = ext_pt,
                .loc = dst,
                .length_1d_copy = (dim->size_IY[int_mem]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y)*
                (dim->size_IX[int_mem]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y)*
                dim->size_C[int_mem],
                .dir = 1
            });
        // not dw so input channels not tiled! then if width are not tiled we can do some 2D transfers
        else if(dim->size_IX[ext_mem]==dim->size_IX[int_mem] && dim->size_C[ext_mem]==dim->size_C[int_mem])
            dma_transfer_2d_async((DmaTransferConf) {
                .ext = ext_pt,
                .loc = dst,
                .number_of_1d_copies = dim->size_IY[int_mem]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y,
                .length_1d_copy = (dim->size_IX[int_mem]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y)*
                dim->size_C[int_mem],
                .stride_1d = dim->size_C[ext_mem]*dim->size_IX[ext_mem],
                .dir = 1
            });
        // unlucky
        else
            dma_transfer_3d_async((DmaTransferConf) {
                .ext = src,
                .loc = dst,
                .number_of_2d_copies = (dim->size_IY[int_mem]+ dim->overlap_IY_x + dim->overlap_IY_y - dim->pad_IY_x - dim->pad_IY_y),
                .number_of_1d_copies = (dim->size_IX[int_mem]+ dim->overlap_IX_x + dim->overlap_IX_y - dim->pad_IX_x - dim->pad_IX_y),
                .length_1d_copy = dim->size_C[int_mem],
                .stride_2d = dim->size_C[ext_mem]*dim->size_IX[ext_mem],
                .stride_1d = dim->size_C[ext_mem],
                .dir = 1
            });
    }
    return dst;
}

unsigned int ne16_mem_transfer_W(common_kernel* common_kernel,dimension_W* dim,unsigned int ext_pt,int ext_mem,int int_mem){
    inc_nnx_db_W(common_kernel->task_id);
    unsigned int dst=memalloc_W(common_kernel->task_id);
    if(common_kernel->task_id==LOADER_TASK || common_kernel->task_id==SINGLE_CORE_TASK){
        
        #ifdef MATCH_LOG_GAP9_VERBOSE
        printf("Dst W %d src %d task id %d\n",dst,ext_pt,common_kernel->task_id);
        #endif
        
        #ifdef MATCH_NE16_BUFFERED
        if(!input_dma_active){
            monitor_produce_begin(get_nnx_monitor()->input);
            dma_mutex_lock();
            input_transfers=dma_transfer_create();
            input_dma_active++;
        }
        #endif
        
        // output stationary so only output channels can be tiled for the weights, so if not tiled do 1D
        if(dim->size_K[ext_mem]==dim->size_K[int_mem])
            dma_transfer_1d_async((DmaTransferConf) {
                .ext = ext_pt,
                .loc = dst,
                .length_1d_copy = dim->size_K[int_mem]*dim->size_FY[int_mem]*dim->size_FX[int_mem]*dim->size_C[int_mem],
                .dir = 1
            });
        // else do some 2D transfers
        else if(!common_kernel->specific_pattern==depthwise_conv2d)
            dma_transfer_2d_async((DmaTransferConf) {
                .ext = ext_pt,
                .loc = dst,
                .number_of_1d_copies = dim->size_K[int_mem],
                .length_1d_copy = dim->size_C[int_mem]*dim->size_FY[int_mem]*dim->size_FX[int_mem],
                .stride_1d = dim->size_C[ext_mem]*dim->size_FY[ext_mem]*dim->size_FX[ext_mem],
                .dir = 1
            });
        else
            dma_transfer_2d_async((DmaTransferConf) {
                .ext = ext_pt,
                .loc = dst,
                .number_of_1d_copies = dim->size_K[int_mem]/16,
                .length_1d_copy = dim->size_K[int_mem]/16*dim->size_FY[int_mem]*dim->size_FX[int_mem],
                .stride_1d = 16*dim->size_FY[ext_mem]*dim->size_FX[ext_mem],
                .dir = 1
            });
    }
    return dst;
}

void ne16_wait_input_transfers(common_kernel* common_kernel){
    //printf("Wait input transfers task id %d\n",common_kernel->task_id);
    if(common_kernel->task_id==LOADER_TASK || common_kernel->task_id==SINGLE_CORE_TASK){
        #ifdef MATCH_NE16_BUFFERED
        //printf("Start waiting input transfers\n");
        dma_mutex_lock();
        dma_transfer_wait(input_transfers);
        input_dma_active=0;
        //dma_transfer_free(input_transfers);
        dma_mutex_unlock();
        //printf("Finished waiting input transfers\n");
        #else
        dma_transfer_wait(transfers);
        #endif
    }
}

void ne16_wait_output_transfers(common_kernel* common_kernel){
    if(common_kernel->task_id==STORER_TASK || common_kernel->task_id==SINGLE_CORE_TASK){
        #ifdef MATCH_NE16_BUFFERED
        //printf("Wait output transfers\n");
        dma_mutex_lock();
        dma_transfer_wait(output_transfers);
        //dma_transfer_free(output_transfers);
        dma_mutex_unlock();
        monitor_consume_end(get_nnx_monitor()->output);
        //printf("Waited output transfers\n");
        #else
        dma_transfer_wait(transfers);
        #endif
    }
}

void ne16_wait_curr_computation(common_kernel* common_kernel){
    if(common_kernel->task_id==STORER_TASK){
        monitor_consume_begin(get_nnx_monitor()->output);

        execute_wait(match_ne16_get_nnx_task(get_nnx_db_O(common_kernel->task_id)));
        monitor_consume_end(get_nnx_monitor()->input);
    }
    else if(common_kernel->task_id==EXECUTE_TASK){
        monitor_produce_end(get_nnx_monitor()->output);
    }
    else if(common_kernel->task_id==SINGLE_CORE_TASK){
        //printf("Wait comp...\n");
        execute_wait(match_ne16_get_nnx_task(get_nnx_db_O(common_kernel->task_id)));
        //printf("Comp finished\n");
    }
}

void ne16_pattern_constant_loading(match_kernel* kernel,unsigned int iter,tile_indexes_W* abs_tile_idx,
                                    tile_indexes_W* relative_tile_idx,void* weights_and_constant_buf){
    if(kernel->common_kernel->task_id==LOADER_TASK || kernel->common_kernel->task_id==SINGLE_CORE_TASK){
        //printf("Constants loading\n");
        #ifdef MATCH_NE16_BUFFERED
        if(!input_dma_active){
            monitor_produce_begin(get_nnx_monitor()->input);
            dma_mutex_lock();
            input_transfers=dma_transfer_create();
            input_dma_active++;
        }
        #endif
        if(iter==0){
            unsigned int size_weights=kernel->common_kernel->dim_W->size_K[l2_mem]*kernel->common_kernel->dim_W->size_C[l2_mem]*
            kernel->common_kernel->dim_W->size_FY[l2_mem]*kernel->common_kernel->dim_W->size_FX[l2_mem];
            dma_transfer_1d_async((DmaTransferConf) {
                .ext = ((unsigned int) weights_and_constant_buf)+size_weights,
                .loc = l1_bias_off+cluster_get_l1_memory_addr(),
                .length_1d_copy = 4*kernel->common_kernel->dim_W->size_K[l2_mem]*2,
                .dir = 1
            });
            kernel->common_kernel->batchnorm_mul=l1_bias_off+cluster_get_l1_memory_addr();
            kernel->common_kernel->batchnorm_add=l1_bias_off+cluster_get_l1_memory_addr()+4*kernel->common_kernel->dim_W->size_K[l2_mem];
        }
        else{
            kernel->common_kernel->batchnorm_mul=l1_bias_off+cluster_get_l1_memory_addr()+4*abs_tile_idx->tile_K;
            kernel->common_kernel->batchnorm_add=l1_bias_off+cluster_get_l1_memory_addr()+4*kernel->common_kernel->dim_W->size_K[l2_mem]+4*abs_tile_idx->tile_K;
        }
        #ifdef MATCH_NE16_BUFFERED
        dma_mutex_unlock();
        #endif
        //printf("Finished constant loading\n");
    }
}

unsigned int ne16_pointer_offset_NHWC_W(common_kernel* common_kernel,tile_indexes_W* tile_idxs,unsigned int memory_level){
    return common_kernel->specific_pattern==depthwise_conv2d?
    tile_idxs->tile_K*common_kernel->prec_W/8:match_pointer_offset_NHWC_W(common_kernel,tile_idxs,memory_level);
}