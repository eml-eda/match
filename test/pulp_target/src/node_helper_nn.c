#if defined(__MATCH_NODE_HELPER_NN_H__)
#include <pulp_target/node_helper_nn.h>

// #define __MATCH_EXECUTE_BASIC_PARALLEL_SCHEDULE__
#define __MATCH_EXECUTE_PULP_NN_SCHEDULE__

void pulp_nn_schedule__(void* args){
	MatchCtx* ctx = (MatchCtx*) args;
	pulp_nn_conv_Ho_parallel(
		// input pt
        (unsigned int)(FunctionVar_0_0->base_pt),
		// Im2Col
        ((unsigned int)cast_out->base_pt) + NODE_OUT_CH*NODE_OUT_H*NODE_OUT_W,
		// bias
        (unsigned int)(FunctionVar_0_2->base_pt),
		// output pt
        (unsigned int)(cast_out->base_pt),
		// weights
        (unsigned int)(FunctionVar_0_1->base_pt),
        // batchnorm pts
		0x0,0x0,
		// out mult
        1,
		// right shift
        NODE_RIGHT_SHIFT,
		// inp w, inp h, inp ch
        NODE_INP_W,NODE_INP_H,NODE_INP_CH,
		// out w, out h, out ch
		NODE_OUT_W, NODE_OUT_H, NODE_OUT_CH,
        // f_h, f_w
        NODE_FIL_H,NODE_FIL_W,
		// paddings
        NODE_PAD_TOP,NODE_PAD_BOTTOM,NODE_PAD_LEFT,NODE_PAD_RIGHT,
		// strides
        NODE_STRIDE_W,NODE_STRIDE_H,
		// relu and batchnorm
        1,0
    );
}

void basic_parallel_pulp_schedule(void* args){
	MatchCtx* ctx = (MatchCtx*) args;
	int core_id = pi_core_id();
	for(int out_ch_idx = 0;
			out_ch_idx<NODE_OUT_CH;
			out_ch_idx++){
		for(int h_idx = core_id;
			h_idx<NODE_OUT_H;
			h_idx+=NUM_CORES){
			for(int w_idx = 0;
				w_idx<NODE_OUT_W;
				w_idx++){
				int conv2d_sum = ((int*)FunctionVar_0_2->base_pt)[out_ch_idx];
				for(int f_h_idx = 0;
					f_h_idx<NODE_FIL_H;
					f_h_idx++){
					for(int f_w_idx = 0;
						f_w_idx<NODE_FIL_W;
						f_w_idx++){
						int inp_h_idx = h_idx*NODE_STRIDE_H + f_h_idx - NODE_PAD_TOP;
						int inp_w_idx = w_idx*NODE_STRIDE_W + f_w_idx - NODE_PAD_LEFT;
						if (inp_h_idx < 0 || inp_h_idx >= NODE_INP_H || inp_w_idx < 0 || inp_w_idx >= NODE_INP_W) continue  ;
						for(int inp_ch_idx = 0;
							inp_ch_idx<NODE_INP_CH;
							inp_ch_idx++){
							conv2d_sum += ((int8_t*)FunctionVar_0_0->base_pt)[inp_h_idx * NODE_INP_W * NODE_INP_CH + inp_w_idx * NODE_INP_CH + inp_ch_idx] * ((int8_t*)FunctionVar_0_1->base_pt)[f_h_idx * NODE_FIL_W * NODE_INP_CH * NODE_OUT_CH + f_w_idx * NODE_INP_CH * NODE_OUT_CH + inp_ch_idx * NODE_OUT_CH + out_ch_idx];
						}
		
					}
					int8_t shifted_val = conv2d_sum >> NODE_RIGHT_SHIFT;
					((int8_t*)cast_out->base_pt)[h_idx * NODE_OUT_W * NODE_OUT_CH + w_idx * NODE_OUT_CH + out_ch_idx] = shifted_val>255?255:shifted_val<0?0:shifted_val;
				}
		
			}
		}
	}
	pi_cl_team_barrier(0);
}

void run_node_schedule_nn(MatchCtx* ctx){
    // allocate memory and set default parallelization
    unsigned int __MATCH_PULP_TARGET_L1_MEMORY_PT__ = pi_cl_l1_malloc(NULL, 90*1024);
	pi_team_config_offload(NUM_CORES);
    DmaTransfer transfer = dma_transfer_create();
	// input transfer
    dma_transfer_1d_async((DmaTransferConf) {
		.ext = (unsigned int)FunctionVar_0_0->base_pt,
		.loc = __MATCH_PULP_TARGET_L1_MEMORY_PT__,
		.length_1d_copy = NODE_INP_CH*NODE_INP_H*NODE_INP_W,
		.dir = 0
	});
    // weights+bias transfer
	dma_transfer_1d_async((DmaTransferConf) {
		.ext = (unsigned int)FunctionVar_0_1->base_pt,
		.loc = __MATCH_PULP_TARGET_L1_MEMORY_PT__+NODE_INP_CH*NODE_INP_H*NODE_INP_W,
		.length_1d_copy = NODE_INP_CH*NODE_OUT_CH*NODE_FIL_H*NODE_FIL_W,
		.dir = 0
	});
    dma_transfer_1d_async((DmaTransferConf) {
		.ext = (unsigned int)FunctionVar_0_2->base_pt,
		.loc = __MATCH_PULP_TARGET_L1_MEMORY_PT__+NODE_INP_CH*NODE_INP_H*NODE_INP_W + NODE_INP_CH*NODE_OUT_CH*NODE_FIL_H*NODE_FIL_W,
		.length_1d_copy = NODE_OUT_CH*4,
		.dir = 0
	});
    // wait for transfers
	dma_transfer_wait(transfer);
    // assign correct pointers
	FunctionVar_0_0->base_pt = __MATCH_PULP_TARGET_L1_MEMORY_PT__;
	FunctionVar_0_1->base_pt = __MATCH_PULP_TARGET_L1_MEMORY_PT__+NODE_INP_CH*NODE_INP_H*NODE_INP_W;
	FunctionVar_0_2->base_pt = __MATCH_PULP_TARGET_L1_MEMORY_PT__+NODE_INP_CH*NODE_INP_H*NODE_INP_W+NODE_INP_CH*NODE_OUT_CH*NODE_FIL_H*NODE_FIL_W;
	cast_out->base_pt = __MATCH_PULP_TARGET_L1_MEMORY_PT__+NODE_INP_CH*NODE_INP_H*NODE_INP_W+NODE_INP_CH*NODE_OUT_CH*NODE_FIL_H*NODE_FIL_W+NODE_OUT_CH*4;
	// run schedule
    #ifdef __MATCH_EXECUTE_PULP_NN_SCHEDULE__
	pi_team_offload_preset(pulp_nn_schedule__,ctx);
	#elif defined(__MATCH_EXECUTE_BASIC_PARALLEL_SCHEDULE__)
	pi_team_offload_preset(basic_parallel_pulp_schedule,ctx);
	#endif
    // wait parallel cores
	pi_team_offload_wait();
    // output transfer
	transfer = dma_transfer_create();
	dma_transfer_1d_async((DmaTransferConf) {
		.ext = (unsigned int)cast_out->base_pt,
		.loc = __MATCH_PULP_TARGET_L1_MEMORY_PT__+NODE_INP_CH*NODE_INP_H*NODE_INP_W+NODE_INP_CH*NODE_OUT_CH*NODE_FIL_H*NODE_FIL_W+NODE_OUT_CH*4,
		.length_1d_copy = NODE_OUT_CH*NODE_OUT_H*NODE_OUT_W,
		.dir = 1
	});
    // wait output transfer and clean
	dma_transfer_wait(transfer);
	dma_transfer_free(transfer);
	pi_cl_l1_free(NULL, __MATCH_PULP_TARGET_L1_MEMORY_PT__, 90*1024);
}

void run_pulp_cluster_fn(void (inner_function)(unsigned int* args_inner_function),unsigned int* args){
    pi_cluster_task(&cluster_task,inner_function,args);
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
}
#endif