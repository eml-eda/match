/*
 * Mohamed Amine Hamdi <mohamed.hamdi@polito.it>
 *
 * Copyright (C) 2024 Politecnico Di Torino
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
*/
// include params file
#include <nodes/default/main_0_params.h>
#include <stdio.h>

// #define __LOOP_OVER_OX_FIRST__
#define __LOOP_OVER_FILTER_WIDTH_FIRST__
#define __UNROLL_CHANNELS__
void* _var_pt__=0x0;
void* _out_pt__=0x0;

void block_0_compute(MatchCtx* ctx){
    // printf("\nBlock 0\n");
	#ifdef __LOOP_OVER_OX_FIRST__
    for(loop_OX_3_set();loop_OX_3_end();loop_OX_3_update()){
    	for(loop_OX_2_set();loop_OX_2_end();loop_OX_2_update()){
    		for(loop_OX_1_set();loop_OX_1_end();loop_OX_1_update()){
    			for(loop_OX_set();loop_OX_end();loop_OX_update()){
	#endif
	for(loop_OY_4_set();loop_OY_4_end();loop_OY_4_update()){
		for(loop_OY_3_set();loop_OY_3_end();loop_OY_3_update()){
			for(loop_OY_2_set();loop_OY_2_end();loop_OY_2_update()){
				for(loop_OY_1_set();loop_OY_1_end();loop_OY_1_update()){
	#ifndef __LOOP_OVER_OX_FIRST__
					for(loop_OX_3_set();loop_OX_3_end();loop_OX_3_update()){
						for(loop_OX_2_set();loop_OX_2_end();loop_OX_2_update()){
							for(loop_OX_1_set();loop_OX_1_end();loop_OX_1_update()){
								for(loop_OX_set();loop_OX_end();loop_OX_update()){
	#endif
    								int sum = main_0_FunctionVar_0_2_const_data[0];
									#ifndef __LOOP_OVER_FILTER_FIRST__
									#ifndef __UNROLL_CHANNELS__
									for(loop_C_set();loop_C_end();loop_C_update()){
									#endif
									#endif
									#ifdef __LOOP_OVER_FILTER_WIDTH_FIRST__
									for(loop_FX_set();loop_FX_end();loop_FX_update()){
									#endif
    									for(loop_FY_set();loop_FY_end();loop_FY_update()){
									#ifndef __LOOP_OVER_FILTER_WIDTH_FIRST__
										for(loop_FX_set();loop_FX_end();loop_FX_update()){
									#endif
											update_FunctionVar_0_0_dim_2();
											update_FunctionVar_0_0_dim_3();									
											if(main_0_FunctionVar_0_0_dim_2_dim->global_idx<0
												|| main_0_FunctionVar_0_0_dim_2_dim->global_idx>=main_0_FunctionVar_0_0_dim_2_dim->size
												|| main_0_FunctionVar_0_0_dim_3_dim->global_idx<0
												|| main_0_FunctionVar_0_0_dim_3_dim->global_idx>=main_0_FunctionVar_0_0_dim_3_dim->size)
												continue;
											#ifdef __LOOP_OVER_FILTER_FIRST__
											#ifndef __UNROLL_CHANNELS__
											for(loop_C_set();loop_C_end();loop_C_update()){
											#endif
											#endif
												unsigned int __const_idx__ = main_0_FunctionVar_0_0_dim_1_dim->global_idx*9
													+main_0_FunctionVar_0_1_dim_2_dim->global_idx*3
													+main_0_FunctionVar_0_1_dim_3_dim->global_idx;
												unsigned int __var_idx__ = main_0_FunctionVar_0_0_dim_1_dim->global_idx*32*32
													+(main_0_FunctionVar_0_0_dim_3_dim->global_idx)*32
													+(main_0_FunctionVar_0_0_dim_2_dim->global_idx);
												sum += ((int8_t*)_var_pt__)[__var_idx__]*main_0_FunctionVar_0_1_const_data[__const_idx__]
														#ifdef __UNROLL_CHANNELS__
														+((int8_t*)_var_pt__)[__var_idx__+32*32]*main_0_FunctionVar_0_1_const_data[__const_idx__+9]
														+((int8_t*)_var_pt__)[__var_idx__+32*32*2]*main_0_FunctionVar_0_1_const_data[__const_idx__+9*2]
														#endif
														;
											#ifndef __UNROLL_CHANNELS__	
											}
											#endif
    									}
    								}
									unsigned int _out_idx_ = main_0_conv2d_out_h_dim->global_idx*16+main_0_conv2d_out_w_dim->global_idx;
									// store ReLU result
									((int*)_out_pt__)[_out_idx_] = sum>0?sum:0;
    							}
    						}
    					}
    				}
    			}
    		}
    	}
    }
}


int __attribute__ ((noinline)) tvmgen_default_match_main_0(
    void* var_FunctionVar_0_0_pt,
    void* out_relu_pt
)
{
	MatchCtx ctx = main_0_ctx_;
    main_0_FunctionVar_0_0_var->base_pt = var_FunctionVar_0_0_pt;
    main_0_relu_out->base_pt = out_relu_pt;
	_var_pt__ = var_FunctionVar_0_0_pt;
	_out_pt__ = out_relu_pt;
	// printf("\nConsts vals 0_1: [%d %d %d %d], 0_2: [%d]\n", main_0_FunctionVar_0_1_const_data[0], main_0_FunctionVar_0_1_const_data[1], main_0_FunctionVar_0_1_const_data[2], main_0_FunctionVar_0_1_const_data[3], main_0_FunctionVar_0_2_const_data[0]);
	// printf("\nConsts ptr 0_1: %d, 0_2: %d\n", main_0_FunctionVar_0_1_const_data, main_0_FunctionVar_0_2_const_data);
	main_0_FunctionVar_0_1_const->base_pt = main_0_FunctionVar_0_1_const_data;
	main_0_FunctionVar_0_2_const->base_pt = main_0_FunctionVar_0_2_const_data;
	// printf("\nConst ptr from base ptr 0_1: %d, 0_2: %d\n", main_0_FunctionVar_0_1_const->base_pt, main_0_FunctionVar_0_2_const->base_pt);
	// printf("\nConst vals from base ptr 0_1: [%d], 0_2: []\n",
	// *((int8_t*)main_0_FunctionVar_0_1_const_data)
	// // ,*((int*)main_0_FunctionVar_0_2_const->base_pt)
	// );

	
	
    // block 0
	
    block_0_compute(&ctx);

    return 0;
}