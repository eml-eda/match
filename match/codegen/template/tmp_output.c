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

/*
expected latency -> 0
expected energy -> 0
*/

/*
[<__main__.ForLoop object at 0x7f13e2e278e0>, <__main__.ForLoop object at 0x7f13e2e27af0>, <__main__.ForLoop object at 0x7f13e2c58340>, <__main__.ForLoop object at 0x7f13e2c58370>, <__main__.ForLoop object at 0x7f13e2c583d0>]
*/

#include <stdio.h>
#include <stdlib.h>


typedef struct loops_idxs_t
{
    unsigned int ch_outer;
    unsigned int w_outer;
    unsigned int height;
    unsigned int channel;
    unsigned int width;
}loops_idxs;

typedef struct dims_idxs_t
{
    unsigned int channels;
    unsigned int height;
    unsigned int width;
}dims_idxs;


static inline get_inp_height(match_context* ctx){
    return (
        (ctx->dims_idxs->height*2)
    )
}

int __attribute__ ((noinline)) testing_llm_longlivematch(
    void* var_inp_K_pt
    ,void* var_inp_Q_pt
    ,void* var_inp_V_pt
    ,void* var_inp_W_pt
    ,void* var_out_pt
    ,void* output_out_pt
)
{
    // declare the ctx
    MatchContext ctx = (MatchContext){
        .ctx_extension = &(void){0},
        .loops_idxs = &(loops_idxs){
            .ch_outer=0
            ,.w_outer=0
            ,.height=0
            ,.channel=0
            ,.width=0
        },
        .pattern_family = conv2d,
        .pattern_name = conv2d_biasadd_req
        // TODO: is there anything else we can assign already? 
    };

    ctx.vars[0] = &MatchVar3D();
    ctx.vars[0]->base_pt = var_inp_K_pt;
    ctx.vars[0]->pt = var_inp_K_pt;
    // TODO: is there anything else to setup like dims?
    ctx.vars[1] = &MatchVar3D();
    ctx.vars[1]->base_pt = var_inp_Q_pt;
    ctx.vars[1]->pt = var_inp_Q_pt;
    // TODO: is there anything else to setup like dims?
    ctx.vars[2] = &MatchVar3D();
    ctx.vars[2]->base_pt = var_inp_V_pt;
    ctx.vars[2]->pt = var_inp_V_pt;
    // TODO: is there anything else to setup like dims?
    ctx.vars[3] = &MatchVar3D();
    ctx.vars[3]->base_pt = var_inp_W_pt;
    ctx.vars[3]->pt = var_inp_W_pt;
    // TODO: is there anything else to setup like dims?
    ctx.vars[0] = &MatchVar3D();
    ctx.vars[0]->base_pt = var_out_pt;
    ctx.vars[0]->pt = var_out_pt;
    // TODO: is there anything else to setup like dims?
    

    //ctx.vars[0].pt = ctx.vars[0].base_pt;
    for(;ctx.loops_idxs->ch_outer<4;loop_ch_outer_set(&ctx)){
        
    //ctx.vars[1].pt = ctx.vars[1].base_pt;
    //ctx.vars[2].pt = ctx.vars[2].base_pt;
    //ctx.vars[3].pt = ctx.vars[3].base_pt;
    for(;ctx.loops_idxs->w_outer<4;loop_w_outer_set(&ctx)){
        
    //ctx.vars[0].pt = ctx.vars[0].base_pt;
    for(;ctx.loops_idxs->height<32;loop_height_set(&ctx)){
        
    for(;ctx.loops_idxs->channel<8;loop_channel_set(&ctx)){
        
    for(;ctx.loops_idxs->width<8;loop_width_set(&ctx)){
        
    }
    loop_width_reset(&ctx);
    }
    loop_width_reset(&ctx);
    }
    loop_width_reset(&ctx);
    }
    loop_width_reset(&ctx);
    }

    return 0;
}