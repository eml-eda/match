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
expected latency -> ${latency}
expected energy -> ${energy}
*/

/*
${for_loops}
*/

% for inc_name in include_list:
#include <${inc_name}>
% endfor


typedef struct loops_idxs_t
{
    % for lp in for_loops:
    unsigned int ${lp.name};
    % endfor
}loops_idxs;

typedef struct dims_idxs_t
{
    % for dim in independent_dims:
    unsigned int ${dim.name};
    % endfor
}dims_idxs;


% for dim,dependency in dims_dependencies.items():
static inline get_${dim.name}(match_context* ctx){
    return (
        % for idx_dep,(dependency_dim,coefficient) in enumerate(dependency.items()):
        ${"+" if idx_dep>0 else ""}(ctx->dims_idxs->${dependency_dim.name}*${coefficient})
        % endfor
    )
}
% endfor

int __attribute__ ((noinline)) ${func_name}(
    % for idx,operand in enumerate(operands):
    % if operand.is_var:
    ${"," if idx>0 else ""}void* var_${operand.name}_pt
    % endif
    % if operand.is_output:
    ${"," if idx>0 else ""}void* output_${operand.name}_pt
    % endif
    % endfor
)
{
    // declare the ctx
    MatchContext ctx = (MatchContext){
        .ctx_extension = &(${ctx_extension}){0},
        .loops_idxs = &(loops_idxs){
            % for idx,lp in enumerate(for_loops):
            ${"," if idx>0 else ""}.${lp.name}=0
            % endfor
        },
        .pattern_family = ${pattern_family},
        .pattern_name = ${specific_pattern}
        // TODO: is there anything else we can assign already? 
    };

    ## define dimension structures
    % for operand in operands:
    % if operand.is_var:
    ctx.vars[${operand.idx}] = &MatchVar${len(operand.dims)}D();
    ctx.vars[${operand.idx}]->base_pt = var_${operand.name}_pt;
    ctx.vars[${operand.idx}]->pt = var_${operand.name}_pt;
    % elif operand.is_output:
    ctx.output = &MatchOutput${len(operand.dims)}D();
    ctx.output->base_pt = output_${operand.name}_pt;
    ctx.output[${operand.idx}]->pt = output_${operand.name}_pt;
    % elif operand.is_constant:
    ctx.consts[${operand.idx}] = &MatchConst${len(operand.dims)}D();
    ctx.consts[${operand.idx}]->base_pt = constants.${operand.name}_data;
    ctx.consts[${operand.idx}]->pt = constants.${operand.name}_data;
    % else:
    // TODO: are all operands vars constants or just output?
    //unsigned int* ${operand}_pt = other_${operand}_pt;
    % endif
    ## """${operands[operand].dims}D_dimension dim_${operand}; ${operands[operand].dims}D_tile_indexes abs_tile_idxs_${operand};"""
    // TODO: is there anything else to setup like dims?
    % endfor
    
    ## TODO: define a way to set all operations attributes, and use them for pointer calculation for example

    ## FOR LOOPS
    % for for_loop in for_loops:
    % for load in for_loop.loads:
    % if load.operand.is_var:
    ## TODO: actually compute the new pt, based from the relative tile idxs
    //ctx.vars[${load.operand.idx}].pt = ctx.vars[${load.operand.idx}].base_pt;
    % endif
    % endfor
    ## WRITE LOOP
    for(;ctx.loops_idxs->${for_loop.name}<${for_loop.size};loop_${for_loop.name}_set(&ctx)){
        
    % endfor
    ## close braces and save output
    % for idx in reversed(range(len(for_loops))):
    }
    ## USELESS TO DO FOR THE OUTER LOOP
    % if idx>0:
    loop_${for_loop.name}_reset(&ctx);
    % endif
    % endfor

    return 0;
}