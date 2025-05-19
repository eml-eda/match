#ifndef __MATCH_NODE_PARAMS_${node_fullname}_H__
#define __MATCH_NODE_PARAMS_${node_fullname}_H__

#include <match/ctx.h>
#include <match/utils.h>
#include <${target.name}.h>
#include <${exec_module.name}.h>
#include <nodes/${model_name}/${name}_data.h>
% for inc_lib in exec_module.include_list():
#include <${inc_lib}.h>
% endfor

// DIMS
extern const char* ${name}_dims_names_[];
extern MatchDim ${name}_dims_[${len(match_node.dims)}+1];
% for idx,dim in enumerate(match_node.dims.values()):
extern MatchDim* ${name}_${dim.name};
% endfor
extern MatchDim* ${name}_default;
extern MatchDims ${name}_dims_cnt_;

// TILES
% for t_tensor_name,t_tensor_tiles in schedule.tensor_tiles.items():
extern MatchTensorTile ${name}_${t_tensor_name}_tiles_[${len(t_tensor_tiles)*t_tensor_tiles[0].tensor.num_dims}];
extern MatchTensorTile* ${name}_${t_tensor_name}_tiles;
% endfor


extern const char* ${name}_tensors_names_[];
extern MatchTensor ${name}_tensors_[${len(schedule.tensors)}];
% for idx,tensor in enumerate(schedule.tensors.values()):
extern unsigned int ${name}_${tensor.name}_pts[${len(schedule.tensor_tiles[tensor.name])}];
extern MatchTensor* ${name}_${tensor.name};
% endfor
extern MatchTensors ${name}_tensors_cnt_;

// ops
extern const char* ${name}_ops_names_[];
% if len(match_node.ops)>0:
extern MatchOp ${name}_ops_[${len(match_node.ops)}];
% else:
extern MatchOp* ${name}_ops_;
% endif
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
extern Match${op.op}Attrs ${name}_op_${op_name}_attrs_;
extern Match${op.op}Attrs* ${name}_${op_name}_attrs;
extern MatchOp* ${name}_${op_name}_op;
% endfor
extern MatchOps ${name}_ops_cnt_;

extern MatchCtx ${name}_ctx_;

extern MatchCtx* ${name}_ctx;

% for dep_dim in match_node.dependent_dims:
inline void ${name}_update_${dep_dim.name}(){
    ${name}_${dep_dim.name}->global_idx = 
    % for idx_dep,(ind_dim,mult) in enumerate(dep_dim.dim_dependency.idx_dependencies.items()):
    ${" + " if idx_dep>0 else ""}(${mult}*${ind_dim if not hasattr(ind_dim,"name") else name+"_"+ind_dim.name+"->global_idx"})
    % endfor
    ;
    ${name}_${dep_dim.name}->curr_max_size = 
    % for idx_dep,(ind_dim,mult) in enumerate(dep_dim.dim_dependency.size_dependencies.items()):
    ${" + " if idx_dep>0 else ""}(${mult}*${ind_dim if not hasattr(ind_dim,"name") else name+"_"+ind_dim.name+"->curr_size"})
    % endfor
    ;
    // if ${dep_dim.name} goes behind 0 it means there was padding so that shouldnt count for the size
    // same if it goes over the real size
    int max_size = (${name}_${dep_dim.name}->global_idx + ${name}_${dep_dim.name}->curr_max_size) - ${name}_${dep_dim.name}->size;
    int min_size = -${name}_${dep_dim.name}->global_idx;
    ${name}_${dep_dim.name}->curr_size = ${name}_${dep_dim.name}->curr_max_size - ((max_size>0?max_size:0) + (min_size>0?min_size:0));
}
% endfor


// loops iters counters
% for block_idx,block in enumerate(schedule.blocks):
% for loop_idx,lp in enumerate(block.loops):
extern int ${name}_block_${block_idx}_loop_${block.loops[loop_idx].name}_iter;

inline void ${name}_block_${block_idx}_loop_${lp.name}_set(){
    ${name}_block_${block_idx}_loop_${block.loops[loop_idx].name}_iter = 0;
    ${name}_${lp.dim.name}->curr_size = ${lp.step};
    % for dep_dim in [dim.name for dim in match_node.dims.values() if dim.dim_dependency is not None and lp.dim in dim.dim_dependency.dependencies]:
    ${name}_update_${dep_dim}();
    % endfor
}
inline int ${name}_block_${block_idx}_loop_${lp.name}_reset(){
    // ${name}_block_${block_idx}_loop_${block.loops[loop_idx].name}_iter = 0;
    ${name}_${lp.dim.name}->global_idx -= ${lp.step*lp.size};
    % for dep_dim in [dim.name for dim in match_node.dims.values() if dim.dim_dependency is not None and lp.dim in dim.dim_dependency.dependencies]:
    ${name}_update_${dep_dim}();
    % endfor
    return 0;
}
inline void ${name}_block_${block_idx}_loop_${lp.name}_update(){
    ${name}_block_${block_idx}_loop_${block.loops[loop_idx].name}_iter += 1;
    ${name}_${lp.dim.name}->global_idx += ${lp.step};
    % for dep_dim in [dim.name for dim in match_node.dims.values() if dim.dim_dependency is not None and lp.dim in dim.dim_dependency.dependencies]:
    ${name}_update_${dep_dim}();
    % endfor
}
inline int ${name}_block_${block_idx}_loop_${lp.name}_end(){
    return ${name}_block_${block_idx}_loop_${block.loops[loop_idx].name}_iter >= ${lp.size} ? ${name}_block_${block_idx}_loop_${lp.name}_reset() : 1;
}

% endfor
% endfor

#endif