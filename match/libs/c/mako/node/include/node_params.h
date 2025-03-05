#ifndef __MATCH_NODE_PARAMS_${node_fullname}_H__
#define __MATCH_NODE_PARAMS_${node_fullname}_H__

#include <match/ctx.h>
#include <match/utils.h>
#include <${target.name}.h>
#include <${exec_module.name}/${exec_module.name}.h>
#include <nodes/${model_name}/${name}_data.h>

// DIMS
extern const char* dims_names_[];
% if len(match_node.dims)>0:
extern MatchDim dims_[${len(match_node.dims)}];
% else:
extern MatchDim *dims_;
% endif
% for idx,dim in enumerate(match_node.dims.values()):
extern MatchDim* ${dim.name};
% endfor
extern MatchDims dims_cnt_;

// TILES
% for t_tensor_name,t_tensor_tiles in schedule.tensor_tiles.items():
extern MatchTensorTile ${t_tensor_name}_tiles_[${len(t_tensor_tiles)*t_tensor_tiles[0].tensor.num_dims}];
extern MatchTensorTile* ${t_tensor_name}_tiles;
% endfor

extern const char* tensors_names_[];
extern MatchTensor tensors_[${len(schedule.tensors)}];
% for idx,tensor in enumerate(schedule.tensors.values()):
extern MatchTensor* ${tensor.name};
% endfor
extern MatchTensors tensors_cnt_;

// ops
extern const char* ops_names_[];
% if len(match_node.ops)>0:
extern MatchOp ops_[${len(match_node.ops)}];
% else:
extern MatchOp *ops_;
% endif
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
extern Match${op.op}Attrs op_${op_name}_attrs_;
extern Match${op.op}Attrs* ${op_name}_attrs;
extern MatchOp* ${op_name}_op;
% endfor
extern MatchOps ops_cnt_;

extern MatchCtx ${name}_ctx_;

extern MatchCtx* ${name}_ctx;

% for dep_dim in match_node.dependent_dims:
inline void update_${dep_dim.name}(){
    ${dep_dim.name}->global_idx = 
    % for idx_dep,(ind_dim,mult) in enumerate(dep_dim.dim_dependency.dependencies.items()):
    ${" + " if idx_dep>0 else ""}(${mult}*${ind_dim if not hasattr(ind_dim,"name") else ind_dim.name+"->global_idx"})
    % endfor
    ;
}
inline void set_${dep_dim.name}(){
    ${dep_dim.name}->curr_size = 
    % for idx_dep,(ind_dim,mult) in enumerate(dep_dim.dim_dependency.dependencies.items()):
    ${" + " if idx_dep>0 else ""}(${mult}*${ind_dim if not hasattr(ind_dim,"name") else ind_dim.name+"->curr_size"})
    % endfor
    ;
}
% endfor


// loops iters counters
% for block_idx,block in enumerate(schedule.blocks):
% for loop_idx,lp in enumerate(block.loops):
extern int block_${block_idx}_loop_${block.loops[loop_idx].name}_iter;

inline void block_${block_idx}_loop_${lp.name}_set(){
    block_${block_idx}_loop_${block.loops[loop_idx].name}_iter = 0;
    ${lp.dim.name}->curr_size = ${lp.step*lp.size};
    % for dep_dim in [dim.name for dim in match_node.dims.values() if dim.dim_dependency is not None and lp.dim in dim.dim_dependency.dependencies]:
    set_${dep_dim}();
    % endfor
}
inline int block_${block_idx}_loop_${lp.name}_reset(){
    // block_${block_idx}_loop_${block.loops[loop_idx].name}_iter = 0;
    ${lp.dim.name}->global_idx -= ${lp.step*lp.size};
    return 0;
    % for dep_dim in [dim.name for dim in match_node.dims.values() if dim.dim_dependency is not None and lp.dim in dim.dim_dependency.dependencies]:
    set_${dep_dim}();
    update_${dep_dim}();
    % endfor
}
inline void block_${block_idx}_loop_${lp.name}_update(){
    block_${block_idx}_loop_${block.loops[loop_idx].name}_iter += 1;
    ${lp.dim.name}->global_idx += ${lp.step};
    % for dep_dim in [dim.name for dim in match_node.dims.values() if dim.dim_dependency is not None and lp.dim in dim.dim_dependency.dependencies]:
    update_${dep_dim}();
    % endfor
}
inline int block_${block_idx}_loop_${lp.name}_end(){
    return block_${block_idx}_loop_${block.loops[loop_idx].name}_iter >= ${lp.size} ? block_${block_idx}_loop_${lp.name}_reset() : 1;
}

% endfor
% endfor

#endif