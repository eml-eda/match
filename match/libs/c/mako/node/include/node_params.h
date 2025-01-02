#ifndef __MATCH_NODE_PARAMS_${node_fullname}_H__
#define __MATCH_NODE_PARAMS_${node_fullname}_H__

#include <match/ctx.h>
#include <match/utils.h>
#include <${target.name}.h>
#include <${exec_module.name}/${exec_module.name}.h>
#include <nodes/${model_name}/${name}_data.h>
// loops iters counters
% for block_idx,block in enumerate(schedule.blocks):
% for loop_idx in range(len(block.loops)):
extern int loop_${block.loops[loop_idx].name}_iter;
% endfor
% endfor

// DIMS
extern const char* ${name}_dims_names_[];
% if len(match_node.dims)>0:
extern MatchDim ${name}_dims_[${len(match_node.dims)}];
% else:
extern MatchDim *${name}_dims_;
% endif
% for idx,dim in enumerate(match_node.dims.values()):
extern MatchDim* ${name}_${dim.name}_dim;
% endfor
extern MatchDims ${name}_dims_cnt_;

// TILES
% for t_tensor_name,t_tensor_tiles in schedule.tensor_tiles.items():
extern MatchTensorTile ${name}_${t_tensor_name}_tiles_[${len(t_tensor_tiles)}][${t_tensor_tiles[0].tensor.num_dims}];
extern MatchTensorTile** ${name}_${t_tensor_name}_tiles;
% endfor
// VARS
extern const char* ${name}_vars_names_[];
% if len(match_node.var_tensors)>0:
extern MatchVarTensor ${name}_vars_[${len(match_node.var_tensors)}];
% else:
extern MatchVarTensor *${name}_vars_;
% endif
% for idx,var in enumerate(match_node.var_tensors.values()):
extern MatchVarTensor* ${name}_${var.name}_var;
% endfor
extern MatchVars ${name}_vars_cnt_;

// CONSTS
extern const char* ${name}_consts_names_[];
% if len(match_node.const_tensors)>0:
extern MatchConstTensor ${name}_consts_[${len(match_node.const_tensors)}];
% else:
extern MatchConstTensor *${name}_consts_;
% endif
% for idx,const_ in enumerate(match_node.const_tensors.values()):
extern MatchConstTensor* ${name}_${const_.name}_const;
% endfor
extern MatchConsts ${name}_consts_cnt_;

// OUTPUTS
extern const char* ${name}_outputs_names_[];
% if len(match_node.output_tensors)>0:
extern MatchOutputTensor ${name}_outputs_[${len(match_node.output_tensors)}];
% else:
extern MatchOutputTensor *${name}_outputs_;
% endif
% for idx,out in enumerate(match_node.output_tensors.values()):
extern MatchOutputTensor* ${name}_${out.name}_out;
% endfor
extern MatchOutputs ${name}_outputs_cnt_;

// ops
extern const char* ${name}_ops_names_[];
% if len(match_node.ops)>0:
extern MatchOp ${name}_ops_[${len(match_node.ops)}];
% else:
extern MatchOp *${name}_ops_;
% endif
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
extern Match${op.op}Attrs ${name}_op_${op_name}_attrs_;
extern Match${op.op}Attrs* ${name}_${op_name}_attrs;
extern MatchOp* ${name}_${op_name}_op;
% endfor
extern MatchOps ${name}_ops_cnt_;

extern MatchCtx ${name}_ctx_;

extern MatchCtx* ${name}_ctx;

#endif