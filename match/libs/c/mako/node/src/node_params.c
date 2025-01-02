#include <nodes/${model_name}/${name}_params.h>

// DIMS
const char* ${name}_dims_names_[] = {
    % for idx,dim in enumerate(match_node.dims):
    ${", " if idx>0 else ""}"${dim}"
    % endfor
};
MatchDim ${name}_dims_[${len(match_node.dims)}] = {
% for idx,dim in enumerate(match_node.dims.values()):
    ${", " if idx>0 else ""}(MatchDim){
        .size = ${dim.size},
        .dynamic = 0,//${int(dim.is_dynamic)},
        .global_idx = 0,
        .curr_size = 0,
    }
% endfor
};

% for idx,dim in enumerate(match_node.dims.values()):
MatchDim* ${name}_${dim.name}_dim = &(${name}_dims_[${idx}]);
% endfor

MatchDims ${name}_dims_cnt_ = (MatchDims){
    .num_dims = ${len(match_node.dims)},
    .dims_names = ${name}_ops_names_,
    .get_dim = default_match_ctx_get_dim,
    .dims = ${name}_dims_
};


// TILES
% for t_tensor_name,t_tensor_tiles in schedule.tensor_tiles.items():
MatchTensorTile ${name}_${t_tensor_name}_tiles_[${len(t_tensor_tiles)}][${t_tensor_tiles[0].tensor.num_dims}] = {
    % for idx_mem_tile,mem_tile in enumerate(t_tensor_tiles):
    ${", " if idx_mem_tile else ""}{
        % for idx_mem_tile_dim,tiled_dim in enumerate(mem_tile.tiled_dims):
        ${", " if idx_mem_tile_dim else ""}(MatchTensorTile){
            .dim = &(${name}_dims_[${list(match_node.dims.keys()).index(tiled_dim.dim.name)}]),
            .size = ${tiled_dim.size},
            .start_idx = 0
        }
        % endfor
    }
    % endfor
};
MatchTensorTile** ${name}_${t_tensor_name}_tiles = (MatchTensorTile**)${name}_${t_tensor_name}_tiles_;
% endfor
// VARS
% if len(match_node.var_tensors)>0:
const char* ${name}_vars_names_[] = {
    % for idx,var in enumerate(match_node.var_tensors.values()):
    ${", " if idx>0 else ""}"${var.name}"
    % endfor
};
% endif

MatchVarTensor ${name}_vars_[${len(match_node.var_tensors)}] = {
% for idx,var in enumerate(match_node.var_tensors.values()):
    ${", " if idx>0 else ""}(MatchVarTensor){
        .num_dims = ${var.num_dims},
        .num_tiles = ${len(schedule.tensor_tiles[var.name])},
        .curr_tile = 0,
        .bits = ${var.bits},
        .base_pt = 0x0,
        .pts = {${str([0x0 for _ in range(len(schedule.tensor_tiles[var.name]))])[1:-1]}},
        .tiles = ${name}_${var.name}_tiles_
    }
% endfor
};

% for idx,var in enumerate(match_node.var_tensors.values()):
MatchVarTensor* ${name}_${var.name}_var = &(${name}_vars_[${idx}]);
% endfor

MatchVars ${name}_vars_cnt_ = (MatchVars){
    .num_vars = ${len(match_node.var_tensors)},
    .vars_names = ${name}_vars_names_,
    .get_var = default_match_ctx_get_var,
    .tensors = ${name}_vars_
};

// CONSTS
% if len(match_node.const_tensors)>0:
const char* ${name}_consts_names_[] = {
    % for idx,const_name in enumerate(match_node.const_tensors):
    ${", " if idx>0 else ""}"${const_name}"
    % endfor
};
% endif

MatchConstTensor ${name}_consts_[${len(match_node.const_tensors)}] = {
% for idx,const_ in enumerate(match_node.const_tensors.values()):
    ${", " if idx>0 else ""}(MatchConstTensor){
        .num_dims = ${const_.num_dims},
        .num_tiles = ${len(schedule.tensor_tiles[const_.name])},
        .curr_tile = 0,
        .bits = ${const_.bits},
        .base_pt = 0x0,
        .pts = {${str([0x0 for _ in range(len(schedule.tensor_tiles[const_.name]))])[1:-1]}},
        .tiles = ${name}_${const_.name}_tiles_
    }
% endfor
};
% for idx,const_ in enumerate(match_node.const_tensors.values()):
MatchConstTensor* ${name}_${const_.name}_const = &(${name}_consts_[${idx}]);
% endfor
MatchConsts ${name}_consts_cnt_ = (MatchConsts){
    .num_consts = ${len(match_node.const_tensors)},
    .consts_names = ${name}_consts_names_,
    .get_const = default_match_ctx_get_const,
    .tensors = ${name}_consts_
};

// OUTPUTS
% if len(match_node.output_tensors)>0:
const char* ${name}_outputs_names_[] = {
    % for idx,out_name in enumerate(match_node.output_tensors):
    ${", " if idx>0 else ""}"${out_name}"
    % endfor
};
% endif
MatchOutputTensor ${name}_outputs_[${len(match_node.output_tensors)}] = {
% for idx,out in enumerate(match_node.output_tensors.values()):
    ${", " if idx>0 else ""}(MatchOutputTensor){
        .num_dims = ${out.num_dims},
        .num_tiles = ${len(schedule.tensor_tiles[out.name])},
        .curr_tile = 0,
        .bits = ${out.bits},
        .base_pt = 0x0,
        .pts = {${str([0x0 for _ in range(len(schedule.tensor_tiles[out.name]))])[1:-1]}},
        .tiles = ${name}_${out.name}_tiles_
    }
% endfor
};
% for idx,out in enumerate(match_node.output_tensors.values()):
MatchOutputTensor* ${name}_${out.name}_out = &(${name}_outputs_[${idx}]);
% endfor

MatchOutputs ${name}_outputs_cnt_ = (MatchOutputs){
    .num_outputs = ${len(match_node.output_tensors)},
    .outputs_names = ${name}_ops_names_,
    .get_out = default_match_ctx_get_out,
    .tensors = ${name}_outputs_
};

// ops
const char* ${name}_ops_names_[] = {
    % for idx,op_name in enumerate(match_node.ops):
    ${", " if idx>0 else ""}"${op_name}"
    % endfor
};
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
Match${op.op}Attrs ${name}_op_${op_name}_attrs_ = (Match${op.op}Attrs){
    .idx = ${idx}
    % for attr_name,attr in op.c_attrs.items():
    ,.${attr_name} = ${attr}
    % endfor
};
Match${op.op}Attrs* ${name}_${op_name}_attrs = &(${name}_op_${op_name}_attrs_);
% endfor
MatchOp ${name}_ops_[${len(match_node.ops)}] = {
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
    ${", " if idx>0 else ""}(MatchOp){
        .op_code = ${op.op_code},
        .attrs = &${name}_op_${op_name}_attrs_
    }
% endfor
};
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
MatchOp* ${name}_${op_name}_op = &(${name}_ops_[${idx}]);
% endfor

MatchOps ${name}_ops_cnt_ = (MatchOps){
    .num_ops = ${len(match_node.ops)},
    .ops_names = ${name}_ops_names_,
    .get_op = default_match_ctx_get_op,
    .ops = ${name}_ops_
};

MatchCtx ${name}_ctx_ = (MatchCtx){
    .ctx_extension = 0x0,
    .vars = &${name}_vars_cnt_,
    .consts = &${name}_consts_cnt_,
    .outputs = &${name}_outputs_cnt_,
    .ops = &${name}_ops_cnt_,
    .dims = &${name}_dims_cnt_,
    .pattern_family = ${pattern_family},
    .pattern_name = ${pattern_name}
};

MatchCtx* ${name}_ctx = &${name}_ctx_;

% for dep_dim in match_node.dependent_dims:
void update_${dep_dim.name}(){
    ${name}_${dep_dim.name}_dim->global_idx = 
    % for idx_dep,(ind_dim,mult) in enumerate(dep_dim.dim_dependency.dependencies.items()):
    ${" + " if idx_dep>0 else ""}(${mult}*${name}_${ind_dim.name}_dim->global_idx)
    % endfor
    ;
    ${name}_${dep_dim.name}_dim->curr_size = 
    % for idx_dep,(ind_dim,mult) in enumerate(dep_dim.dim_dependency.dependencies.items()):
    ${" + " if idx_dep>0 else ""}(${mult}*${name}_${ind_dim.name}_dim->curr_size)
    % endfor
    ;
}
% endfor


// loops iters counters
% for block_idx,block in enumerate(schedule.blocks):
% for loop_idx,lp in enumerate(block.loops):
int loop_${lp.name}_iter = 0;
void loop_${lp.name}_set(int idx){
    ${name}_${lp.dim.name}_dim->global_idx = idx * ${lp.step};
    ${name}_${lp.dim.name}_dim->curr_size = ${lp.step};
    % for dep_dim in [dim.name for dim in match_node.dims.values() if dim.dim_dependency is not None and lp.dim in dim.dim_dependency.dependencies]:
    update_${dep_dim}();
    % endfor
}
% endfor
% endfor