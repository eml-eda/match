#include <nodes/${model_name}/${name}_params.h>

// DIMS
const char* ${name}_dims_names_[] = {
    % for idx,dim in enumerate(match_node.dims):
    ${", " if idx>0 else ""}"${dim}"
    % endfor
};
MatchDim ${name}_dims_[${len(match_node.dims)}+1] = {
% for idx,dim in enumerate(match_node.dims.values()):
    (MatchDim){
        .size = ${dim.size},
        .dynamic = 0,//${int(dim.is_dynamic)},
        .global_idx = ${dim.start_idx},
        .curr_size = ${dim.size},
        .curr_max_size = ${dim.max_size}
    },
% endfor
    // default dim
    (MatchDim){
        .size = 1,
        .dynamic = 0,//1,
        .global_idx = 0,
        .curr_size = 1,
        .curr_max_size = 1
    }
};

% for idx,dim in enumerate(match_node.dims.values()):
MatchDim* ${name}_${dim.name} = &(${name}_dims_[${idx}]);
% endfor
MatchDim* ${name}_default = &(${name}_dims_[${len(match_node.dims)}]);

MatchDims ${name}_dims_cnt_ = (MatchDims){
    .num_dims = ${len(match_node.dims)},
    .dims_names = ${name}_dims_names_,
    .get_dim = default_match_ctx_get_dim,
    .get_dim_idx = default_match_ctx_get_dim_idx,
    .dims = ${name}_dims_
};


// TILES
% for t_tensor_name,t_tensor_tiles in schedule.tensor_tiles.items():
MatchTensorTile ${name}_${t_tensor_name}_tiles_[${len(t_tensor_tiles)*t_tensor_tiles[0].tensor.num_dims}] = {
    % for idx_mem_tile,mem_tile in enumerate(t_tensor_tiles):
    % for idx_mem_tile_dim,tiled_dim in enumerate(mem_tile.tiled_dims):
    ${", " if (idx_mem_tile_dim+idx_mem_tile)>0 else ""}(MatchTensorTile){
        .dim = &(${name}_dims_[${list(match_node.dims.keys()).index(tiled_dim.dim.name) if tiled_dim.dim.name in match_node.dims else len(match_node.dims)}]),
        .size = ${tiled_dim.size},
        .max_size = ${tiled_dim.max_size},
        .start_idx = ${tiled_dim.dim.start_idx},
        .curr_idx = ${tiled_dim.dim.start_idx}
    }
    % endfor
    % endfor
};
MatchTensorTile* ${name}_${t_tensor_name}_tiles = (MatchTensorTile*)${name}_${t_tensor_name}_tiles_;
% endfor

const char* ${name}_tensors_names_[] = {
    % for idx,tens in enumerate(schedule.tensors.values()):
    ${", " if idx>0 else ""}"${tens.name}"
    % endfor
};

% for tensor in schedule.tensors.values():
unsigned int ${name}_${tensor.name}_pts[${len(schedule.tensor_tiles[tensor.name])}] = {${str([0x0 for _ in range(len(schedule.tensor_tiles[tensor.name]))])[1:-1]}};
% endfor

MatchTensor ${name}_tensors_[${len(schedule.tensors)}] = {
% for idx,tensor in enumerate(schedule.tensors.values()):
    ${", " if idx>0 else ""}(MatchTensor){
        .num_dims = ${tensor.num_dims},
        .num_tiles = ${len(schedule.tensor_tiles[tensor.name])},
        .curr_tile = 0,
        .bits = ${tensor.bits},
        .base_pt = 0x0,
        .pt = 0x0,
        .pts = ${name}_${tensor.name}_pts,
        .tiles = ${name}_${tensor.name}_tiles_
    }
% endfor
};

% for idx,tensor in enumerate(schedule.tensors.values()):
MatchTensor* ${name}_${tensor.name} = &(${name}_tensors_[${idx}]);
% endfor

MatchTensors ${name}_tensors_cnt_ = (MatchTensors){
    .num_tensors = ${len(schedule.tensors)},
    .tensors_names = ${name}_tensors_names_,
    .get_tensor = default_match_ctx_get_tensor,
    .get_tensor_idx = default_match_ctx_get_tensor_idx,
    .tensors = ${name}_tensors_
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
    .get_op_idx = default_match_ctx_get_op_idx,
    .ops = ${name}_ops_
};

MatchCtx ${name}_ctx_ = (MatchCtx){
    .ctx_extension = 0x0,
    .tensors = &${name}_tensors_cnt_,
    .ops = &${name}_ops_cnt_,
    .dims = &${name}_dims_cnt_,
    .exec_module = ${exec_module.name.upper()},
    .pattern_name = ${pattern_name}
};

MatchCtx* ${name}_ctx = &${name}_ctx_;

% for block_idx,block in enumerate(schedule.blocks):
% for loop_idx,lp in enumerate(block.loops):
int ${name}_block_${block_idx}_loop_${block.loops[loop_idx].name}_iter = 0;
% endfor
% endfor