#include <nodes/${model_name}/${name}_params.h>

// DIMS
const char* dims_names_[] = {
    % for idx,dim in enumerate(match_node.dims):
    ${", " if idx>0 else ""}"${dim}"
    % endfor
};
MatchDim dims_[${len(match_node.dims)}] = {
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
MatchDim* ${dim.name} = &(dims_[${idx}]);
% endfor

MatchDims dims_cnt_ = (MatchDims){
    .num_dims = ${len(match_node.dims)},
    .dims_names = ops_names_,
    .get_dim = default_match_ctx_get_dim,
    .get_dim_idx = default_match_ctx_get_dim_idx,
    .dims = dims_
};


// TILES
% for t_tensor_name,t_tensor_tiles in schedule.tensor_tiles.items():
MatchTensorTile ${t_tensor_name}_tiles_[${len(t_tensor_tiles)}][${t_tensor_tiles[0].tensor.num_dims}] = {
    % for idx_mem_tile,mem_tile in enumerate(t_tensor_tiles):
    ${", " if idx_mem_tile else ""}{
        % for idx_mem_tile_dim,tiled_dim in enumerate(mem_tile.tiled_dims):
        ${", " if idx_mem_tile_dim else ""}(MatchTensorTile){
            .dim = &(dims_[${list(match_node.dims.keys()).index(tiled_dim.dim.name)}]),
            .size = ${tiled_dim.size},
            .start_idx = 0
        }
        % endfor
    }
    % endfor
};
MatchTensorTile** ${t_tensor_name}_tiles = (MatchTensorTile**)${t_tensor_name}_tiles_;
% endfor

const char* tensors_names_[] = {
    % for idx,tens in enumerate(schedule.tensors.values()):
    ${", " if idx>0 else ""}"${tens.name}"
    % endfor
};

MatchTensor tensors_[${len(schedule.tensors)}] = {
% for idx,tensor in enumerate(schedule.tensors.values()):
    ${", " if idx>0 else ""}(MatchTensor){
        .num_dims = ${tensor.num_dims},
        .num_tiles = ${len(schedule.tensor_tiles[tensor.name])},
        .curr_tile = 0,
        .bits = ${tensor.bits},
        .base_pt = 0x0,
        .pts = {${str([0x0 for _ in range(len(schedule.tensor_tiles[tensor.name]))])[1:-1]}},
        .tiles = ${tensor.name}_tiles_
    }
% endfor
};

% for idx,tensor in enumerate(schedule.tensors.values()):
MatchTensor* ${tensor.name} = &(tensors_[${idx}]);
% endfor

MatchTensors tensors_cnt_ = (MatchTensors){
    .num_tensors = ${len(schedule.tensors)},
    .tensors_names = tensors_names_,
    .get_tensor = default_match_ctx_get_tensor,
    .get_tensor_idx = default_match_ctx_get_tensor_idx,
    .tensors = tensors_
};

// ops
const char* ops_names_[] = {
    % for idx,op_name in enumerate(match_node.ops):
    ${", " if idx>0 else ""}"${op_name}"
    % endfor
};
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
Match${op.op}Attrs op_${op_name}_attrs_ = (Match${op.op}Attrs){
    .idx = ${idx}
    % for attr_name,attr in op.c_attrs.items():
    ,.${attr_name} = ${attr}
    % endfor
};
Match${op.op}Attrs* ${op_name}_attrs = &(op_${op_name}_attrs_);
% endfor
MatchOp ops_[${len(match_node.ops)}] = {
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
    ${", " if idx>0 else ""}(MatchOp){
        .op_code = ${op.op_code},
        .attrs = &op_${op_name}_attrs_
    }
% endfor
};
% for idx,(op_name,op) in enumerate(match_node.ops.items()):
MatchOp* ${op_name}_op = &(ops_[${idx}]);
% endfor

MatchOps ops_cnt_ = (MatchOps){
    .num_ops = ${len(match_node.ops)},
    .ops_names = ops_names_,
    .get_op = default_match_ctx_get_op,
    .get_op_idx = default_match_ctx_get_op_idx,
    .ops = ops_
};

MatchCtx ${name}_ctx_ = (MatchCtx){
    .ctx_extension = 0x0,
    .tensors = &tensors_cnt_,
    .ops = &ops_cnt_,
    .dims = &dims_cnt_,
    .pattern_family = ${pattern_family},
    .pattern_name = ${pattern_name}
};

MatchCtx* ${name}_ctx = &${name}_ctx_;

% for block_idx,block in enumerate(schedule.blocks):
% for loop_idx,lp in enumerate(block.loops):
int block_${block_idx}_loop_${block.loops[loop_idx].name}_iter = 0;
% endfor
% endfor