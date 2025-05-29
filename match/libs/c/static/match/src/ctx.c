#include <match/ctx.h>

MatchTensor* default_match_ctx_get_tensor(struct MatchTensors_t* self, const char* name) {
    for (int i = 0; i < self->num_tensors; i++) {
        if (match_strcmp(self->tensors_names[i], name) == 0) {
            return &(self->tensors[i]);
        }
    }
    return NULL;
}

int default_match_ctx_get_tensor_idx(struct MatchTensors_t* self, const char* name) {
    for (int i = 0; i < self->num_tensors; i++) {
        if (match_strcmp(self->tensors_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

MatchOp* default_match_ctx_get_op(struct MatchOps_t* self, const char* name) {
    for (int i = 0; i < self->num_ops; i++) {
        if (match_strcmp(self->ops_names[i], name) == 0) {
            return &(self->ops[i]);
        }
    }
    return NULL;
}

int default_match_ctx_get_op_idx(struct MatchOps_t* self, const char* name) {
    for (int i = 0; i < self->num_ops; i++) {
        if (match_strcmp(self->ops_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

MatchDim* default_match_ctx_get_dim(struct MatchDims_t* self, const char* name) {
    for (int i = 0; i < self->num_dims; i++) {
        if (match_strcmp(self->dims_names[i], name) == 0) {
            return &(self->dims[i]);
        }
    }
    return NULL;
}

int default_match_ctx_get_dim_idx(struct MatchDims_t* self, const char* name) {
    for (int i = 0; i < self->num_dims; i++) {
        if (match_strcmp(self->dims_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}