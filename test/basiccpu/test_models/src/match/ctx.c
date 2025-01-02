#include <match/ctx.h>

MatchVarTensor* default_match_ctx_get_var(struct MatchVars_t *self,const char *name){
    for (int i = 0; i < self->num_vars; i++) {
        if (match_strcmp(self->vars_names[i], name) == 0) {
            return &(self->tensors[i]);
        }
    }
    return NULL;
}

MatchConstTensor* default_match_ctx_get_const(struct MatchConsts_t *self,const char *name) {
    for (int i = 0; i < self->num_consts; i++) {
        if (match_strcmp(self->consts_names[i], name) == 0) {
            return &(self->tensors[i]);
        }
    }
    return NULL;
}

MatchOutputTensor* default_match_ctx_get_out(struct MatchOutputs_t *self,const char *name){
    for (int i = 0; i < self->num_outputs; i++) {
        if (match_strcmp(self->outputs_names[i], name) == 0) {
            return &(self->tensors[i]);
        }
    }
    return NULL;
}

MatchOp* default_match_ctx_get_op(struct MatchOps_t *self,const char *name){
    for (int i = 0; i < self->num_ops; i++) {
        if (match_strcmp(self->ops_names[i], name) == 0) {
            return &(self->ops[i]);
        }
    }
    return NULL;
}

MatchDim* default_match_ctx_get_dim(struct MatchDims_t *self,const char *name){
    for (int i = 0; i < self->num_dims; i++) {
        if (match_strcmp(self->dims_names[i], name) == 0) {
            return &(self->dims[i]);
        }
    }
    return NULL;
}