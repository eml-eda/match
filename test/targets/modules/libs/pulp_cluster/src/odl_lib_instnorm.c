#include <pulp_cluster/odl_lib.h>

// Forward instance norm tail
// Ordering:
// 0: x      (1,C,S)
// 1: mean   (1,C) or (C)
// 2: var    (1,C) or (C)
// 3: eps    (1) or ()
// 4: gamma  (optional) (1,C) or (C)
// 5: beta   (optional) (1,C) or (C)
// last: out (1,C,S)
void odl_fw_instance_norm_tail_fp32(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int num_ops = ctx->ops->num_ops;
    int out_idx_tensor = num_tensors - 1;

    float * __restrict__ x_pt    = (float*)tensors[0].pt;
    float * __restrict__ mean_pt = (float*)tensors[1].pt;
    float * __restrict__ var_pt  = (float*)tensors[2].pt;
    float * __restrict__ gamma_pt = NULL;
    float * __restrict__ beta_pt  = NULL;
    if(num_ops > 6){
        gamma_pt = (float*)tensors[4].pt;
        beta_pt  = (float*)tensors[5].pt;
    }
    // 6 op means either gamma or beta, since there are always 5(subtract, add, rsqrt, multiply, reshape)
    // 2 optionals are multiply and add, so if last op is multiply then gamma, else beta
    else if(num_ops == 6){
        // last op is reshape, last to one is multiply so GAMMA
        if(ctx->ops->ops[num_ops-2].op_code==MATCH_OP_MULTIPLY)
            gamma_pt = (float*)tensors[4].pt; // only gamma
        // last op is reshape, last to one is add so BETA
        else if(ctx->ops->ops[num_ops-2].op_code==MATCH_OP_ADD)
            beta_pt = (float*)tensors[4].pt; // only beta
    }
    float * __restrict__ out_pt  = (float*)tensors[out_idx_tensor].pt;

    const float eps = *((float*)tensors[3].pt);

    // Tensor 0 has 3 dims: (B=1, C, S)
    const int B = tensors[0].tiles[L1_SCRATCHPAD*3+0].size; // expected 1
    const int C = tensors[0].tiles[L1_SCRATCHPAD*3+1].size;
    const int S = tensors[0].tiles[L1_SCRATCHPAD*3+2].size; // flattened spatial (H*W)

    // Parallelization strategy: choose larger between C and S to distribute across cores
    int start_c = 0, stop_c = C;
    int start_s = 0, stop_s = S;
    #ifdef ODL_DEBUG_KERNEL
    if(pi_core_id() == 0){
        // Debug print for tensor shapes
        printf("InstanceNorm: x=(%d,%d,%d), mean=(%d,%d), var=(%d,%d), gamma=(%d,%d), beta=(%d,%d), out=(%d,%d,%d)\n",
            B, C, S, 1, C, 1, C, num_tensors > 4 ? 1 : 0, C, num_tensors > 5 ? 1 : 0, C, B, C, S);
        printf("InstanceNorm: eps=%f\n", eps);
        printf("InstanceNorm: start_c=%d, stop_c=%d, start_s=%d, stop_s=%d\n", start_c, stop_c, start_s, stop_s);
        printf("InstanceNorm: x_pt=%p, mean_pt=%p, var_pt=%p, gamma_pt=%p, beta_pt=%p, out_pt=%p\n",
            x_pt, mean_pt, var_pt, gamma_pt, beta_pt, out_pt);
        printf("InstanceNorm: num_tensors=%d, num_ops=%d\n", num_tensors, num_ops);
        printf("InstanceNorm: last 3 ops: %d, %d, %d\n",
            num_tensors > 3 ? ctx->ops->ops[num_ops-3].op_code : -1,
            num_tensors > 4 ? ctx->ops->ops[num_ops-2].op_code : -1,
            num_tensors > 5 ? ctx->ops->ops[num_ops-1].op_code : -1);
    }
    #endif
#if NUM_CORES > 1
    if(C >= S){
        int block_c = (C + NUM_CORES - 1) / NUM_CORES;
        start_c = pi_core_id() * block_c;
        stop_c = start_c + block_c; if(stop_c > C) stop_c = C;
    } else {
        int block_s = (S + NUM_CORES - 1) / NUM_CORES;
        start_s = pi_core_id() * block_s;
        stop_s = start_s + block_s; if(stop_s > S) stop_s = S;
    }
#endif

    for(int c = start_c; c < stop_c; ++c){
        float mean_val = mean_pt[c];
        float var_val  = var_pt [c];
        #ifdef ODL_USE_UNSAFE_FAST_MATH
        float inv_std = odl_fast_rsqrt(var_val + eps);
        #else
        float inv_std = 1.0f / __builtin_pulp_f32sqrt(var_val + eps);
        #endif
        float g = gamma_pt ? gamma_pt[c] : 1.0f;
        float b = beta_pt  ? beta_pt[c] : 0.0f;
        int base = c * S; // since B==1
        for(int s = start_s; s < stop_s; ++s){
            float norm = (x_pt[base + s] - mean_val) * inv_std;
            out_pt[base + s] = norm * g + b;
        }
    }
}

// Backward instance normalization tail
// Tensors ordering (example names):
// 0: input_0 (dY * gamma) or dY (shape [B,F])
// 1: input_1 (variance, shape [B,1])
// 2: input_2 (x_hat, shape [B,F])
// 3: input_3 (sum(dY*gamma*x_hat) across feature dim, shape [B,1])
// 4: add_arg_1 (epsilon) scalar tensor (optional)
// 5: divide_arg_0 (1.0) scalar (optional)
// 6: multiply_3_arg_0 (-0.5) scalar (optional)
// 7: multiply_5_arg_1 (1/F) scalar (optional)
// 8: multiply_7_arg_1 (2.0) scalar (optional)
// last: output tensor (shape [B,F])
void odl_bw_instance_norm_tail_fp32(void* args){
    MatchCtx* ctx = (MatchCtx*)args;
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int output_tensor_idx = num_tensors - 1;

    // Assume first 4 tensors are the required inputs
    float * __restrict__ dy_pt = (float*)tensors[0].pt;          // dY (already scaled by gamma if graph fused it)
    float * __restrict__ var_pt = (float*)tensors[1].pt;         // variance per instance (B,1)
    float * __restrict__ xhat_pt = (float*)tensors[2].pt;        // normalized activations (B,F)
    float * __restrict__ sum_dy_xhat_pt = (float*)tensors[3].pt; // sum(dY*gamma*x_hat) per instance (B,1)
    float * __restrict__ out_pt = (float*)tensors[output_tensor_idx].pt; // output (B,F)

    // Feature dims from first tensor tile (2D: [B,F])
    int B = tensors[0].tiles[L1_SCRATCHPAD*2+0].size;
    int F = tensors[0].tiles[L1_SCRATCHPAD*2+1].size;

    #ifdef ODL_DEBUG_KERNEL
    if(pi_core_id() == 0){
        printf("Instance Norm BW: B=%d, F=%d\n", B, F);
        printf("dy_pt: %p, var_pt: %p, xhat_pt: %p, sum_dy_xhat_pt: %p, out_pt: %p\n",
            dy_pt, var_pt, xhat_pt, sum_dy_xhat_pt, out_pt);
    }
    #endif
    // Optional constants (scalars) if provided
    float eps = 1e-5f;
    float neg_half = -0.5f;
    float inv_F_const = 1.0f / (float)tensors[0].tiles[1].size;
    float two = 2.0f;
    if(num_tensors > 4 && tensors[4].num_dims == 0 && tensors[4].pt) eps = *((float*)tensors[4].pt);
    if(num_tensors > 6 && tensors[6].num_dims == 0 && tensors[6].pt) neg_half = *((float*)tensors[6].pt);
    if(num_tensors > 7 && tensors[7].num_dims == 0 && tensors[7].pt){
        float v = *((float*)tensors[7].pt);
        if(v != 0.0f) inv_F_const = v; // expected 1/F
    }
    if(num_tensors > 8 && tensors[8].num_dims == 0 && tensors[8].pt) two = *((float*)tensors[8].pt);

    // Parallelize over batch dimension
    int start_b = 0, stop_b = B, start_f = 0, stop_f = F;
    #if NUM_CORES > 1
    if(B >= F && B >= NUM_CORES){
        int block = (B + NUM_CORES - 1) / NUM_CORES;
        start_b = pi_core_id() * block;
        stop_b = start_b + block; if(stop_b > B) stop_b = B;
    }
    else{
        int block = (F + NUM_CORES - 1) / NUM_CORES;
        start_f = pi_core_id() * block;
        stop_f = start_f + block;
    }
    #endif

    for(int b = start_b; b < stop_b; ++b){
        float var_val = var_pt[b]; // variance
        float sum_val = sum_dy_xhat_pt[b];
        #ifdef ODL_USE_UNSAFE_FAST_MATH
        float inv_std = odl_fast_rsqrt(var_val + eps);
        #else
        float inv_std = 1.0f / __builtin_pulp_f32sqrt(var_val + eps);
        #endif
        #ifdef ODL_USE_UNSAFE_FAST_MATH
        float inv_std3 = odl_faster_pow2(inv_std) * inv_std; // inv_std^3
        #else
        float inv_std3 = inv_std * inv_std * inv_std;
        #endif
        float term_scaled = neg_half * sum_val * inv_std3 * inv_F_const; // (-0.5 * sum * inv_std^3) * (1/F)
        int base = b * F;
        for(int f = start_f; f < stop_f; ++f){
            float dy_val = dy_pt[base + f];
            float xhat_val = xhat_pt[base + f];
            out_pt[base + f] = dy_val * inv_std + two * xhat_val * term_scaled;
        }
    }
}