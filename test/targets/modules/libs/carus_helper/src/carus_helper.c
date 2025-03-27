#include <carus_helper/carus_helper.h>

void mnist_handle_output(int* output_pt, match_runtime_ctx* runtime_ctx){
    int max_idx = 0;
    int max_val = output_pt[0];
    for(int i=1; i<10; i++){
        if(output_pt[i]>max_val){
            max_val = output_pt[i];
            max_idx = i;
        }
    }
    print_big_number(max_idx);
}

void carus_helper_init_l1_mem(){
    l1_hal_init(0, l1_loader, sizeof(l1_loader), 1);
}

void carus_compute_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int out_chs = tensors[num_tensors-1].tiles[1].size;
    int inp_chs = tensors[0].tiles[1].size;
    // load into l1 memory activations
    xmr(m0, tensors[0].base_pt, 1, inp_chs, 1, 1);
    // load into l1 memory weights stored in CN format
    xmr(m1, tensors[1].base_pt, inp_chs, out_chs, 1, 1);
    // load into l1 memory outputs
    xmr(m2, tensors[num_tensors-1].base_pt, 1, out_chs, 1, 1);
    // dense op
    carus_mmul_tiling(m2, m0, m1, mNONE, 0, 0);
}
