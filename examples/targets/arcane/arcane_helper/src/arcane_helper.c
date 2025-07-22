#include <arcane_helper/arcane_helper.h>

/*
void mnist_handle_output(int* output_pt, int classes, int runtime_status){
    int max_idx = 0;
    int max_val = output_pt[0];
    for(int i=1; i<classes; i++){
        if(output_pt[i]>max_val){
            max_val = output_pt[i];
            max_idx = i;
        }
    }
    print_big_number(max_idx);
}
*/

void arcane_helper_init_l1_mem(){
    l1_hal_init(0, l1_loader, sizeof(l1_loader), 1);
}

void arcane_compute_wrapper(MatchCtx* ctx){
    MatchTensor* tensors = ctx->tensors->tensors;
    int num_tensors = ctx->tensors->num_tensors;
    int out_chs = tensors[num_tensors-1].tiles[1].size;
    int inp_chs = tensors[0].tiles[1].size;
    // reserve matrix in the ARCANE NMC module for the activations
    xmr(m0, tensors[0].pt, 1, inp_chs, 1, 1);
    // reserve matrix in the ARCANE NMC module for weights stored in CN format
    xmr(m1, tensors[1].pt, inp_chs, out_chs, 1, 1);
    // reserve matrix in the ARCANE NMC module for the outputs
    xmr(m2, tensors[num_tensors-1].pt, 1, out_chs, 1, 1);
    // dense op
    carus_mmul_tiling(m2, m0, m1, mNONE, 0, 0);
}
