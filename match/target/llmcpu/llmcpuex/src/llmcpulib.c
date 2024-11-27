#include <llmcpulib.h>

void* llm_matmul(match_kernel* kernel){
    float* inp_pt = (float*)kernel->common_kernel->I_pt;
    float* out_pt = (float*)kernel->common_kernel->I_pt;
    float* weights_pt = (float*)kernel->common_kernel->I_pt;
    int c_w = kernel->common_kernel->c_w;
    for(int k=0;k<kernel->common_kernel->k_o;k++){
        out_pt[k]=0;
        for(int c=0;c<kernel->common_kernel->c_i;c++)
            out_pt[k]+=inp_pt[c]*weights_pt[k*c_w+c];
    }
}