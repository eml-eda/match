
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <match_runtime.h>
#include <tvm_runtime.h>  // Include TVM runtime API
#include <match_default_inputs.h>
#include <time.h>

int __CPU_ONLY_ACTIVE__ = 0;

int check_default_differences_with_cpu(struct tvmgen_cpuonly_inputs* cpu_inps,struct tvmgen_cpuonly_outputs* cpu_outs,
                        struct tvmgen_default_inputs* model_inps,struct tvmgen_default_outputs* model_outs){
    tvmgen_cpuonly_run(cpu_inps,cpu_outs);tvmgen_default_run(model_inps,model_outs);
    int diffs = 0;
    for(int h_idx=0;h_idx<16;h_idx++)
        for(int w_idx=0;w_idx<16;w_idx++)
            if(((int*)cpu_outs->output)[h_idx*16+w_idx]!=((int*)model_outs->output)[h_idx*16+w_idx]){
                printf("CPU and Default outputs DO NOT match at h_idx %d w_idx %d CPU: %d default: %d diff: %d\n"
                    ,h_idx,w_idx,((int*)cpu_outs->output)[h_idx*16+w_idx],((int*)model_outs->output)[h_idx*16+w_idx]
                    ,((int*)cpu_outs->output)[h_idx*16+w_idx]-((int*)model_outs->output)[h_idx*16+w_idx]
                );
                diffs++;
            }
    return diffs;
}

void benchmark_cpu_model(int iterations, struct tvmgen_cpuonly_inputs* cpu_inps, struct tvmgen_cpuonly_outputs* cpu_outs){
    int status = 0;
    int fails = 0;
    clock_t start, end;
    start = clock();
    for(int i=0;i<iterations;i++){
        status=tvmgen_cpuonly_run(cpu_inps,cpu_outs);
        if(status) fails++;
    }
    end = clock();

    double time_elapsed_ms = ((double)(end - start))/CLOCKS_PER_SEC * 1000;
    printf("[CPU_BENCH] time %fms; time per iterations %fms; fails %d\n",
        time_elapsed_ms, time_elapsed_ms/iterations, fails);
}

void benchmark_default_model(
    int iterations,
    struct tvmgen_default_inputs* model_inps,
    struct tvmgen_default_outputs* model_outs){
    int status = 0;
    int fails = 0;
    clock_t start, end;
    
    start = clock();
    for(int i=0;i<iterations;i++){
        status=tvmgen_default_run(model_inps,model_outs);
        if(status) fails++;
    }
    end = clock();

    double time_elapsed_ms = ((double)(end - start))/CLOCKS_PER_SEC * 1000;
    printf("[DEFAULT_BENCH] time %fms; time per iteration %fms; fails %d\n",
        time_elapsed_ms, time_elapsed_ms/iterations, fails);
}
void match_default_runtime(
    match_runtime_ctx* match_ctx){
    int* default_output_pt = (int*)malloc(16*16*sizeof(int));
    int* cpu_output_pt = (int*)malloc(16*16*sizeof(int));
    struct tvmgen_cpuonly_inputs cpu_inps = {
        .input_0 = input_0_default,
    };
    struct tvmgen_cpuonly_outputs cpu_outs = {
        .output = cpu_output_pt,
    };
    struct tvmgen_default_inputs model_inps = {
        .input_0 = input_0_default,
    };
    struct tvmgen_default_outputs model_outs = {
        .output = default_output_pt,
    };
    // check if CPU matches default results
    int diffs = check_default_differences_with_cpu(&cpu_inps,&cpu_outs,&model_inps,&model_outs);
    if(diffs)   printf("Differences between CPU and default: %d\n",diffs);
    else    printf("CPU and Default outputs match\n");
    // Measure clock cycles
    int benchmark_iterations = 100000;
    benchmark_cpu_model(benchmark_iterations,&cpu_inps,&cpu_outs);
    benchmark_default_model(benchmark_iterations,&model_inps,&model_outs);

    // clean up
    free(default_output_pt);
    free(cpu_output_pt);
}