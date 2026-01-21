#include <pmsis.h>
#include <pulp_utils/pulp_rt_profiler_wrapper.h>


void  __attribute__((noinline, optimize("O0"))) init_global_perf_counter(){
}

void __attribute__((noinline, optimize("O0"))) start_perf_counter(){
    // perf is globally defined in pulp_rt_profiler_wrapper.h
    pi_perf_conf(1<<PI_PERF_CYCLES);
    pi_perf_reset();
    pi_perf_stop();
    pi_perf_start();
}

int32_t __attribute__((noinline, optimize("O0"))) stop_perf_counter(){
    // perf is globally defined in pulp_rt_profiler_wrapper.h
    pi_perf_stop();
    int32_t perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
    pi_perf_reset();
    return perf_cyc;
}

// General one
int32_t __attribute__((noinline, optimize("O0"))) start_match_perf_counter(){
    pi_perf_reset();
    pi_perf_conf(1<<PI_PERF_CYCLES);
    pi_perf_reset();
    pi_perf_stop();
    pi_perf_start();
    return 0;
}

int32_t __attribute__((noinline, optimize("O0"))) stop_match_perf_counter(){
    pi_perf_stop();
    int32_t perf_cyc = pi_perf_read(PI_PERF_CYCLES);
    pi_perf_reset();
    accumulate_perf+=perf_cyc;
    return perf_cyc;
}


int32_t __attribute__((noinline, optimize("O0"))) get_match_acc_perf_counter(){
    return accumulate_perf;
}

int32_t __attribute__((noinline, optimize("O0"))) reset_match_perf_counter(){
    accumulate_perf = 0;
    return 0;
}