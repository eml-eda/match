#include <pulp_cluster/cluster_dev.h>

struct pi_device cluster_dev = {0};
struct pi_cluster_task cluster_task = {0};

void pulp_open_init(){
    // set soc frequency
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_FC, 370000000);
    pi_time_wait_us(10000);
    // set periph frequency
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 370000000);
    pi_time_wait_us(10000);
}


void pulp_cluster_init() {

    #if NUM_CORES > 0
    struct pi_cluster_conf conf;
    // First open the cluster
    pi_cluster_conf_init(&conf);
    conf.id=0;
    pi_open_from_conf(&cluster_dev, &conf);
    if (pi_cluster_open(&cluster_dev))
        return;
    
    // set cluster frequency after cluster open
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, 370000000);
    pi_time_wait_us(10000);

    // set cluster task stack size
    #ifdef GAP_SDK
    #ifndef TARGET_CHIP_FAMILY_GAP9
    cluster_task.stack_size = 3500;
    #endif
    cluster_task.slave_stack_size = 3400;
    #else
    cluster_task.stack_size = 4096;
    cluster_task.slave_stack_size = 3072;
    #endif
    #endif

}

void pulp_cluster_close() {
    #if NUM_CORES > 0
    // TODO: Do i need a reset?
    //cluster_dev = {0};
    //cluster_task = {0};
    pi_cluster_close(&cluster_dev);
    #endif
}
