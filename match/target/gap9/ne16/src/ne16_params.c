#include <ne16_params.h>

void match_ne16_set_nnx_dev(){
    nnx_dev = ne16_pulp_get_dev();
}

ne16_dev_t* match_ne16_get_nnx_dev(){
    return nnx_dev;
}

ne16_task_t* match_ne16_get_nnx_task(int n){
    return &nnx_tasks[n];
}

unsigned int get_nnx_db_O(int n){
    return nnx_db_O[n];
}

unsigned int get_nnx_db_I(int n){
    return nnx_db_I[n];
}

unsigned int get_nnx_db_W(int n){
    return nnx_db_W[n];
}

void inc_nnx_db_O(int n){
    nnx_db_O[n]=(nnx_db_O[n]+1)%DB_BUFFER_SIZE;
}

void inc_nnx_db_I(int n){
    nnx_db_I[n]=(nnx_db_I[n]+1)%DB_BUFFER_SIZE;
}

void inc_nnx_db_W(int n){
    nnx_db_W[n]=(nnx_db_W[n]+1)%DB_BUFFER_SIZE;
}

nnx_monitor_t* get_nnx_monitor(){
    return &nnx_monitor;
}