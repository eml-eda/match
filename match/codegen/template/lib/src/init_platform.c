void match_init_platform(void (*inner_function)(unsigned int* args_inner_function),unsigned int* args){
    // API to offload execution to a controller
    inner_function(args);
}

void match_smth(){}