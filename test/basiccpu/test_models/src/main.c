#include <match_default_inputs.h>
#include <match_runtime.h>

// target specific inlcudes

int main(int argc,char** argv){
    // target specific inits
    
    match_runtime_ctx match_ctx;


    match_default_runtime(
        &match_ctx);
    
    // target specific cleaning functions
    return 0;
}