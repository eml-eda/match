#include <match_model_gen_apis.h>

int token_id = 1;

void match_api_prepare_inputs_from_prev_out(
    % for idx,(inp_name,inp) in enumerate(match_inputs.items()):
    ${inp["c_type"]}* inp_pt_${idx},
    int* freed_inp_${idx},
    % endfor
    % for idx,(out_name,out) in enumerate(match_outputs.items()):
    ${out["c_type"]}* out_pt_${idx},
    % endfor
    int* generation_done,
    int prev_model_idx
){
    % if app=="textgen_logits_only":
    int batch_size=1;
    int sequence_length=16;
    int vocab_size=1000;
    int max_index=0;
    float max_logit=-1000;
    float* logits=out_pt_0;
    for (int b = 0; b < batch_size; b++) {
        int max_index = 0; // Index of the maximum logit
        int t=token_id-1;
        for (int v = 0; v < vocab_size; v++) {  // Start from the second token in the vocab
            float logit = logits[t * vocab_size + v];
            //if(!t) printf("v %d log %f\n",v,logit);
            if (logit > max_logit) {
                max_logit = logit;
                max_index = v;
            }
        }
    }
    printf("MATCH: Predicted token %d with confidence %f\n",max_index,max_logit);
    inp_pt_0[token_id]=max_index;
    token_id++;
    if(max_index==vocab_size-1 || token_id>=sequence_length)  *generation_done=1;
    % for idx in range(len(match_inputs)):
    *freed_inp_${idx}=0;
    % endfor
    % endif
    return;
}

void match_api_update_sizes(){
    return;
}