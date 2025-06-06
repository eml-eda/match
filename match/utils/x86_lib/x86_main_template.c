#include <stdio.h>
#include <stdint.h>
#include "tvmgen_default.h"
#include <tvm_runtime.h>

int abs(int v) {return v * ((v > 0) - (v < 0)); }

int main(int argc, char** argv) {
  // TODO: print otherwise malloc is corrupted
  printf("\tMatch x86: inputs filled with ones\n");

  uint32_t output_size = ${match_output["size"]};
  uint8_t *output = (uint8_t*)malloc(output_size * sizeof(uint8_t));
  struct tvmgen_default_outputs outputs = { .output = output, };
  
  % for match_input in match_inputs:
  uint32_t ${match_input["name"]}_size = ${match_input["size"]};
  uint8_t *${match_input["name"]} = (uint8_t*)malloc(${match_input["name"]}_size * sizeof(uint8_t));
  // Fill input with 1
  for (uint32_t i = 0; i < ${match_input["name"]}_size; i++){
    ${match_input["name"]}[i] = 1;
  }
  % endfor

  struct tvmgen_default_inputs inputs = {
    % for idx,match_input in enumerate(match_inputs):
    % if idx>0:
    ,
    % endif
    .${match_input["name"]} = ${match_input["name"]}
    % endfor
  };

  int32_t status = 0;
  status = tvmgen_default_run(&inputs, &outputs);
  printf("\n{\"output\":[");
  for(int k=0;k<output_size;k++) {printf("%d",(uint8_t)output[k]);if(k!=output_size-1) printf(", ");}
  printf("]}\n");
  // TODO: free causes SEG FAULT
  % for match_input in match_inputs:
  //free(${match_input["name"]});
  % endfor
  //free(output);
  if(status != 0){
    abort();
  }
  return 0;
}
