## This is a Mako template for generating a dynamic runtime in C for a static LLM model with multiple inputs and outputs.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tvm_runtime.h"  // Include TVM runtime API

## Define inputs and outputs using dictionaries
## Example:
## inputs = {
##     "input_1": {'name': 'input1', 'dims': [13, 16, "seq_len"]},
##     "input_2": {'name': 'input2', 'dims': [3, 5]}
## }
## dynamic_dims = {
##     "seq_len": {"max": 128, "min": 1}
## }
## outputs = {
##     "output_1": {'name': 'output1', 'dims': [10]},
##     "output_2": {'name': 'output2', 'dims': [2, 4]}
## }
## model_sizes represents the available static models for different dimensions
## Example: model_sizes = [{'name': 'input1', 'sizes': [8, 16, 32]}, {'name': 'input2', 'sizes': [8, 16, 32]}]

#define MAX_DIMENSIONS 10  // Max number of dimensions per input/output
#define NUM_MODELS ${len(model_sizes[0]['sizes'])}

// Dynamic dimension placeholders
% for dim in dynamic_dims:
int ${dim}_value = ${dynamic_dims[dim]['min']}; // Initialize dynamic dimension
% endfor

// Function to find the closest static model index based on the current input sizes
int get_closest_model_idx(int* input_lengths) {
    for (int i = 0; i < NUM_MODELS; ++i) {
        int all_fits = 1;
        % for key, inp in inputs.items():
        if (${inp['name']}_sizes[i] < input_lengths[${key}]) all_fits = 0;
        % endfor
        if (all_fits) {
            return i;
        }
    }
    return NUM_MODELS - 1;  // Use the largest model if no match is found
}

// Function to pad input arrays to the nearest static model size
void pad_input(float* input, int input_dims[], int target_dims[]) {
    int total_input_size = 1;
    int total_target_size = 1;

    for (int i = 0; i < MAX_DIMENSIONS; i++) {
        total_input_size *= input_dims[i];
        total_target_size *= target_dims[i];
    }

    // Pad the input with zeros if necessary
    for (int i = total_input_size; i < total_target_size; ++i) {
        input[i] = 0.0f;  // Padding with 0
    }
}

// Function to load and run the correct static model based on the input/output dimensions
void run_model(
% for key, inp in inputs.items():
    float* ${inp['name']}, int ${inp['name']}_dims[MAX_DIMENSIONS]
    % if not loop.last:
    ,
    % endif
% endfor
% for key, out in outputs.items():
    float* ${out['name']}, int ${out['name']}_dims[MAX_DIMENSIONS]
    % if not loop.last:
    ,
    % endif
% endfor
    int* input_lengths) {

    // Choose the appropriate model
    int model_idx = get_closest_model_idx(input_lengths);
    printf("Running model with dimensions:\n");
    % for key, inp in inputs.items():
    printf("    Input ${inp['name']}: %d (target: %d)\n", input_lengths[${key}], ${inp['name']}_sizes[model_idx]);
    % endfor

    // Run the model (this is where you'd call tvmgen_default_run for the selected static model)
    tvmgen_default_run();  // Placeholder for the actual model call
}

// Main runtime logic
int main() {
    srand(time(0));  // Initialize random seed

    // Input sizes for each of the inputs
    int input_lengths[] = {${', '.join(['0' for _ in inputs])}};  // Initialize input lengths

    // Allocate input and output buffers
    % for key, inp in inputs.items():
    float* ${inp['name']} = (float*)malloc(${inp['name']}_max_dims[0] * ${inp['name']}_max_dims[1] * sizeof(float));  // Adjust based on max dimensions
    % endfor
    % for key, out in outputs.items():
    float* ${out['name']} = (float*)malloc(${out['name']}_max_dims[0] * ${out['name']}_max_dims[1] * sizeof(float));  // Adjust based on max dimensions
    % endfor

    // Initialize random input values for each input
    % for key, inp in inputs.items():
    for (int i = 0; i < input_lengths[${key}]; ++i) {
        ${inp['name']}[i] = (float)rand() / RAND_MAX;
    }
    % endfor

    // Simulation of generation process (e.g., LLM token generation)
    int generation_done = 0;
    while (!generation_done) {
        // Pad each input to match the nearest static model size
        int model_idx = get_closest_model_idx(input_lengths);

        // Pad inputs to match the nearest model size
        % for key, inp in inputs.items():
        pad_input(${inp['name']}, input_lengths, ${inp['name']}_sizes[model_idx]);
        % endfor

        // Call the model with the current inputs and outputs
        run_model(
        % for key, inp in inputs.items():
            ${inp['name']}, ${inp['name']}_dims
            % if not loop.last:
            ,
            % endif
        % endfor
        % for key, out in outputs.items():
            ${out['name']}, ${out['name']}_dims
            % if not loop.last:
            ,
            % endif
        % endfor
        input_lengths);

        // Simulate updating dynamic input sizes (e.g., sequence length or token count)
        % for key, inp in inputs.items():
        input_lengths[${key}] += 1;  // Increment the input length for dynamic dimensions
        if (input_lengths[${key}] > ${dynamic_dims[inp['dims'][-1]]['max']}) {
            input_lengths[${key}] = ${dynamic_dims[inp['dims'][-1]]['min']};  // Reset to minimum if max exceeded
        }
        % endfor

        // Check if generation is complete (in this case, limit to 10 iterations)
        static int step_count = 0;
        if (++step_count >= 10) {
            generation_done = 1;
        }
    }

    // Free allocated memory
    % for key, inp in inputs.items():
    free(${inp['name']});
    % endfor
    % for key, out in outputs.items():
    free(${out['name']});
    % endfor

    return 0;
}
