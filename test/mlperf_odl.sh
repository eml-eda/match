#!/usr/bin/env bash

# Function to run ODL tests for a specific model, target, and direction
run_odl_test() {
    local model=$1
    local target=$2
    local direction=$3
    local target_dir=$4
    local base_dir=$5

    echo "Running ODL test for model: $model, target: $target, direction: $direction"
    # run the odl test script with the specified parameters
    python3 test_odl.py --target $target --relay_model models/odl/$model/model_graph_$direction.relay --relay_params_filename models/odl/$model/model_params.txt --executor graph
    # create the directory structure if it doesn't exist
    mkdir builds/$base_dir
    mkdir builds/$base_dir/$model
    mkdir builds/$base_dir/$model/$direction
    mkdir builds/$base_dir/$model/$direction/$target_dir
    # move the last build to the target directory
    rm builds/$base_dir/$model/$direction/$target_dir -r 2>/dev/null || true
    mv builds/last_build builds/$base_dir/$model/$direction/$target_dir
}

# Define test parameters
models=("kws" "vww" "imcls")
targets=("pulp_open" "GAP9")
directions=("fw" "bw")
NAME_BASE_DIR="conv_accelerated"
# Map target names to directory names
declare -A target_dirs
target_dirs["pulp_open"]="popen"
target_dirs["GAP9"]="gap"

# Run tests for all combinations
for model in "${models[@]}"; do
    echo "# ${model^^} TESTS"
    for target in "${targets[@]}"; do
        echo "# ${target^^}"
        for direction in "${directions[@]}"; do
            run_odl_test "$model" "$target" "$direction" "${target_dirs[$target]}" "$NAME_BASE_DIR"
        done
    done
done