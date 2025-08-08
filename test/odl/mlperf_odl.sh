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
    cd ..
    python3 test_odl.py --target $target --relay_model models/odl/$model/model_graph_$direction.relay --relay_params_filename models/odl/$model/model_params.txt --executor graph
    # create the directory structure if it doesn't exist
    mkdir builds/$base_dir
    mkdir builds/$base_dir/$model
    mkdir builds/$base_dir/$model/$direction
    mkdir builds/$base_dir/$model/$direction/$target_dir
    # move the last build to the target directory
    rm builds/$base_dir/$model/$direction/$target_dir -r 2>/dev/null || true
    mv builds/last_build builds/$base_dir/$model/$direction/$target_dir
    cd odl
}

execute_gap_test() {
    local model=$1
    local target=$2
    local direction=$3
    local target_dir=$4
    local base_dir=$5

    echo "Executing GAP test for model: $model, target: $target, direction: $direction"
    cd ../builds/$base_dir/$model/$direction/$target_dir
    echo -e "\033[34mCompiling...\033[0m"
    make all -j16 >> compilation.log 2>&1
    echo -e "\033[33mRunning...\033[0m"
    make run >> run.log 2>&1
    make clean
    rm BUILD -r
    cd ../../../../..
    cd odl
}

execute_pulp_open_test() {
    local model=$1
    local target=$2
    local direction=$3
    local target_dir=$4
    local base_dir=$5

    echo "Executing GAP test for model: $model, target: $target, direction: $direction"
    cd ../builds/$base_dir/$model/$direction/$target_dir
    echo -e "\033[34mCompiling...\033[0m"
    make all >> compilation.log 2>&1
    echo -e "\033[33mRunning...\033[0m"
    make run >> run.log 2>&1
    make clean
    rm BUILD -r
    cd ../../../../..
    cd odl
}
# Define test parameters
models=("kws" "vww" "imcls")
targets=("GAP9")
directions=("bw")
NAME_BASE_DIR="convbw_test"
# Map target names to directory names
declare -A target_dirs
target_dirs["pulp_open"]="popen"
target_dirs["GAP9"]="gap"
execute_func["pulp_open"]="execute_pulp_open_test"
execute_func["GAP9"]="execute_gap_test"

# Run tests for all combinations
for model in "${models[@]}"; do
    echo "# ${model^^} TESTS"
    for target in "${targets[@]}"; do
        echo "# ${target^^}"
        for direction in "${directions[@]}"; do
            run_odl_test "$model" "$target" "$direction" "${target_dirs[$target]}" "$NAME_BASE_DIR"
            ${execute_func[$target]} "$model" "$target" "$direction" "${target_dirs[$target]}" "$NAME_BASE_DIR"
        done
    done
done