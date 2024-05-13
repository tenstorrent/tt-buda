#!/bin/bash

: << 'COMMENT'
SAMPLE:
Script run command : bash ./scripts/bisect.sh

INPUTS:
Enter Pytest Command: 
pytest --devtype golden pybuda/test/model_demos/high_prio/cnn/pytorch/test_xception.py::test_xception_timm[Golden-xception] --device-config gs_e150
Enter Passing Commit Id: 
8e576abe7fdc250ba88775322b448fa05acf52d1 #passing commit id
Enter Failing Commit Id:
6c2a0f68aab744ce08174f5c59abc946be6b8395 #failing commit id
Enter Architecture(grayskull/wormhole_b0): 
grayskull
Device config(e150/e300):
e150
Enter Run type(compile/silicon):
compile

COMMENT

# Enabling required flags based on the architecture and run type 
set_evn_flags() {
    local arch=$1
    local runtype=$2
    local device_config=$3
    export PYBUDA_VERIFY_POST_AUTOGRAD_PASSES=1
    export PYBUDA_VERIFY_POST_PLACER=1
    export PYBUDA_VERIFY_NET2PIPE=3
    export PYTEST_ADDOPTS=" -svv"

    if [ "$arch" = "wormhole_b0" ] ; then
        export BACKEND_ARCH_NAME=wormhole_b0
        export ARCH_NAME=wormhole_b0

        if [ "$device_config" = "no" ] ; then
            export PYBUDA_FORCE_EMULATE_HARVESTED=1 
        fi
        
        if [ "$runtype" = "compile" ] ; then
            export GOLDEN_WORMHOLE_B0=1 
            export PYBUDA_DEVMODE=1 
            export PYBUDA_EMULATE_SILICON_DEVICE=1 
            export PYBUDA_VERIFY_GOLDEN=1
        else
            export PYTEST_ADDOPTS=" -svv --silicon-only"
        fi
    fi

    if [ "$arch" = "grayskull" ] ; then
        export BACKEND_ARCH_NAME=grayskull
        export ARCH_NAME=grayskull

        if [ "$device_config" = "e300" ] ; then
            export PYBUDA_FORCE_EMULATE_HARVESTED=1 
        fi

        if [ "$runtype" = "compile" ] ; then
            export PYBUDA_DEVMODE=1 
            export PYBUDA_EMULATE_SILICON_DEVICE=1 
            export PYBUDA_VERIFY_GOLDEN=1
        else
            export PYTEST_ADDOPTS=" -svv --silicon-only"
        fi

    fi
}

# Getting inputs from the user
get_inputs() {
    local pytest_cmd
    read -p "Enter Pytest Command: " pytest_cmd 
    read -p "Enter Passing Commit Id: " pass_id 
    read -p "Enter Failing Commit Id: " fail_id 
    read -p "Enter Architecture(grayskull/wormhole_b0): " arch
    
    if [ "$arch" = "wormhole_b0" ] ; then
        read -p "Is it 1x1 config(yes/no): " device_config
    else
        read -p "Device config(e150/e300): " device_config
    fi
    read -p "Enter Run type(compile/silicon): " runtype

    echo "$pytest_cmd,$pass_id,$fail_id,$arch,$runtype,$device_config"
}

# If any build issues, it will show build error and exit
error_handling() {
    if [ $? -ne 0 ]; then
        local stage="$2"
        echo "Error: $stage  Command failed"
        exit 1
    fi
}

# Clean previous all cacche and build folder. Build based on the architecture
# Input :
# $1: Architecture (based on that build will be run)
env_clean_and_build() {
    local arch="$1"
    git submodule update --init --recursive >/dev/null 2>&1
    git submodule update --init --checkout --depth 1 -f third_party/confidential_customer_models >/dev/null 2>&1
    echo "Submodules Updated"
    if [ -d "build" ]; then
        echo "Build directory exists. Doing a clean up..."
        rm -rf .pkl_memoize_py3 
        rm -rf .pytest_cache 
        rm -rf device_images/ 
        rm -rf .hlkc_cache  
        rm -rf wheel_out/  
        rm -rf wheel_env/  
        rm -rf pybuda.egg-info/ 
        rm -rf wheele_env/ 
        rm -rf generated_modules 
        rm -rf tt_build 
        rm -rf net2pipe_output 
        rm -rf tensor_binaries 
        rm -rf imagenet_classes*
        rm -rf core*
        rm -rf *.log 
        rm -rf *.summary 
        rm -rf *.yaml 
        rm -rf *.png 
        rm -rf *.jpg 
        rm -rf *.pt 
        rm -rf *.xml
        rm -rf *.json
        make clean >/dev/null 2>&1
        error_handling "$?" "Clean"
        echo "Build and cache is cleaned!"
    fi
    
    # Disable this code if your testing old regression
    if [ "$arch" = "wormhole_b0" ] ; then 
        source env_for_wormhole_b0.sh >/dev/null 2>&1
    else 
        source env_for_silicon.sh >/dev/null 2>&1
    fi
    error_handling "$?" "Build"

    #Enable below code for old regression

    #export TORCH_VISION_INSTALL=0 
    #if [ "$arch" = "wormhole_b0" ] ; then 
    #    source env_for_wormhole_b0.sh >/dev/null 2>&1
    #else 
    #    source env_for_silicon.sh >/dev/null 2>&1
    #fi
    #error_handling "$?" "Build"
    #if [ -d "vision" ]; then
    #    echo "Vision Directory exists. Doing a clean up..."
    #    rm -rf vision
    #fi
    #export TORCH_VISION_INSTALL=1
    #make torchvision >/dev/null 2>&1
    #error_handling "$?" "Torchvision"
    
    echo "Build Successfully"

}

# Runs the pytest command and stores all logs in log path
# Input :
# $1: Pytest Command
# $2: Log path
# Output:
#   Last line of the pytest result
pytest_run() {
    local cmd="$1" 
    local log_path=$2
    command="$cmd >$log_path"
    eval "$command" >/dev/null 2>&1
    result=$(tail -1 "$log_path") 
    echo "$result"
}

#Based on the pytest result, it will bisect good or bad
# Input :
# $1: pytest results
# $2: Expected string to be replicated
# Output:
#   Current Test case is pass or failed
#   First line of the bisect output
comparison_result() {
    local pytest_result=$1
    local expecting_string=$2
    local replication=1
    local bis_out

    if [ "$expecting_string" = "NA" ]; then
        replication=0
        expecting_string="passed"
    fi

    if echo "$pytest_result" | grep -q "skipped" ; then
        echo "============================= Testcase got skipped ============================="
        exit 1
    fi

    if echo "$pytest_result" | grep -q "$expecting_string" ; then
        if [ "$replication" -eq 0 ] ; then
            echo "============================= Test case got $expecting_string ============================="
        else 
            echo "============================= $expecting_string case got replicated =============================" 
        fi
    else
        if [ "$replication" -eq 0 ] ; then
            echo "============================= Test case got failed ============================="
            expecting_string="failed" 
        else 
            echo "============================= Not able to replicate $expecting_string case =============================" 
            exit 1
        fi
    fi

    echo "Bisect results" 
    if [ "$expecting_string" = "passed" ] ; then
        bis_out=$(git bisect good | head -n 1 )
    else 
        if [ "$expecting_string" = "failed" ] ; then
            bis_out=$(git bisect bad | head -n 1 )
        fi
    fi
    echo "$bis_out"
}

# This function calls env_clean_and_build function, pytest_run function and comparison_result function
# Input :
# $1: Expected string to be replicated
# $2: Architecture
# $3: Pytest Command
# $4: Log Path 
# $5: Run count
# Output:
#   First line of the bisect output
bisect_run() {
    replica_string=$1
    arch=$2
    pytest_command=$3
    local Log_path
    if [ "$replica_string" = "NA" ]; then
        run_count=$5
        extension="_$run_count.txt"
        Log_path="$4/revision$extension"
    else
        extension="_replication.txt"
        Log_path="$4/$replica_string$extension"
    fi

    env_clean_and_build "$arch"
    pytest_result=$(pytest_run "$pytest_command" "$Log_path")
    bisect_output=$(comparison_result "$pytest_result" "$replica_string")
    echo "$bisect_output"
    deactivate
}

########################### main #################################

#INPUTS
# get_inputs function get 6 inputs from user and returns 4 outputs
# Parameters:
#   $1: Pytest Command
#   $2: Passing Commit Id
#   $3: Failing Commit Id
#   $4: Architecture
#   $5: Device config
#   $6: Run type
# Returns:
#   pytest_command, pass_id, fail_id, arch, runtype, device_config

inputs=$(get_inputs)
IFS=',' read -r pytest_command pass_id fail_id arch runtype device_config <<< "$inputs"

# set_evn_flags function is to set all environmental flags based on the architecture, runtype and device config
# Parameters:
#   $1: architecture
#   $2: runtype
#   $3: device config
set_evn_flags "$arch" "$runtype" "$device_config"
run_count=0

#Creating folder for dumping the logs
file_path=$(echo "$pytest_command" | cut -d'.' -f1)
model_name=$(echo "$file_path" | awk -F'/' '{print $NF}')
if ! [ -d "Logs" ]; then
    mkdir "Logs"
fi
folder_path="Logs/$model_name"
if ! [ -d "$folder_path" ]; then
    mkdir "$folder_path"
else
    echo "Log Directory exists. Doing a clean up and creating new one..."
    rm -rf "$folder_path"
    mkdir "$folder_path"
fi

#To Avoid clash with previous bisect run we are resetting and starting.
git bisect reset >/dev/null 2>&1
git bisect start

# bisect_run function get 3 inputs from user and returns result
# $1: Expected string to be replicated
# $2: Architecture
# $3: Pytest Command
# $4: Folder Path
# $5: Run count
# Returns:
# bisect result

#Replicating Pipeline last passing commit id in local run
echo -e "\nGoing to replicate pass case in last passing commit id..."
git checkout $pass_id >/dev/null 2>&1
bisect_run "passed" "$arch" "$pytest_command" "$folder_path" "$run_count"

#Replicating Pipeline first failing commit id in local run
echo "Going to replicate fail case in first failing commit id..."
git checkout "$fail_id" >/dev/null 2>&1
bisect_output=$(bisect_run "failed" "$arch" "$pytest_command" "$folder_path" "$run_count")
echo "$bisect_output"
result=$(echo "$bisect_output" | awk '/Bisect results/{p=1; next} p') 

#This loop will be continued untill we are getting first regressed commit id
while ! echo "$result" | grep -q "first bad commit"; do
    run_count=$((run_count+1))
    bisect_output=$(bisect_run "NA" "$arch" "$pytest_command" "$folder_path" "$run_count")
    echo "$bisect_output"
    result=$(echo "$bisect_output" | awk '/Bisect results/{p=1; next} p') 
    sleep 1 
done

extension="/bisect_log.txt"
git bisect log | tee "$folder_path$extension"