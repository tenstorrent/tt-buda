#!/bin/bash

LOGFILE="autotune.log"

extract_commands() {
    grep "pybuda/test/benchmark/benchmark.py" "$1" | sed 's/ -o perf.json//'
}

# Read commands into an array
mapfile -t commands < <(extract_commands "pybuda/test/benchmark/run_benchmark")

for cmd in "${commands[@]}"; do
    # Extract the model name for the cache file
    model=$(echo "$cmd" | grep -oP "(?<=-m |--model )\w+")
    config=$(echo "$cmd" | grep -oP "(?<=-c |--config )\w+")

    # Reset the device
    /mnt/motor/syseng/bin/tt-smi/wh/stable -wr all wait

    # Autotune the model
    pybuda/pybuda/tools/autotune.py --cache ".cache/${model}_${config}.ttc" "$cmd"
done

# Dump the results
echo "Autotuning complete, see results in $LOGFILE"
python3 scripts/parse_autotune.py $LOGFILE
