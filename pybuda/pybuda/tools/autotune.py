#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import getpass
import json
import os
import subprocess
import shutil
from loguru import logger

def parse_perf_json(perf_path, backup_cache_path):
    results = [["Run ID", "Samples/sec", "Cache file used"]]
    original_result = 0
    best_result = 0
    best_result_index = 0

    with open(perf_path, 'r') as file:
        data = json.load(file)
        original_result = data[0]['samples_per_sec']
        results.append(["Original", original_result, "n/a"])

        best_result = original_result
        num_loops = len(data)
        for i in range(1, num_loops):
            samples_per_sec = data[i]['samples_per_sec']
            if samples_per_sec > best_result:
                best_result = samples_per_sec
                best_result_index = i
            result = ["Iteration " + str(i), samples_per_sec, f"{backup_cache_path}{i-1}"]
            results.append(result)
    return results, original_result, best_result, best_result_index

def get_summary_table(results):
    # Function to format a row with pipe separators
    def format_row(row, widths):
        return '| ' + ' | '.join(val.ljust(width) for val, width in zip(row, widths)) + ' |'

    # Convert all data to strings and find max width of cols
    str_data = [[str(item) for item in row] for row in results]
    widths = [max(map(len, column)) for column in zip(*str_data)]

    # Print in markdown format - header, divider, contents
    table = ""
    table += format_row(results[0], widths) + "\n"
    table += '|' + '|'.join(['-' * (width + 2) for width in widths]) + "|\n"
    for row in str_data[1:]:
        table += format_row(row, widths) + "\n"

    return table

def main(command, num_loops, cache_path, perf_path):
    logger.info("Autotune started for command: " + args.command)

    backup_user_path = f"/tmp/{getpass.getuser()}"
    if not os.path.exists(backup_user_path):
        os.makedirs(backup_user_path)

    backup_cache_path = f"{backup_user_path}/{os.path.basename(cache_path)}.bak"

    env_vars = os.environ.copy()  # Copy the current environment
    env_vars["LOGGER_LEVEL"] = "None"
    env_vars["PYBUDA_COMPILER_CACHE"] = cache_path

    # Run the original command
    subprocess.run(command + " -o " + perf_path, shell=True, env=env_vars)

    # Run the autotuning loop
    for i in range(num_loops):
        subprocess.run(command + " --perf_analysis --loop_count 1", shell=True, env=env_vars)
        subprocess.run(f"pybuda/pybuda/tools/perf_analysis.py --cache {cache_path}", shell=True, env=env_vars)
        subprocess.run(command + " -o " + perf_path, shell=True, env=env_vars)

        # Make a backup of the cache file at the end of each loop
        shutil.copyfile(cache_path, f"{backup_cache_path}{i}")

    # Parse the output perf file
    results, original_result, best_result, best_result_index = parse_perf_json(perf_path, backup_cache_path)

    logger.info(f"Autotune Summary:\n{get_summary_table(results)}")
    logger.info(f"Autotune completed {num_loops} loops, best result = {best_result} samples/sec from iteration {best_result_index} (original = {original_result})")

    if best_result_index == 0:
        repro_command = command + " -o " + perf_path
    else:
        # cache used is the one generated from the previous iteration
        best_result_cache = f"{backup_cache_path}{best_result_index-1}"
        shutil.copyfile(best_result_cache, cache_path)
        # copy the snapshot from the best iteration to cache path for repro
        repro_command = f"PYBUDA_COMPILER_CACHE={cache_path} " + command + " -o " + perf_path

    logger.info(f"repro: {repro_command}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a command with performance analysis.')
    parser.add_argument('command', type=str, help='The command to run')
    parser.add_argument('-c', '--cache', type=str, required=True, help='Path for the cache file (eg: .cache/perf_cache.ttc)')
    parser.add_argument('-o', '--output', type=str, default='perf.json', help='Output json file to write results to, if it exists it will be overwritten')
    parser.add_argument('--loops', type=int, default=10, help='Number of loops to run (default: 10)')

    args = parser.parse_args()

    logger.add("autotune.log")

    if '-o ' in args.command or '--output ' in args.command:
        logger.error(f"Error: output file cannot be specified in <command> directly, move it outside of '{args.command}'")
        exit(1)

    # Remove cache and perf.json if they exist
    if os.path.exists(args.cache):
        os.remove(args.cache)
    if os.path.exists(args.output):
        os.remove(args.output)

    main(args.command, args.loops, args.cache, args.output)
