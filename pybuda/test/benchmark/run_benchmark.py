#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
This script is an attempt to improve the bash `run_benchmark` script. Currently, it only runs locally, but is soon
planned to run on [run_perf] tag in jenkins instead of the `run_benchmark` script.

It runs all the models listed in `run_benchmark` - it won't apply other commands from the script such as env vars,
removing perf.json, etc.

The board is reset before the script starts and after each hang. There are many potential improvements to be made here,
such as saving netlists for post-mortem analysis, parallelizing across multiple boards, controlling sets of models to
run (e.g. NLP vs CNN), etc.
"""

import datetime
import json
import os
import random
import re
import shlex
import string
import subprocess


# Timeouts in seconds
SHORT_TIMEOUT = 20  # 20 sec
LONG_TIMEOUT = 60 * 30  # 30 min
RESET_TIMEOUT = 60 * 2  # 2 min


def get_benchmark_file_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")


def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_git_root_path():
    # git rev-parse --show-toplevel
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("ascii").strip()


def get_git_commit_id():
    # git rev-parse --short=8 HEAD
    return subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"]).decode("ascii").strip()


def make_experiment_folder(debug=False):
    # if debug=True, create experiments_debug, avoid polluting results
    git_root_path = get_git_root_path()
    timestamp_string = get_current_timestamp()
    git_commit_id = get_git_commit_id()

    experiment_path = os.path.join(git_root_path, "experiments_debug" if debug else "experiments", f"{timestamp_string}_{git_commit_id}")
    os.makedirs(experiment_path)

    return experiment_path


def decode_text(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub("", text.decode("utf-8"))


def get_blob_ruby_commands_and_outputs(text):
    ruby_commands = []
    ruby_outputs = []
    lines = text.split("\n")
    for line in lines:
        if line.find("Runtime | FATAL") > -1:
            if line.find("Running blob command failed") > -1:
                ruby_command = line[line.find("ruby"):]
                ruby_commands.append(ruby_command)

    for ruby_command in ruby_commands:
        ruby_outputs.append(decode_text(subprocess.run(ruby_command.split(), capture_output=True, timeout=SHORT_TIMEOUT).stderr))

    assert len(ruby_commands) == len(ruby_outputs)
    return list(zip(ruby_commands, ruby_outputs))


def get_net2pipe_commands_and_outputs(text):
    net2pipe_commands = []
    net2pipe_outputs = []
    lines = text.split("\n")
    for line in lines:
        if line.find("Runtime | FATAL") > -1:
            if line.find("Running net2pipe command failed") > -1:
                net2pipe_command = line[line.find("Running net2pipe command failed: ") + len("Running net2pipe command failed: "):]
                net2pipe_commands.append(net2pipe_command)

    for net2pipe_command in net2pipe_commands:
        net2pipe_outputs.append(decode_text(subprocess.run(net2pipe_command.split(), capture_output=True, timeout=SHORT_TIMEOUT).stderr))

    assert len(net2pipe_commands) == len(net2pipe_outputs)
    return list(zip(net2pipe_commands, net2pipe_outputs))


def check_for_blob_errors(run_object, log_path):
    commands_and_outputs = get_blob_ruby_commands_and_outputs(decode_text(run_object.stdout))

    with open(f"{log_path}_ruby_errors.txt", "w") as ruby_out:
        for idx, (command, output) in enumerate(commands_and_outputs):
            ruby_out.write(f"{idx}\n\n")
            ruby_out.write(f"Command: {command}\n\n")
            ruby_out.write(f"Output: {output}\n\n")

    return len(commands_and_outputs)


def check_for_net2pipe_errors(run_object, log_path):
    commands_and_outputs = get_net2pipe_commands_and_outputs(decode_text(run_object.stdout))

    with open(f"{log_path}_net2pipe_errors.txt", "w") as ruby_out:
        for idx, (command, output) in enumerate(commands_and_outputs):
            ruby_out.write(f"{idx}\n\n")
            ruby_out.write(f"Command: {command}\n\n")
            ruby_out.write(f"Output: {output}\n\n")

    return len(commands_and_outputs)


def get_benchmark_commands(benchmark_commands_file_path):
    abs_path = os.path.join(os.path.dirname(__file__), benchmark_commands_file_path)
    lines = open(abs_path, "r").readlines()
    benchmark_commands = [line.strip() for line in lines if line.startswith("pybuda/test/benchmark/benchmark.py")]
    return benchmark_commands


def reset_board():
    try:
        print(f"Resetting board...")
        run = subprocess.run(
                args="/home/software/syseng/wh/tt-smi -wr all wait -er".split(),
                capture_output=True,
                timeout=RESET_TIMEOUT,
            )
    except Exception as e:
        print(f"Warm reset failed: {e}\n")


def get_model_from_benchmark_command(command):
    args = shlex.split(command)
    try:
        idx = args.index("-m")
        return args[idx + 1]
    except Exception as e:
        print(f"Exception raised: {e}")
        return "".join(random.choices(string.ascii, k=10))


def get_config_from_benchmark_command(command):
    args = shlex.split(command)
    try:
        idx = args.index("-c")
        return args[idx + 1]
    except Exception as e:
        print(f"Exception raised: {e}")
        return "".join(random.choices(string.ascii, k=10))


def main():
    import shlex

    # os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # import argparse

    # parser = argparse.ArgumentParser(description='CNN Benchmark script parser')
    # parser.add_argument('-bp',  '--balancer-policy', choices=['default', 'CNN', 'Ribbon', 'NLP'], default='default', help='Set balancer policy.')

    # args = parser.parse_args()

    experiment_path = make_experiment_folder(debug=False)
    # experiment_path = make_experiment_folder(debug=True)

    all_perf_path = os.path.join(experiment_path, "perf.json")

    benchmark_commands = get_benchmark_commands("run_benchmark")
    # benchmark_commands = get_benchmark_commands("run_benchmark_debug")

    # Reset board before starting to ensure good state
    reset_board()

    # Loop tests
    for idx, command in enumerate(benchmark_commands):
        model = get_model_from_benchmark_command(command)
        config = get_config_from_benchmark_command(command)

        print(f"Running {idx + 1}/{len(benchmark_commands)}... Model: {model}, config: {config}")

        log_path = os.path.join(experiment_path, f"{model}_{config}")
        model_perf_path = f"{log_path}_perf.json"

        try:
            benchmark_args = shlex.split(command)  # doesn't split string within quotes (whereas str.split() does)
            benchmark_args = [*benchmark_args, "-o", model_perf_path]  # add perf output path

            with open(f"{log_path}_args.txt", "w") as out_args:
                out_args.write(f"{'_'.join(benchmark_args)}\n")

            run = subprocess.run(
                args=benchmark_args,
                capture_output=True,
                timeout=LONG_TIMEOUT,
            )

            # stdout
            with open(f"{log_path}_stdout.txt", "w") as out:
                out.write(decode_text(run.stdout))

            # stderr
            with open(f"{log_path}_stderr.txt", "w") as err:
                err.write(decode_text(run.stderr))

            with open(model_perf_path, "r") as perf_read:
                # Perf is already dumped by benchmark.py, collect it in one place for all models in current experiment
                run_output = perf_read.read()
                with open(all_perf_path, "a") as perf_write:
                    perf_write.write(f"Model: {model}, config: {config}\n")
                    perf_write.write(run_output)
                    perf_write.write("\n\n")

                run_output_as_dict = json.loads(run_output)

                if "samples_per_sec" in run_output_as_dict[0]:
                    print(f"  Perf: {round(run_output_as_dict[0]['samples_per_sec'], 2)} samples/s")

                # Check if error and process
                if "error" in run_output_as_dict[0]:
                    error_msg = run_output_as_dict[0]["error"]

                    # Check if "Backend compile failed" error
                    # If so, grab all commands that compile failed on and run them
                    if error_msg == "Backend compile failed":
                        num_errors = 0
                        num_errors += check_for_blob_errors(run, log_path)
                        num_errors += check_for_net2pipe_errors(run, log_path)

                        assert num_errors > 0  # TODO: turn into exception and handle properly

                    # Usually a hang
                    if error_msg == "Error raised, aborting benchmark":
                        if decode_text(run.stderr).count('raise RuntimeError("Timeout while reading " + outq.name)') > 0:
                            with open(f"{log_path}_hang.txt") as hang_out:
                                hang_out.write("hang :/")
                            print("  HANGED :/")
                            reset_board()

        except Exception as e:
            with open(f"{log_path}_exception.txt", "w") as exception_out:
                exception_out.write(str(e))

    try:
        import shutil
        shutil.copy2(all_perf_path, os.path.join(get_git_root_path(), "perf.json"))
    except Exception as e:
        print(f"Failed to copy perf.json: {e}")

    print("end.")


if __name__ == "__main__":
    main()
