# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Convenience script to run tests, reset on hangs and produce summary.

import os
import glob
import socket
import random
import string
import argparse
import itertools
import subprocess
from datetime import date

import pybuda
from pybuda import (
    BackendType,
    BackendDevice,
)

# Warm reset command for WH B0
tt_root_smi_path = "/home/software/syseng"
reset_command_gs = tt_root_smi_path + "/gs/tt-smi -tr all"
reset_command_wh_b0 = tt_root_smi_path + "/wh/tt-smi -lr all wait -er"

# High priority directory
high_prio_dir = "pybuda/test/model_demos/high_prio/"

# Override desired testlist (if not defined, it'll be automatically generated to run full test suite based on high_prio_dir)
testlist = [

]

# Test variants to ignore
testlist_to_ignore = [

]

# Globals
hostname = socket.gethostname().replace("-", "_")

def set_env_vars_to_match_ci(device_type):
    # General
    pytest_addopts = "-svv --durations=0"
    pytest_addopts += " --silicon-only" if device_type == BackendType.Silicon else " --no-silicon"
    os.environ["PYTEST_ADDOPTS"] = pytest_addopts

    # PyBuda
    os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"
    os.environ["PYBUDA_VERIFY_POST_AUTOGRAD_PASSES"] = "1"
    os.environ["PYBUDA_VERIFY_POST_PLACER"] = "1"
    os.environ["PYBUDA_VERIFY_NET2PIPE"] = "3"


def get_git_hash():
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
        if git_hash.isalnum():
            return git_hash
        else:
            return None
    except:
        return None
    
    
def generate_test_list():
    global high_prio_dir

    # pytest pybuda/test/model_demos/high_prio --setup-plan
    res = subprocess.check_output(["pytest", high_prio_dir, "--setup-plan"], stderr=subprocess.STDOUT).decode("utf-8")
    
    test_list = []
    test_count = 0

    lines = res.split("\n")
    for line in lines:
        if "warnings summary" in line or "slowest durations" in line:
            break
        
        if line and line.startswith("        " + high_prio_dir) and "::" in line:
            line = line.strip()
            line = line.split(" (fixtures used:")[0] if " (fixtures used:" in line else line

            test_list.append(line)
            test_count += 1

    return test_list


def print_test_start_info(test_name, current_i, test_count):
    if current_i == 0:
        print("Test progress & run details\n")
        print("#" * 32 + "\n")
    print(f"Running: {test_name}")
    print(f"Current progress {current_i}/{test_count}\n")
    print("#" * 32 + "\n")


def write_test_info_to_log_file(test_log_file, test, test_log_file_path):
    test_log_file.write("Test details\n")
    test_log_file.write("#" * 32 + "\n\n")
    test_log_file.write("Hostname:\n")
    test_log_file.write(hostname + "\n\n")
    test_log_file.write("Test command:\n")
    test_log_file.write("pytest -svv --durations=0 " + test + "\n\n")
    test_log_file.write("Log path:\n")
    test_log_file.write(test_log_file_path + "\n\n")
    test_log_file.write("#" * 32 + "\n\n")
    test_log_file.flush()
    

def write_test_out_to_file(test_log_file, res):
    """
    Write stdout to log file.

    For some reason, loguru (python) logs are captured as stderr. Therefore 
    these are not fully in sync when changing between different loggers (e.g
    python or C++ one). However, there are more logs then before, so it's
    still beneficial to use as is.
    """
    if res.stderr is not None:
        test_log_file.write(res.stderr.decode("utf-8"))
    if res.stdout is not None:
        test_log_file.write(res.stdout.decode("utf-8"))
    test_log_file.flush()
    test_log_file.close()


def extract_test_path_info(full_test_path, info_type):
    try:
        if info_type == "file_name":
            res = full_test_path.split("::")[0].split("/")[-1].split(".")[0]
        elif info_type == "test_name":
            res = full_test_path.split("::")[1].split("[")[0]
        elif info_type == "test_variant":
            res = full_test_path.split("::")[1].split("[")[1].split("]")[0].replace("-", "_").replace("/", "_").replace(" ", "_").replace(".", "_")
        else:
            raise ValueError("Invalid info_type")
        
        return res
    except Exception as ex:
        print("RunScriptError: ", ex)
        return "unknown"


def extract_test_details(test_log_dir_path, full_test_path):
    test_log_file_name = extract_test_path_info(full_test_path, "file_name")
    test_log_test_name = extract_test_path_info(full_test_path, "test_name")
    test_log_test_variant = extract_test_path_info(full_test_path, "test_variant")
    
    test_log_file_path = test_log_dir_path + "/" + test_log_file_name + "_" + test_log_test_name + "_" + test_log_test_variant + ".log"
    
    return test_log_file_path


def collect_error_logs(run_date, commit_sha):
    logs_path = "logs" + "/" + run_date + "/" + commit_sha
    
    # Find all summary files    
    summary_files = glob.glob(f"{logs_path}/summary*.log")
    assert len(summary_files) > 0

    failed_variants = []
    for summary_file in summary_files:
        with open(summary_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                test_variant, results = line.split(": ")
                result = results.strip()
                if result != "0":
                    test_log_file_path = extract_test_details("", test_variant)
                    failed_variants.append(test_log_file_path[1:])
        
        for root, dirs, files in os.walk(logs_path):
            for file in files:
                if file.endswith(".log"):
                    file_path = os.path.join(root, file)
                    if file_path.split('/')[-1] in failed_variants:
                        print(file_path)
                        with open(file_path, "r") as f:
                            lines = f.readlines()
                            count = 0
                            for i, line in enumerate(lines[::-1]):
                                if "ERROR" in line or "error" in line or "Error" in line:
                                    count += 1
                                    if count > 10:
                                        break
                                    print(line)
                                    print()

def compile_test_configurations(test_list):
    for i, test in enumerate(test_list):
        # Generate configuration file with all overrides that needs to be tested in combinations
        os.environ["PYBUDA_OVERRIDES_VETO"] = "1"
        os.environ["PYBUDA_OVERRIDES_VETO_CUSTOM_SETUP"] = test
        print(f"Running {i}/{len(test_list)}) {test}")
        sys_out = subprocess.run(["pytest", "-svv", "--durations=0", test], capture_output=True)
        
        with open("out.txt", "a") as f:
            f.write(sys_out.stdout.decode("utf-8"))
            f.write(sys_out.stderr.decode("utf-8"))
            

def run_tests():
    global reset_command_wh_b0, reset_command_gs, testlist
    
    parser = argparse.ArgumentParser(description="Script for manually running all high priority model demos (instead of using CI)")
    # General run functionality
    parser.add_argument("-r", "--reverse", help="Run reversely sorted test list", action="store_true")
    parser.add_argument("-np", "--netlist-prefix", help="Specify short commit sha on which this script is run. Has to be 9 char long (e.g. a5d778af5)")
    
    # Helper functionality (e.g. log collection)
    parser.add_argument("-co", "--collect-only", help="Collect error logs based on failed variants", action="store_true")
    parser.add_argument("-d", "--date", help="Specify date of run in format dd_mm (e.g. 27_03)")
    parser.add_argument("-s", "--sha", help="Specify short commit sha on which this script is run. Has to be 9 char long (e.g. a5d778af5)")
    parser.add_argument("-po", "--print-only", help="Instead of running, just prints out the test list", action="store_true")
    
    parser.add_argument("-ovc", "--override-veto-compile", help="Compile list of each variant configuration set in test (both general and env based configs)", action="store_true")

    args = parser.parse_args()
    
    if args.collect_only:
        assert args.date and args.sha, "Date and commit sha has to be specified when collecting logs"
        
        if len(args.date) != 5 and args.date[2] != "_":
            raise ValueError("Date has to be in format dd_mm (e.g. 27_03)")
        
        if len(args.sha) != 9:
            raise ValueError("Commit sha has to be 9 char long")
        
        collect_error_logs(args.date, args.sha)
        
        return
    
    detected_devices = pybuda.detect_available_devices()
    device_type = BackendType.Golden if len(detected_devices) == 0 else BackendType.Silicon
    if device_type == BackendType.Silicon:
        if detected_devices[0] == BackendDevice.Grayskull:
            reset_command = reset_command_gs
        elif detected_devices[0] == BackendDevice.Wormhole_B0:
            reset_command = reset_command_wh_b0
        else:
            raise ValueError("Unknown device")
    else:
        reset_command = ""
            
    # Sanity reset run if machine is in bad state
    if device_type == BackendType.Silicon:
        os.system(reset_command)
    
    # Set needed env vars
    set_env_vars_to_match_ci(device_type)
    
    # Generate or fetch test list
    if testlist == []:
        testlist = generate_test_list()

    if args.print_only:
        for i, test in enumerate(testlist):
            print(f"{test}")
        return

    test_count = len(testlist)
    
    if args.reverse:
        testlist = testlist[::-1]
        
    if args.override_veto_compile:
        compile_test_configurations(testlist)
        return
    
    # Get commit hash and date-time references
    commit = get_git_hash()
    if commit is None:
        commit = "unknown"
    run_date = date.today().strftime("%d_%m")

    # Setup result summary file and directory
    sum_log_dir_path = "logs" + "/" + run_date + "/" + commit
    if not os.path.exists(sum_log_dir_path):
        os.makedirs(sum_log_dir_path)
    sum_log_file_suffix = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(9))        
    sum_log_file_path = sum_log_dir_path + f"/summary_{hostname}_{sum_log_file_suffix}.log"
    test_sum_file = open(sum_log_file_path, "w")

    # Run each test variant as subprocess
    for i, test in enumerate(testlist):
        if test in testlist_to_ignore:
            print(f"Skipping {test}")
            continue

        print_test_start_info(test, i, test_count)

        # Setup log file and directory
        test_log_dir_path = sum_log_dir_path + "/" + "/".join(test.split("::")[0].split("/")[:-1])
        if not os.path.exists(test_log_dir_path):
            os.makedirs(test_log_dir_path)

        test_log_file_path = extract_test_details(test_log_dir_path, test)
        test_log_file = open(test_log_file_path, "w")

        write_test_info_to_log_file(test_log_file, test, test_log_file_path)
        
        # Override graph name (Netlist name)
        os.environ["PYBUDA_GRAPH_NAME_SUFFIX"] = args.netlist_prefix if args.netlist_prefix else "" + test_log_file_path.split('/')[-2] + "_" + test_log_file_path.split('/')[-1].split('.')[0]

        # Run PyTest as subprocess
        res = subprocess.run(["pytest", "-svv", "--durations=0", test], capture_output=True)

        write_test_out_to_file(test_log_file, res)

        new_line_char = "\n"
        test_sum_file.write(f"{test}: {res.returncode}{new_line_char}")
        test_sum_file.flush()
        if res.returncode != 0:
            if device_type == BackendType.Silicon:
                os.system(reset_command)

    test_sum_file.close()


if __name__ == "__main__":
    run_tests()
