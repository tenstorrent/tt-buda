# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import re
import json
import copy

TARGET_PUSH_BANDWIDTH = {"float32": {0: 8.7, 11: 0.011, 21: 0.009, 30: 0.018, 17: 0.024},
                         "float16": {0: 8.7, 11: 0.0075, 21: 0.006, 30: 0.012, 17: 0.016},
                         "bfloat16": {0: 8.9, 11: 0.008, 21: 0.006, 30: 0.012, 17: 0.016}}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse output log and check bandwidth')
    parser.add_argument("--log", help="Log", default="A random string.")
    args = parser.parse_args()

    results = copy.deepcopy(TARGET_PUSH_BANDWIDTH)
    with open(args.log) as f:
        lines = f.readlines()
        curr_push = 0
        for line in lines:
            if "bandwidth test" in line:
                df = re.findall("torch.(.*) bandwidth test",line)[0]
                chip = int(re.findall("chip (.*) dram",line)[0])

                print(f"Pushing {df} to chip {chip}")
            if "Pushed inputs" in line:
                size = re.findall("Pushed inputs \((.*)\)",line)[0]
                curr_push = float(re.findall("inputs/s, (.*) GB/s",line)[0])
                print(f"{curr_push:.3f} GB/s, {size}")
                results[df][chip] = curr_push

    for df, chip_data in results.items():
        for chip, bw in chip_data.items():
            target = TARGET_PUSH_BANDWIDTH[df][chip]
            if (bw < target * 0.95):
                print(f"Measured push bandwidth is too low on chip {chip}: expected {target} GB/s got {bw:.3f} GB/s.")
                assert False
            if (bw > target * 1.20):
                print(f"Measured push bandwidth is too high on chip {chip}: expected {target} GB/s got {bw:.3f} GB/s. Update target values to measured bandwidth.")
                assert False
