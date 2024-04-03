#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict

import os
import sys
import curses
import csv
import yaml
import curses.ascii
import textwrap
import re
import pickle
from collections import defaultdict
from loguru import logger

def arch_clk(arch):
    """
    Return clock speed for an arch TODO: get this from somewhere?
    """
    if arch == "wormhole_b0":
        return 10**9
    if arch == "grayskull":
        return 1.2 * 10**9

    print(f"Unknown arch found in netlist: {arch}")
    sys.exit(5)

def try_parse_float(string, else_default=0.0):
    if string == "N/A" or (isinstance(string, str) and string.strip() == ""):
        return else_default
    return float(string)

def as_gb_sec(clock_speed, bytes_cycle):
    one_gb = 1e9
    return (bytes_cycle * clock_speed) / one_gb

def load_perf_analysis(epoch_count, config) -> List[Dict]:
    """
    Load backend graph perf report for each epoch. Generate per-kernel numbers from totals, since current version
    in pybuda only has totals. Remove once BBE is pulled in with new backend perf analyzer that has per-kernel numbers.
    """

    print(f"Loading performance analysis data for {epoch_count} epochs...")
    epoch_data = []
    for e in range(epoch_count):

        # Input file path
        te = config["spatial_temporal_map"][e]
        dir = config["test_dir"]
        input_file_path_te = f"{dir}/perf_results/analyzer_results/fwd_0_{e}_temporal_epoch_{te}/graph_perf_report.csv"
        input_file_path_e = f"{dir}/perf_results/analyzer_results/fwd_0_{e}/graph_perf_report.csv"
        
        # Grayskull and Wormhole dump graph names in different structures
        if os.path.exists(input_file_path_te):
            input_file_path = input_file_path_te
        elif os.path.exists(input_file_path_e):
            input_file_path = input_file_path_e
        else:
            print(f"Error: None of the backend perf analyzer result files {input_file_path_te}, {input_file_path_e} exist.")
            sys.exit(3)
            
        # Data structure to hold the rows
        data_table = []

        # Open the input CSV file
        try:
            with open(input_file_path, 'r') as infile:
                reader = csv.reader(infile)
                
                # Read the header and find the required column indices
                header = next(reader)
                idx_kernel_total_runtime = header.index('kernel_total_runtime')
                idx_bw_bound_total_runtime = header.index('bw_bound_total_runtime')
                idx_first_to_last_input = header.index('first_to_last_input')
                
                # Process each row
                for row in reader:
                    # Extract values
                    kernel_total_runtime = try_parse_float(row[idx_kernel_total_runtime])
                    bw_bound_total_runtime = try_parse_float(row[idx_bw_bound_total_runtime])
                    first, last = map(int, row[idx_first_to_last_input].split('->'))
                    number_of_inputs = last - first + 1
                    
                    # Calculate new columns
                    kernel_single_runtime = int(kernel_total_runtime / number_of_inputs)
                    bw_bound_single_runtime = int(bw_bound_total_runtime / number_of_inputs)
                    
                    # Store row as a dictionary
                    row_dict = {col_name: value for col_name, value in zip(header, row)}
                    row_dict['kernel_single_runtime'] = kernel_single_runtime
                    row_dict['bw_bound_single_runtime'] = bw_bound_single_runtime
                    
                    # Append to the data table
                    data_table.append(row_dict)

                epoch_data.append(data_table)
        except Exception as e:
            print(f"Error: Failed to load backend perf analysis data from {input_file_path}. Details: {e}")
            sys.exit(3)

    return epoch_data

def load_netlist(config):
    """
    Load netlist, and extract relevant fields from each op. Ignore non-op data. Also return device arch.
    """
    netlist = config["test_dir"] + "/" + config["netlist"]
    print(f"Loading netlist {netlist}...")
    if not os.path.exists(netlist):
        print(f"Error: Netlist {netlist} does not exist.")
        sys.exit(2)
    
    try:
        with open(netlist, 'r') as file:
            content = yaml.safe_load(file)
    except Exception as e:
        print(f"Error: An error occurred while reading the YAML file. Details: {e}")
        sys.exit(2)

    graphs_section = content['graphs']
    queues_section = content['queues']
    programs_section = content['programs']
    data_table = []
    graph_queues = {}

    # Populate output queues
    for graph_name, graph_details in graphs_section.items():
        graph_queues[graph_name] = []
        for queue_name, queue in queues_section.items():
            if queue["input"] != "HOST":
                graph_queues[graph_name].append(queue["input"])

    # Populate non-prologue input queues
    for program in programs_section:
        for program_name, instructions in program.items():
            for instruction in instructions:
                if "execute" in instruction:
                    exe_instr = instruction["execute"]
                    for queue_name, settings in exe_instr["queue_settings"].items():
                        if not settings["prologue"]:
                            graph_queues[exe_instr["graph_name"]].append(queue_name)

    # Find bottom-rightmost core used in the nelist
    br_core = [0, 0]
    for graph_name, graph_details in graphs_section.items():
        ops = {}
        for op_name, op_details in graph_details.items():
            if not isinstance(op_details, dict):
                continue

            ops[op_name] = {
                "graph": graph_name,
                "op_name": op_name,
                "type": op_details['type'],
                "grid_size": op_details['grid_size'],
                "t_value": op_details['t'],
                "mblock": op_details['mblock'],
                "ublock": op_details['ublock'],
                "inputs": op_details['inputs'],
                "input_is_dram": [name in graph_queues[graph_name] for name in op_details['inputs']],
                "output_is_dram": op_name in graph_queues[graph_name],
            }
            if 'attributes' in op_details and 'u_kt' in op_details['attributes']:
                ops[op_name]["m_k"] = op_details['attributes']['m_k']
                ops[op_name]["u_kt"] = op_details['attributes']['u_kt']
                
            op_br = [sum(x) for x in zip(op_details['grid_loc'], op_details['grid_size'], [-1, -1])]
            for i in range(2):
                if op_br[i] > br_core[i]:
                    br_core[i] = op_br[i]

        data_table.append(ops)

    # figure out spatial to temporal epoch maping
    spatial_temporal_map = []
    for graph_name in graphs_section:
        m = re.search(r'_temporal_epoch_(\d+)$', graph_name)
        if m:
            spatial_temporal_map.append(int(m.group(1)))
        else:
            spatial_temporal_map.append(0 if len(spatial_temporal_map) == 0 else spatial_temporal_map[-1] + 1)
    config["spatial_temporal_map"] = spatial_temporal_map

    # record the architecture
    config["arch"] = content["devices"]["arch"]

    # Figure out the number of cores. This is wonky, we need a more reliable source... for now, assume 1 row harvested, and then upsize if 
    # netlists uses more rows or columns
    if config["arch"] == "wormhole_b0":
        grid = [9, 8]
    elif config["arch"] == "grayskull":
        grid = [11, 10]
    else:
        print(f"Unknown arch found in netlist: {config['arch']}")
        sys.exit(5)
    
    for i in range(2):
        if br_core[i] > grid[i]:
            grid[i] = br_core[i]
    config["core_count"] = grid[0] * grid[1]

    # Figure out the number of devices
    devices = set()
    for graph_name, graph_details in graphs_section.items():
        for k, v in graph_details.items():
            if k == "target_device":
                devices.add(v)
    assert len(devices) > 0, "No target_device statements found in netlist graphs"
    config["device_count"] = len(devices)

    #logger.debug(f"Found {config['core_count']} cores and {config['device_count']} devices in netlist.")

    return data_table

def load_estimated_cycles():
    """
    Load the cycle counts that were estimated for each op during compile time.
    """
    file_path = "op_perf.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist. Run with PYBUDA_OP_PERF=1 to generate it, if running with pybuda. Loading will continue without it.")
        return {}

    print(f"Loading {file_path}...")
    data_table = {}
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row[' cycles'] = int(row[' cycles'])
                row[' limiter_cycles'] = int(row[' limiter_cycles'])
                data_table[row["name"]] = row
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}.")
        sys.exit(4)
        
    return data_table

def load_balancer_score():
    """
    Load balancer score for every epoch and for total solution.
    """
    file_path = "balancer_score.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist. Run with PYBUDA_OP_PERF=1 to generate it, if running with pybuda. Loading will continue without it.")
        return {}

    print(f"Loading {file_path}...")
    data_table = {}
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row[' score'] = float(row[' score'])
                data_table[row["epoch"]] = row
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}.")
        sys.exit(6)

    return data_table

def verify_data(netlist_data, perf_data, estimated_data):
    """ 
    Verify that the data is consistent between the 3 sources.
    """
    print("Verifying data...")
    assert len(netlist_data) == len(perf_data), "Netlist and perf data have different number of epochs."

    # Check that each op in the perf data is present in the netlist data, and estimated data
    for epoch_idx, epoch in enumerate(perf_data):
        for op in epoch:
            assert op['op_name'] in netlist_data[epoch_idx], f"Op {op['op_name']} in perf data not found in netlist data."
            if len(estimated_data) > 0:
                assert op['op_name'] in estimated_data, f"Op {op['op_name']} in perf data not found in estimated data."

    print("Verified!")

def merge_data(netlist_data, perf_data, estimated_data, config):
    """
    Merge data into one, per-epoch, table of ops with all data
    """
    print("Merging data...")
    merged_data = []
    for epoch_idx, epoch in enumerate(perf_data):
        data_table = {}
        for op in epoch:
            data_table[op['op_name']] = op
            for k, d in netlist_data[epoch_idx][op['op_name']].items():
                data_table[op['op_name']][k] = d
            if len(estimated_data) > 0:
                data_table[op['op_name']]['estimated_cycles'] = estimated_data[op['op_name']][' cycles']
                data_table[op['op_name']]['estimated_lim_cycles'] = estimated_data[op['op_name']][' limiter_cycles']
                data_table[op['op_name']]['tiles'] = estimated_data[op['op_name']][' tiles']
            else:
                data_table[op['op_name']]['estimated_cycles'] = 0
                data_table[op['op_name']]['estimated_lim_cycles'] = 0
                data_table[op['op_name']]['tiles'] = 0

        merged_data.append(data_table)

    if not config["spatial_epochs"]:
        # Merge spatial epochs into temporal epochs, since they run at the same time
        temporal_data = []
        for epoch_idx, data in enumerate(merged_data):
            temporal_epoch = config["spatial_temporal_map"][epoch_idx]
            while len(temporal_data) < temporal_epoch + 1:
                temporal_data.append({})    
            for k, v in data.items():
                temporal_data[temporal_epoch][k] = v

        merged_data = temporal_data

    return merged_data

def summarize_epoch(epoch, epoch_data, balancer_score_data, config):
    """
    Summarize data for ops in one epoch to get a summary - epoch speed, utilization, slowest op, etc.
    """
    summary = {"epoch": epoch}

    slowest_op = max(epoch_data.items(), key=lambda d: d[1].get('bw_kernel', float('-inf')), default=None)
    slowest_estimated_op = max(epoch_data.items(), key=lambda d: d[1].get('estimated_lim_cycles', float('-inf')), default=None)
    estimated_pipeline_cycles = slowest_estimated_op[1]['estimated_lim_cycles']
    sum_estimated_kernel_err = sum([abs(item["estimated_cycles"] - item["kernel_single_runtime"]) for item in epoch_data.values()])
    sum_estimated_lim_err = sum([abs(item["estimated_lim_cycles"] - item["bw_bound_single_runtime"]) for item in epoch_data.values()])
    sum_kernel = sum([item["kernel_single_runtime"] for item in epoch_data.values()])
    sum_kernel_bw = sum([item["bw_bound_single_runtime"] for item in epoch_data.values()])

    summary["slowest_op"] = slowest_op[1]["op_name"]
    summary["slowest_op_short"] = summary["slowest_op"] if len(summary["slowest_op"]) <= 50 else summary['slowest_op'][:47] + "..."
    summary["pipeline_cycles"] = slowest_op[1]["bw_bound_single_runtime"]

    total_cores = config["core_count"]
    if not config["spatial_epochs"]:
        total_cores *= config["device_count"] # Temporal epochs use all devices

    assert total_cores > 0, "Total available cores is zero"

    clock_speed = arch_clk(config["arch"])

    rate = int(clock_speed / summary["pipeline_cycles"])
    if rate == 0: 
        rate = int(100 * clock_speed / summary["pipeline_cycles"]) / 100 # 2 decimal places
    summary["inputs_per_second"] = rate

    # Calculate:
    # - the total utilization. For each op, multiply cores with math utilization, times the ratio of op's runtime to slowest op.
    # - number of cores used by matmuls
    # - balancer utilization - same as total util, but using esimated numbers. This is the "balancer score" in effect, how well balancer did
    #   its job with data it was given
    util = 0.0
    balancer_util = 0.0
    matmul_cores = 0
    required_noc_bw = 0.0
    required_dram_bw = 0.0
    actual_noc_bw = 0.0
    actual_dram_bw = 0.0
    for op in epoch_data.values():
        cores = op["grid_size"][0] * op["grid_size"][1]
        if summary["pipeline_cycles"] > 0:
            util += cores * try_parse_float(op["bw_bound_math_utilization"]) * try_parse_float(op["bw_bound_single_runtime"]) / summary["pipeline_cycles"]
        if estimated_pipeline_cycles > 0:
            balancer_util += cores * try_parse_float(op["estimated_cycles"]) / estimated_pipeline_cycles
        if op["type"] == "matmul":
            matmul_cores += cores
        for bw in op["input_bws"] + [op["output_bw"]]:
            if bw["is_dram"]:
                required_dram_bw += bw["required"]
                actual_dram_bw += bw["actual"]
            else:
                required_noc_bw += bw["required"]
                actual_noc_bw += bw["actual"]

    util /= total_cores
    util = int(10 * util) / 10 # 1 decimal place
    balancer_util /= total_cores
    balancer_util = int(1000 * balancer_util) / 10 # 1 decimal place
    required_noc_bw = int(10 * as_gb_sec(clock_speed, required_noc_bw)) / 10 # 1 decimal place
    required_dram_bw = int(10 * as_gb_sec(clock_speed, required_dram_bw)) / 10 # 1 decimal place
    actual_noc_bw = int(10 * as_gb_sec(clock_speed, actual_noc_bw)) / 10 # 1 decimal place
    actual_dram_bw = int(10 * as_gb_sec(clock_speed, actual_dram_bw)) / 10 # 1 decimal place

    if str(epoch) in balancer_score_data:
        balancer_epoch_score = balancer_score_data[str(epoch)][' score']
    else:
        balancer_epoch_score = 0.0

    summary["real_utilization"] = util
    summary["balancer_util"] = balancer_util
    summary["matmul_cores"] = matmul_cores
    summary["required_noc_bw"] = required_noc_bw
    summary["required_dram_bw"] = required_dram_bw
    summary["actual_noc_bw"] = actual_noc_bw
    summary["actual_dram_bw"] = actual_dram_bw
    summary["sum_estimated_kernel_err"] = sum_estimated_kernel_err
    summary["sum_estimated_lim_err"] = sum_estimated_lim_err
    summary["sum_kernel"] = sum_kernel
    summary["sum_kernel_bw"] = sum_kernel_bw
    summary["balancer_epoch_score"] = balancer_epoch_score
    return summary

def summarize_model(epoch_summary_data, balancer_score_data, config):
    """
    Calculate overall model summary - total speed, utilization, etc.
    """

    # overal speed is roughly 1/total_time, where total_time is the sum of slowest ops for each epoch
    clock_speed = arch_clk(config["arch"])
    overall_speed = int(clock_speed / sum([e["pipeline_cycles"] for e in epoch_summary_data]))

    # overall utilization is utilization of each epoch times the cycles spent in that epoch, divided by the total cycles
    overall_util = sum([e["real_utilization"] * e["pipeline_cycles"] for e in epoch_summary_data]) / sum([e["pipeline_cycles"] for e in epoch_summary_data])
    overall_util = int(100 * overall_util) / 100 # 2 decimal places
    sum_estimated_kernel_err = sum(item["sum_estimated_kernel_err"] for item in epoch_summary_data)
    sum_estimated_lim_err = sum(item["sum_estimated_lim_err"] for item in epoch_summary_data)
    sum_kernel = sum(item["sum_kernel"] for item in epoch_summary_data)
    sum_kernel_bw = sum(item["sum_kernel_bw"] for item in epoch_summary_data)

    kernel_estimation_error = int((sum_estimated_kernel_err / float(sum_kernel)) * 100)
    kernel_bw_estimation_error = int((sum_estimated_lim_err / float(sum_kernel_bw)) * 100)

    if "total" in balancer_score_data:
        balancer_solution_score = balancer_score_data["total"][' score']
    else:
        balancer_solution_score = 0.0

    return {
        "overall_speed": overall_speed,
        "overall_util": overall_util,
        "balancer_solution_score": balancer_solution_score,
        "kernel_estimation_error": kernel_estimation_error,
        "kernel_bw_estimation_error": kernel_bw_estimation_error
    }

def summarize_data(data, balancer_score_data, config):
    """
    Summarize data for ops in epochs to get a per-epoch summary - epoch speed, utilization, slowest op, etc.
    """
    epoch_summary_data = [summarize_epoch(i, d, balancer_score_data, config) for i, d in enumerate(data)]
    model_summary_data = summarize_model(epoch_summary_data, balancer_score_data, config)
    return epoch_summary_data, model_summary_data

def process_epoch_data(epoch_data, config):
    """
    Analyze and generate new columns of data that are helpful to the user
    """

    for epoch in epoch_data:
        for op in epoch.values():
            # string values for compact display
            op["op_name_short"] = op["op_name"] if len(op["op_name"]) <= 50 else op["op_name"][:47] + "..."
            op["grid_size_str"] = f"{op['grid_size'][0]},{op['grid_size'][1]}"
            op["mblock_str"] = f"{op['mblock'][0]:2},{op['mblock'][1]:2}"
            op["ublock_str"] = f"{op['ublock'][0]:2},{op['ublock'][1]:2}"

            op["input_bws"] = []
            for i in range(len(op["inputs"])):
                if f"input_pipe_bw_{i}" not in op:
                    op[f"input_pipe_bw_{i}"] = 0.0
                if f"required_input_bw_{i}" not in op:
                    op[f"required_input_bw_{i}"] = 0.0

                op[f"input_pipe_bw_{i}"] = max(0.0, try_parse_float(op[f"input_pipe_bw_{i}"]))

                input_bw = {
                    "actual": try_parse_float(op[f"input_pipe_bw_{i}"]),
                    "required": try_parse_float(op[f"required_input_bw_{i}"]),
                    "is_dram": op["input_is_dram"][i],
                }
                op["input_bws"].append(input_bw)

            op["output_bw"] = {
                "actual": float(op["output_pipe_bw_0"]),
                "required": float(op["required_output_pipe_bw_0"]),
                "is_dram": op["output_is_dram"],
            }

            if 'u_kt' in op:
                op["m_k/u_kt"] = f"{op['m_k']:2}/{op['u_kt']:2}"

            # Find which pipe is the worst, percentage wise
            # figure out the source of bw-based slowdown
            op["bw_problem"] = "" 
            if op["bw_bound_single_runtime"] > op["kernel_single_runtime"]:

                factors = [i["required"] / i["actual"] if i["actual"] > 0.0 else 0.0 for i in op["input_bws"]]
                worst_factor = max(factors)
                worst_in_pipe = factors.index(worst_factor)

                # Compare to out pipe
                out_factor = op["output_bw"]["required"] / op["output_bw"]["actual"]
                if worst_factor > out_factor:
                    is_noc = op["inputs"][worst_in_pipe] in epoch # reading from another op
                    if is_noc: 
                        op["bw_problem"] = f"noc in{worst_in_pipe}"
                    else:
                        op["bw_problem"] = f"dram in{worst_in_pipe}"

                else:
                    op["bw_problem"] = "output"


    return epoch_data


def load_data(config):
    netlist_data = load_netlist(config)
    epoch_count = len(netlist_data)
    perf_data = load_perf_analysis(epoch_count, config)
    estimated_data = load_estimated_cycles()
    balancer_score_data = load_balancer_score()

    verify_data(netlist_data, perf_data, estimated_data)
    epoch_data = merge_data(netlist_data, perf_data, estimated_data, config)
    epoch_data = process_epoch_data(epoch_data, config)
    epoch_summary_data, model_summary_data = summarize_data(epoch_data, balancer_score_data, config)
    model_summary_data['netlist'] = config["netlist"]

    data = {
        'epochs': epoch_data,
        'epoch_summary': epoch_summary_data,
        'model_summary': model_summary_data
    }
    return data

def cache_data(data, cache_file):
    cache_data = defaultdict(dict)
    cache_dir = os.path.dirname(cache_file)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as file:
            cache_data = pickle.load(file)
            # logger.debug(cache_data)

    # Updating the cache 'op_model' context
    context = "op_model"
    if context not in cache_data:
        cache_data[context] = defaultdict(dict)

    for epoch in data['epochs']:
        for op in epoch.values():
            runtime = op["kernel_single_runtime"]  # bw_bound_single_runtime
            op_type = op["type"]
            if op_type == "matmul":
                shapes = (op["mblock"][0], op["mblock"][1], op["ublock"][0], op["ublock"][1], op["t_value"], op['m_k'], op['u_kt'])
            else:
                shapes = (op["mblock"][0], op["mblock"][1], op["ublock"][0], op["ublock"][1], op["t_value"])
            cache_data[context][op['type']][shapes] = runtime
            cache_data[context][op['op_name']][shapes] = runtime
            # logger.debug(f"Updated cache[{op['type']}][{shapes}]={runtime} from {op['op_name']} ")

    with open(cache_file, 'wb') as file:
        pickle.dump(cache_data, file)

    summary = data['model_summary']
    print(f"Updated perf cache {cache_file} (performance: {summary['overall_speed']}/s, util: {summary['overall_util']}%)")

def draw_table(win, table, header, config, highlight_funcs=None, ljust_cols = [0]):

    row_offset = config['row_offset']
    col_offset = config['col_offset']

    max_height, max_width = win.getmaxyx()
    available_width = max_width
    available_height = max_height - 6  # Reserve space for header, separators, and key map

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(header, *table)]
    num_columns = len(col_widths)

    # Truncate columns if they don't fit. Always keep the first column (name)
    while col_widths[0] + sum(col_widths[col_offset+1:col_offset+num_columns]) + num_columns * 3 > available_width and num_columns > 0:
        num_columns -= 1

    if num_columns == 0:
        return # nothing to draw

    header = [header[0]] + header[col_offset+1:col_offset+num_columns]
    col_widths = [col_widths[0]] + col_widths[col_offset+1:col_offset+num_columns]
    
    # Create a formatted string for the header
    header_fmt = " | ".join("{{:{}}}".format(w) for w in col_widths)
    win.addstr(2, 0, header_fmt.format(*header), curses.A_BOLD)
    
    # Draw the double line separator, accounting for the padding and column dividers
    win.addstr(3, 0, '=' * (sum(col_widths) + len(col_widths) * 3 - 1), curses.A_BOLD)

    # Draw the table rows with custom highlighting
    row_screen_offset = 4 # 2 for table header, and 2 for epoch summary
    for i, row in enumerate(table[row_offset:row_offset + available_height]):

        x_pos = 0
        row = [row[0]] + row[col_offset+1:col_offset+num_columns]
        for j, cell in enumerate(row):
            if j in ljust_cols:
                cell_str = str(cell).ljust(col_widths[j])
            else:
                cell_str = str(cell).rjust(col_widths[j])
            if highlight_funcs and header[j] in highlight_funcs:
                win.addstr(i+ row_screen_offset, x_pos, cell_str, highlight_funcs[header[j]](cell, dict(zip(header, row))))
            else:
                win.addstr(i+ row_screen_offset, x_pos, cell_str)
            x_pos += len(cell_str) + 3  # Moving cursor to next column
            if j < len(row) - 1:
                win.addstr(i+ row_screen_offset, x_pos - 3, " | ")

# Some definition for good/bad u_kt to highlight in the table
def great_u_kt(x):
    if "/" not in x:
        return False
    m_k, u_kt = [int(d) for d in x.split("/")]
    return m_k == 1

def bad_u_kt(x):
    if "/" not in x:
        return False
    m_k, u_kt = [int(d) for d in x.split("/")]
    return (u_kt < 4 and m_k >= 4) or (u_kt == 1 and m_k > 1)

# Some definition for good/bad bw to highlight in the table
highlight_funcs = {
    "bw_util": lambda x, _: curses.color_pair(2) if float(x) > 50 else curses.color_pair(1) if float(x) < 25 else curses.color_pair(0), 
    "util": lambda x, _: curses.color_pair(2) if float(x) > 50 else curses.color_pair(1) if float(x) < 25 else curses.color_pair(0), 
    "balancer util": lambda x, _: curses.color_pair(2) if float(x) > 50 else curses.color_pair(1) if float(x) < 25 else curses.color_pair(0), 
    "m/u_kt": lambda x, _: curses.color_pair(2) if great_u_kt(x) else curses.color_pair(1) if bad_u_kt(x) else curses.color_pair(0),
    "out_bw": lambda x, d: curses.color_pair(1) if x.isnumeric() and float(x) < float(d["out_req"]) else curses.color_pair(0),
    'est': lambda x, d: curses.color_pair(1) if abs(x-d['kernel']) >= 0.5 * d['kernel'] else curses.color_pair(3) if abs(x-d['kernel']) >= 0.2 * d['kernel'] else curses.color_pair(0),
    'est_lim': lambda x, d: curses.color_pair(1) if abs(x-d['bw_kernel']) >= 0.5 * d['bw_kernel'] else curses.color_pair(3) if abs(x-d['bw_kernel']) >= 0.2 * d['bw_kernel'] else curses.color_pair(0)
}
for i in range(8):
    highlight_funcs[f"in{i}_bw"] = (lambda x, d, i=i: curses.color_pair(1) if f"in{i}_req" in d and try_parse_float(x) < try_parse_float(d[f"in{i}_req"]) else curses.color_pair(0))

def draw_epoch_summary(win, epoch, epoch_summary_data):
    data = epoch_summary_data[epoch]
    win.addstr(0, 0, f"Epoch ")
    win.addstr(f"{epoch}", curses.A_BOLD)
    win.addstr(" Speed: ")
    win.addstr(f"{data['inputs_per_second']}/s", curses.A_BOLD)
    win.addstr(", Utilization: ")
    win.addstr(f"{data['real_utilization']}%", curses.A_BOLD)
    win.addstr(", Balancer utilization: ")
    win.addstr(f"{data['balancer_util']}%", curses.A_BOLD)
    win.addstr(", Balancer score: ")
    win.addstr(f"{data['balancer_epoch_score']}", curses.A_BOLD)

def draw_model_summary(win, data):
    win.addstr(0, 0, "Netlist: ")
    win.addstr(data['netlist'], curses.A_BOLD)
    win.addstr(" Approximate performance: ")
    win.addstr(f"{data['overall_speed']}/s", curses.A_BOLD)
    win.addstr(" Balancer score: ")
    win.addstr(f"{data['balancer_solution_score']}", curses.A_BOLD)
    win.addstr(" Approximate utilization: ")
    win.addstr(f"{data['overall_util']}%", curses.A_BOLD)
    win.addstr(" Kernel estimate error: ")
    win.addstr(f"{data['kernel_estimation_error']}%", curses.A_BOLD)
    win.addstr(" Kernel BW estimate error: ")
    win.addstr(f"{data['kernel_bw_estimation_error']}%", curses.A_BOLD)

def draw_window(stdscr, text):
    
    # Get terminal size
    stdscr.refresh()
    height, width = stdscr.getmaxyx()

    # Define window size and position
    window_height = int(height / 1.5)
    window_width = int(width / 1.5)
    window_y = (height - window_height) // 2
    window_x = (width - window_width) // 2

    # Create a window
    win = curses.newwin(window_height, window_width, window_y, window_x)
    win.box()

    # Wrap the text to fit inside the window, taking into account the borders
    body = '\n'.join(['\n'.join(textwrap.wrap(line, window_width - 4,
                 break_long_words=False, replace_whitespace=False))
                 for line in text.splitlines() if line != ''])
    wrapped_text = body.splitlines()
    #wrapped_text = textwrap.wrap(text, window_width - 4)

    # Write the wrapped text to the window, starting at (1, 1), and making sure not to exceed the window height
    for i, line in enumerate(wrapped_text[:window_height - 2]):
        win.addstr(i + 1, 1, line)

    win.refresh()
    win.getch()
    win.clear()

def draw_help(win, config):
    if config["epoch"] is None:
        # summary
        help_text = """
Summary Window
 
This window shows the overall model summary at the top, and a table summarizing each epoch of the model. 
 
The overall performance of the model is based purely on the slowest ops in each epoch. Assuming the batch number is high enough to make epoch reconfiguration and pipeline fill/drain negligible, and that there are no major delays due to data transfer between host and the device, this should be a reasonable, albeit slightly optimistic, approximation. The overall utilization is similarly calculated using math utilizations measured on each core and could be optimistic if other delays are not negligible. 
 
A known limitation is that current backend measurement doesn't take into account fork-join delays, so if overall performance, or a particular epoch performance here looks much better than the real measured time, it could be a sign of a fork-join problem.
 
The fields in the summary table are:
    cycles: Pipeline stage cycles, i.e. the cycles of the slowest op
    speed: The number of inputs/s this epoch is processing
    util: The math utilization of the pipeline this epoch, in steady state
    mm cores: The number of cores occupied by matrix multiplication ops
    balancer util: Similar to util, but calculated using estimated op speeds given to the compiler/balancer. This measures how well balancer did its job, given the information it was given. 
        """
    else:
        help_text = """
Epoch Window
 
This window shows op performance and utilization for each op in the current epoch. Use P/N keys to move to previous/next epoch, and arrow keys to scroll the rows and columns of the table if it doesn't fit on the screen. Use F to toggle between full op names and shortened version.
  
The performance values in the table are measured on silicon using backend perf analyzer. The table is sorted with the slowest op at the top.
 
Some of the key fields in the table are:
    est: The estimated cycles for the op, given to the compiler/balancer. 
    kernel: The measured time kernel took to execute, with infinite input/output bandwidths.
    bw_kernel: The measure time for the kernel with real pipes feeding data in/out of the core. This is the "real" time it took for the op to complete.
    bw problem: Identifies the cause of the bw slowdown - input vs output pipe, and if input, then noc vs dram
    in/out columns: Bandwidths, in bytes/cycle, required for the kernel to run at full speed, and measured with real pipes. 
 
A known limitation is that current backend measurement doesn't take into account fork-join delays, so if overall performance, or a particular epoch performance here looks much better than the real measured time, it could be a sign of a fork-join problem.
  
For any problems with this tool, contact @ssokorac. If estimated op values are far from measured ones, file a bug on @macimovic. If measured values appear wrong, contact @kdabiri.

        """

    draw_window(win, help_text)



status_prompt = "[E] epoch [P] previous [N] next [S] summary [F] op names [R] reload [H] help [Q] quit [ARROWS] scroll"
epoch_prompt = " Epoch #: "

def display_screen(table_data, mapping, stdscr, config):
    """
    Main drawing function, repeatedly called on every key press
    """
    stdscr.erase()
    num_epochs = len(data['epochs'])

    epoch = config['epoch']

    # Figure out what to display and how
    if epoch is not None and 0 <= epoch < num_epochs:
        # Epoch data
        ljust_cols = [0] # left-justify op name, and nothing else
        draw_epoch_summary(stdscr, epoch, data['epoch_summary'])
    else:
        # Summary
        ljust_cols = [0, 1] # left-justify epoch, and slowest op
        draw_model_summary(stdscr, data['model_summary'])

    header = list(mapping.keys())
    display_data = []
    for row in table_data:
        row_values = []
        for k in mapping.values():
            if k in row: 
                row_values.append(row[k])
            else:
                row_values.append(" ")
        display_data.append(row_values)

    draw_table(stdscr, display_data, header, config, highlight_funcs, ljust_cols)
    
    # Display key map
    key_map = status_prompt
    if config['prompt_epoch']:
        key_map += epoch_prompt


    # Make sure the key map fits on the screen
    max_y, max_x = stdscr.getmaxyx()
    stdscr.addnstr(max_y - 1, 0, key_map, max_x-1)

def main(stdscr, data):

    # Curses config
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    key = ord('S')
    num_epochs = len(data['epochs'])
    config = {
        'prompt_epoch': False, # show epoch number prompt at the bottom
        'full_op_names': False, # show full op names (vs. shortened)
        'row_offset': 0, # row scrolling offset in tables
        'col_offset': 0, # column scrolling offset in tables
        'epoch': None, # current epoch, or None if we're on the summary screen
    }
    while key != ord('q') and key != ord('Q'):
        
        # Figure out which data to display
        if config['epoch'] is not None:
            # Epoch data
            table_data = data['epochs'][config['epoch']].values()

            # Order and names of columns to show on the epoch screen
            mapping = {
                    'name': 'op_name' if config['full_op_names'] else 'op_name_short',
                    'grid': 'grid_size_str',
                    'mb': 'mblock_str',
                    'ub': 'ublock_str',
                    't' : 't_value',
                    'm/u_kt': 'm_k/u_kt',
                    'est': 'estimated_cycles',
                    'kernel': 'kernel_single_runtime',
                    'util': 'kernel_math_utilization',
                    'est_lim': 'estimated_lim_cycles',
                    'bw_kernel': 'bw_bound_single_runtime',
                    'bw_util': 'bw_bound_math_utilization',
                    'bw problem': 'bw_problem',
                    'out_req': 'required_output_pipe_bw_0',
                    'out_bw': 'output_pipe_bw_0',
                    'in0_req': 'required_input_bw_0',
                    'in0_bw': 'input_pipe_bw_0',
                    'in1_req': 'required_input_bw_1',
                    'in1_bw': 'input_pipe_bw_1',
                    'in2_req': 'required_input_bw_2',
                    'in2_bw': 'input_pipe_bw_2',
                    'in3_req': 'required_input_bw_3',
                    'in3_bw': 'input_pipe_bw_3',
                    'in4_req': 'required_input_bw_4',
                    'in4_bw': 'input_pipe_bw_4',
                    'in5_req': 'required_input_bw_5',
                    'in5_bw': 'input_pipe_bw_5',
                    'in6_req': 'required_input_bw_6',
                    'in6_bw': 'input_pipe_bw_6',
                    'in7_req': 'required_input_bw_7',
                    'in7_bw': 'input_pipe_bw_7',
                    }
            max_rows = len(data['epochs'][config['epoch']]) - 2
            if all(d["balancer_util"] == 0 for d in data['epoch_summary']): # we had no balancer util numbers loaded
                del mapping["est"]
                del mapping["est_lim"]

        else:
            # Summary columns
            mapping = {
                    'epoch': 'epoch',
                    'slowest op': 'slowest_op' if config['full_op_names'] else 'slowest_op_short',
                    'cycles': 'pipeline_cycles',
                    'speed': 'inputs_per_second',
                    'util': 'real_utilization',
                    'mm cores': 'matmul_cores',
                    'balancer util': 'balancer_util',
                    'req noc bw GB/s': 'required_noc_bw',
                    'act noc bw GB/s': 'actual_noc_bw',
                    'req dram bw GB/s': 'required_dram_bw',
                    'act dram bw GB/s': 'actual_dram_bw',
                    }
            if all(d["balancer_util"] == 0 for d in data['epoch_summary']): # we had no balancer util numbers loaded
                del mapping["balancer util"]

            table_data = data['epoch_summary']
            max_rows = len(data['epoch_summary']) - 2
        
        max_columns = len(mapping) - 2

        display_screen(table_data, mapping, stdscr, config)
        key = stdscr.getch()
        
        if key == ord('R') or key == ord('r'):
            return True # reload
    
        elif key == ord('s') or key == ord('S'):
            config['epoch'] = None
            config['col_offset'] = 0 
            config['row_offset'] = 0 

        elif key == ord('f') or key == ord('F'):
            config["full_op_names"] = not config["full_op_names"]

        elif key == ord('h') or key == ord('H'):
            draw_help(stdscr, config)

        elif (key == curses.KEY_UP or key == ord('k')) and config['row_offset'] > 0:
            config['row_offset'] -= 1

        elif (key == curses.KEY_DOWN or key == ord('j')) and config['row_offset'] < max_rows:
            config['row_offset'] += 1

        elif key == curses.KEY_RIGHT and config['col_offset'] > 0:
            config['col_offset'] -= 1

        elif key == curses.KEY_LEFT and config['col_offset'] < max_columns:
            config['col_offset'] += 1

        elif key == ord('E') or key == ord('e'):
            config['prompt_epoch'] = True
            display_screen(table_data, mapping, stdscr, config)
            curses.echo()  # Enable echo to display input
            epoch_num = stdscr.getstr(curses.LINES - 1, len(status_prompt + epoch_prompt))
            curses.noecho()  # Disable echo
            try:
                if config['epoch'] is None:
                    config['col_offset'] = 0 # reset since we're going from summary to epoch
                config['epoch'] = int(epoch_num)
                if config['epoch'] < 0 or config['epoch'] >= num_epochs:
                    config['epoch'] = None
            except ValueError:
                config['epoch'] = None
            config['prompt_epoch'] = False
            config['row_offset'] = 0
            # don't reset col offset, so it's easy to compare columns across epochs

        elif key == ord('P') or key == ord('p'):
            if config['epoch'] is None:
                config['epoch'] = num_epochs - 1
            elif config['epoch'] > 0:
                config['epoch'] -= 1

        elif key == ord('N') or key == ord('n'):
            if config['epoch'] is None:
                config['epoch'] = 0
            elif config['epoch'] < num_epochs - 1:
                config['epoch'] += 1

        else:
            config['prompt_epoch'] = False

    return False # no reload


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="""
    Perf analyzer collects performance data from various sources and displays it in terminal. To use, run any pybuda test with PYBUDA_OP_PERF=1 and TT_BACKEND_PERF_ANALYZER=1 switches to generate data, and then run this script in pybuda root, providing the netlist.
    """)
    parser.add_argument('-n', '--netlist', help='Model netlist')
    parser.add_argument('-s', '--spatial_epochs', action='store_true', help='Show individual spatial epochs instead of temporal ones. Caution - overall performance estimate on multi-chip runs will not be accurate in this mode.')
    parser.add_argument('-d', '--dir', help='User specified output directory')
    parser.add_argument(      '--save', help='Save collected data into provided file')
    parser.add_argument(      '--load', help='Load data from a previously saved file, instead of from current workspace')
    parser.add_argument('-c', '--cache', help='Cache performance results in a file, to aid future compiles')
    args = parser.parse_args()

    logger.add("perf_analysis_debug.log")

    if not args.load and not args.netlist:
        fallback_dir = "tt_build/test_out"
        netlist_yaml = [file for file in os.listdir(fallback_dir) if file.endswith("netlist.yaml")]
        if len(netlist_yaml) == 1:
            args.netlist = netlist_yaml[0]
        else:
            print("Cannot locate netlist.yaml, --load or --netlist must be provided.")
            sys.exit(10)

    if args.load:
        if args.netlist:
            print("Both --load and --netlist are provided. Pick one or the other!")
            sys.exit(10)

        print(f"Loading collected data from {args.load}")
        try:
            with open(args.load, 'rb') as file:
                data = pickle.load(file)
        except Exception as e:
            print(f"Error: Failed to load analysis data from {args.load}. Details: {e}")
            sys.exit(10)
    else:
        config = {"netlist": args.netlist, "spatial_epochs": args.spatial_epochs, "test_dir": args.dir or os.path.dirname(os.path.realpath(args.netlist))}
        data = load_data(config)

    ui = True

    if args.cache:
        cache_data(data, args.cache)
        ui = False

    if args.save:
        if args.load:
            print("Both --load and --save are provided, --save ignored.")
        else:
            print(f"Saving collected data to {args.save}")
            with open(args.save, 'wb') as file:
                pickle.dump(data, file)
            ui = False

    if ui:
        print("Done loading data. Let's analyze!")
        reload = True
        while reload:
            reload = curses.wrapper(main, data)
