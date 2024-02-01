# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
import yaml
import tempfile
import re
import shutil
import os
import pybuda._C.backend_api as backend_api
from pybuda._C import DataFormat 
import subprocess as sp
import json
import copy
from loguru import logger

# Track all temp directories used for intermediate steps
# Delete them as part of cleanup
temp_directories = []
def str_to_format(format_str):
    if(format_str == "Float32"):
        return DataFormat.Float32
    elif(format_str == "Float16"):
        return DataFormat.Float16
    elif(format_str == "Float16_b"):
        return DataFormat.Float16_b
    elif(format_str == "Bfp8"):
        return DataFormat.Bfp8
    elif(format_str == "Bfp8_b"):
        return DataFormat.Bfp8_b
    elif(format_str == "Bfp4"):
        return DataFormat.Bfp4
    elif(format_str == "Bfp4_b"):
        return DataFormat.Bfp4_b
    elif(format_str == "Bfp2"):
        return DataFormat.Bfp2
    elif(format_str == "Bfp2_b"):
        return DataFormat.Bfp2_b
    elif(format_str == "RawUInt32"):
        return DataFormat.RawUInt32
    elif(format_str == "RawUInt16"):
        return DataFormat.RawUInt16
    elif(format_str == "RawUInt8"):
        return DataFormat.RawUInt8
    else:
        assert False, "Invalid Format"
    
def uniquify_global_structures(model_paths):
    temp_dir = tempfile.mkdtemp()
    temp_directories.append(temp_dir)
    uniquified_netlist_paths = []
    for i in range(0, len(model_paths)):
        unique_global_struct_names = {}
        model_path = model_paths[i]
        
        with open(model_path, 'r') as file:
            file_content = file.read()
        
        with open(model_path, 'r') as file:
            netlist_data = yaml.load(file, Loader = yaml.FullLoader)
            for queue in netlist_data["queues"].keys():
                unique_global_struct_names[queue] = "model_" + str(i) + "_" + queue
            for graph in netlist_data["graphs"].keys():
                unique_global_struct_names[graph] = "model_" + str(i) + "_" + graph
                for op in netlist_data["graphs"][graph]:
                    if(op == "target_device" or op == "input_count"):
                        continue
                    unique_global_struct_names[op] = "model_" + str(i) + "_" + op
                         
            for program in netlist_data["programs"]:
                program_name = list(program.keys())[0]
                unique_global_struct_names[program_name] = "model_" + str(i) + "_" + program_name
            if "fused_ops" in netlist_data:
                for sched in netlist_data["fused_ops"].keys():
                    for op in netlist_data["fused_ops"][sched]["schedules"][0]:
                        op_name = list(op.keys())[0]
                        unique_global_struct_names[op_name] = "model_" + str(i) + "_" + op_name
        replacement_keys = list(unique_global_struct_names.keys())
        replacement_keys.reverse()
        for unique_key in replacement_keys:
            pattern = re.compile(r'\b' + re.escape(unique_key) + r'\b')
            file_content = pattern.sub(unique_global_struct_names[unique_key], file_content)
        
        indexed_model_path = str(model_path).split(".yaml")[0] + "_" + str(i) + ".yaml"
        base_filename = os.path.basename(indexed_model_path)
        temp_file_path = os.path.join(temp_dir, base_filename)
        
        with open(temp_file_path, "w+") as file:
            file.write(file_content)
        uniquified_netlist_paths.append(temp_file_path)
    return uniquified_netlist_paths

def merge_unique_netlists(unique_netlist_paths, overlay_blob_size_per_model):
    merged_model = {"devices" : [],
                    "queues" : {},
                    "graphs" : {},
                    "programs" : []}
    
    fused_op_counter = 0
    for (i, netlist) in enumerate(unique_netlist_paths):
        fused_op_idx_updates = {}
        with open(netlist, 'r') as file:
            netlist_data = yaml.load(file, Loader = yaml.FullLoader)
            if(i == 0):
                merged_model["devices"] = netlist_data["devices"]
            for queue in netlist_data["queues"].keys():
                merged_model["queues"][queue] = netlist_data["queues"][queue]
            for graph in netlist_data["graphs"].keys():
                for op in netlist_data["graphs"][graph]:
                    if(op == "target_device" or op == "input_count"):
                        continue
                    if netlist_data["graphs"][graph][op]["type"] == "fused_op":
                        local_id = netlist_data["graphs"][graph][op]["attributes"]["fused_op_id"]
                        if not local_id in fused_op_idx_updates:
                            fused_op_idx_updates[local_id] = fused_op_counter
                            fused_op_counter = fused_op_counter + 1
                        netlist_data["graphs"][graph][op]["attributes"]["fused_op_id"] = fused_op_idx_updates[local_id]
                    
                    if "attributes" in netlist_data["graphs"][graph][op]:
                        if "kernel_broadcast" in netlist_data["graphs"][graph][op]["attributes"]:
                            updated_kernel_bcast = {}
                            for input in netlist_data["graphs"][graph][op]["attributes"]["kernel_broadcast"]:
                                updated_kernel_bcast[input.replace("model_" + str(i) + "_", "")] = netlist_data["graphs"][graph][op]["attributes"]["kernel_broadcast"][input]
                            netlist_data["graphs"][graph][op]["attributes"]["kernel_broadcast"] = updated_kernel_bcast
                    if i in overlay_blob_size_per_model:
                        netlist_data["graphs"][graph][op]["overlay_size"] = int(overlay_blob_size_per_model[i])
                
                merged_model["graphs"][graph] = netlist_data["graphs"][graph]
            for program in netlist_data["programs"]:
                program_name = list(program.keys())[0]
                program_dict = {program_name : program[program_name]}
                merged_model["programs"].append(program_dict)
            if "fused_ops" in netlist_data:
                if not "fused_ops" in merged_model:
                    merged_model["fused_ops"] = {}
                for sched in netlist_data["fused_ops"].keys():
                    merged_model["fused_ops"][fused_op_idx_updates[sched]] = netlist_data["fused_ops"][sched]
                    for op in merged_model["fused_ops"][fused_op_idx_updates[sched]]["schedules"][0]:
                        for op_idx in range(len(op[list(op.keys())[0]]["inputs"])):
                            input_name = op[list(op.keys())[0]]["inputs"][op_idx]
                            op[list(op.keys())[0]]["inputs"][op_idx] = input_name.replace("model_" + str(i) + "_", "")
                        
                    merged_model["fused_ops"][fused_op_idx_updates[sched]]
                    
    return merged_model

def update_buffer_ranges(merged_model, queue, buf_info, buf_group_ranges):
    FRAGMENTATION_SIZE = 32
    if not buf_info[0] in buf_group_ranges:
        buf_group_ranges[buf_info[0]] = []
    for group in buf_group_ranges[buf_info[0]]:
        if buf_info[1] >= group[0] and buf_info[1] <= group[1] + FRAGMENTATION_SIZE:
            group[1] = max(group[1], buf_info[1] + get_queue_size(merged_model, queue))
            return
        elif buf_info[1] >= group[0] - get_queue_size(merged_model, queue) - FRAGMENTATION_SIZE and buf_info[1] + get_queue_size(merged_model, queue) <= group[1]:
            group[0] = min(group[0], buf_info[1])
            return
    buf_group_ranges[buf_info[0]].append([buf_info[1], buf_info[1] + get_queue_size(merged_model, queue)])
        
def get_dynamic_queue_info(merged_model):
    dynamic_queues = []
    dyn_queue_group_range = {}
    start_offset_to_queue_buf_per_model = {}
    
    for prog_idx in range(0, len(merged_model["programs"])):
        for prog in merged_model["programs"][prog_idx]:
            for instrn in merged_model["programs"][prog_idx][prog]:
                if type(instrn) == dict and list(instrn.keys())[0] == "allocate_queue":
                    for queue in list(instrn.values())[0]:
                        dynamic_queues.append(queue)
                        for alloc in merged_model["queues"][queue]["dram"]:
                            update_buffer_ranges(merged_model, queue, alloc, dyn_queue_group_range)
    
    model_name_pattern = r'model_\d+'
    for queue in dynamic_queues:
        model_name = re.search(model_name_pattern, queue).group()
        if not model_name in start_offset_to_queue_buf_per_model:
             start_offset_to_queue_buf_per_model[model_name] = {}
        for buf_idx, alloc in enumerate(merged_model["queues"][queue]["dram"]):
            buf_groups = dyn_queue_group_range[alloc[0]]
            start_offset_found = False
            for group in buf_groups:
                if alloc[1] >= group[0] and alloc[1] <= group[1]:
                    start_offset = group[0]
                    start_offset_found = True
                    break
            assert start_offset_found, "start offset not found"
            if not alloc[0] in start_offset_to_queue_buf_per_model[model_name]:
                start_offset_to_queue_buf_per_model[model_name][alloc[0]] = {}
                
            if not start_offset in start_offset_to_queue_buf_per_model[model_name][alloc[0]]:
                start_offset_to_queue_buf_per_model[model_name][alloc[0]][start_offset] = []
            start_offset_to_queue_buf_per_model[model_name][alloc[0]][start_offset].append([queue, buf_idx])
    return dynamic_queues, start_offset_to_queue_buf_per_model

def get_queue_size(netlist, queue):
    is_untilized = False
    if("layout" in netlist["queues"][queue]):
        is_untilized = (netlist["queues"][queue]["layout"] != "tilized")
    format = str_to_format(netlist["queues"][queue]["df"])
    ublock_ct = netlist["queues"][queue]["ublock"][0]
    ublock_rt = netlist["queues"][queue]["ublock"][1]
    mblock_m = netlist["queues"][queue]["mblock"][0]
    mblock_n = netlist["queues"][queue]["mblock"][1]
    t = netlist["queues"][queue]["t"]
    entries = netlist["queues"][queue]["entries"]
    tile_height = 32
    tile_width = 32
    if("tile_dim" in netlist["queues"][queue]):
        tile_height = netlist["queues"][queue]["tile_dim"][0]
        tile_width = netlist["queues"][queue]["tile_dim"][1]
    return backend_api.get_io_size_in_bytes(format, is_untilized, ublock_ct, ublock_rt, mblock_m, mblock_n, t, entries, tile_height, tile_width)
    
def reallocate_queues(merged_model, dynamic_queues, start_offset_to_queue_buf_per_model, soc_descriptor, switch_chans_if_capacity_hit, overlap_dynamic_queues):
    dev_cfg = backend_api.DeviceConfig("wormhole_b0",
                             soc_descriptor,
                             "",
                             "",
                             "",
                             False,
                             [])
    max_reserved_backend_space = dev_cfg.get_dram_backend_reserved_max()
    backend_reserved_dram_memory = {0 : max_reserved_backend_space, 1 : max_reserved_backend_space, 2 : max_reserved_backend_space, 3 : max_reserved_backend_space, 
                                   4 : max_reserved_backend_space, 5 : max_reserved_backend_space}
    memory_consumed_per_host_channel = {}
    for host_chan in range(dev_cfg.get_host_memory_num_channels()):
        memory_consumed_per_host_channel[host_chan] = dev_cfg.get_host_memory_channel_start_address()
    
    static_queue_dram_space = copy.copy(backend_reserved_dram_memory)
    MAX_DRAM_SPACE = 2**31
    
    if not switch_chans_if_capacity_hit:
        logger.warning("Memory Optimization Allowing Buffer Channels to be Reallocated is disabled")
        
    if overlap_dynamic_queues:
        # Memory optimization: Allow dynamic queues to overlap in merged netlist
        for model in start_offset_to_queue_buf_per_model:
            dram_usage_across_groups = copy.copy(backend_reserved_dram_memory)
            for chan in start_offset_to_queue_buf_per_model[model]:
                for offset in start_offset_to_queue_buf_per_model[model][chan]:
                    max_usage_per_chan = dram_usage_across_groups[chan]
                    for q_buf_pair in start_offset_to_queue_buf_per_model[model][chan][offset]:
                        queue_name = q_buf_pair[0]
                        buf_idx = q_buf_pair[1]
                        addr = merged_model["queues"][queue_name]["dram"][buf_idx][1]
                        shifted_addr = addr - (offset - max_usage_per_chan)
                        merged_model["queues"][queue_name]["dram"][buf_idx][1] = shifted_addr
                        dram_usage_across_groups[chan] = max(dram_usage_across_groups[chan], shifted_addr + get_queue_size(merged_model, queue_name))

            for chan in static_queue_dram_space:
                static_queue_dram_space[chan] = max(dram_usage_across_groups[chan], static_queue_dram_space[chan])
    else:
        logger.warning("Memory Optimization Allowing Dynamic Queues to Overlap is Disabled")
               
    for queue in merged_model["queues"]:
        if queue in dynamic_queues and overlap_dynamic_queues:
            # Dynamic queues was already allocated. Skip allocation here.
            continue
        queue_size = get_queue_size(merged_model, queue)
        if(merged_model["queues"][queue]["loc"].lower() == "dram"):
            for alloc in merged_model["queues"][queue]["dram"]:
                if static_queue_dram_space[alloc[0]] + queue_size > MAX_DRAM_SPACE:
                    if switch_chans_if_capacity_hit:
                        logger.info("DRAM Channel {} capacity hit. Bytes Used: {}. Reallocating queue to a different channel", alloc[0], static_queue_dram_space[alloc[0]])
                        for i in static_queue_dram_space:
                            if static_queue_dram_space[i] + queue_size <= MAX_DRAM_SPACE:
                                alloc[0] = i
                            
                alloc[1] = static_queue_dram_space[alloc[0]]
                static_queue_dram_space[alloc[0]] += queue_size
                assert static_queue_dram_space[alloc[0]] <= MAX_DRAM_SPACE, "DRAM space exceeded for DRAM channel " + str(alloc[0]) + " when trying to allocate memory for queue " + queue + " Bytes used: " + str(static_queue_dram_space[alloc[0]])
                static_queue_dram_space[alloc[0]] = backend_api.get_next_aligned_address(static_queue_dram_space[alloc[0]])
        else:
            for (alloc_idx, alloc) in enumerate(merged_model["queues"][queue]["host"]):
                if(type(alloc) == list):
                    # Support for new host queue layout ... multi-channel
                    if memory_consumed_per_host_channel[alloc[0]] + queue_size > dev_cfg.get_host_memory_channel_size(alloc[0]):
                        if switch_chans_if_capacity_hit:
                            logger.info("Host Channel {} capacity hit. Bytes Used: {}. Reallocating queue to a different channel", alloc[0], memory_consumed_per_host_channel[alloc[0]])
                            for i in memory_consumed_per_host_channel:
                                if memory_consumed_per_host_channel[i] + queue_size <= dev_cfg.get_host_memory_channel_size(i):
                                    alloc[0] = i
                    alloc[1] = memory_consumed_per_host_channel[alloc[0]]
                    memory_consumed_per_host_channel[alloc[0]] += queue_size
                    assert memory_consumed_per_host_channel[alloc[0]] <= dev_cfg.get_host_memory_channel_size(alloc[0]), "Host memory space exceeded for channel " + str(alloc[0]) + " when trying to allocate memory for queue " + queue + " Bytes used: " + str(memory_consumed_per_host_channel[alloc[0]])
                    memory_consumed_per_host_channel[alloc[0]] = backend_api.get_next_aligned_address(memory_consumed_per_host_channel[alloc[0]])
                else:
                    # Support for legacy host queue layout ... single channel
                    assert type(alloc) == int
                    merged_model["queues"][queue]["host"][alloc_idx] = memory_consumed_per_host_channel[0]
                    memory_consumed_per_host_channel[0] += queue_size
                    assert memory_consumed_per_host_channel[0] <= dev_cfg.get_host_memory_channel_size(0), "Host memory space exceeded for channel 0 when trying to allocate memory for queue " + queue + " Bytes used: " + str(memory_consumed_per_host_channel[0])
                    memory_consumed_per_host_channel[0] = backend_api.get_next_aligned_address(memory_consumed_per_host_channel[0])
    logger.info("Displaying memory footprint per DRAM channel (MB):")
    for chan in static_queue_dram_space:
        logger.info("{} : {}", chan, round(static_queue_dram_space[chan] / (2**20), 2))
    logger.info("Displaying memory footprint per Host channel (MB):")
    for host_chan in memory_consumed_per_host_channel:
        logger.info("{} : {}", host_chan, round(memory_consumed_per_host_channel[host_chan] / (2**20), 2))
    return merged_model

def uniquify_tensor_bin_names(unzipped_tti_paths, merged_tti_path):
    logger.info("Uniquifying parameter names for merged model...")
    for (i, tti_path) in enumerate(unzipped_tti_paths):
        tensor_path = os.path.join(tti_path, "unzipped_tti", "tensors")
        for tensor_bin in os.listdir(tensor_path):
            if os.path.isfile(os.path.join(tensor_path, tensor_bin)):
                shutil.copy(os.path.join(tensor_path, tensor_bin), 
                            os.path.join(merged_tti_path, "unzipped_tti", "tensors", "model_" + str(i) + "_" + tensor_bin))


def merge_device_metadata(unzipped_tti_paths, merged_tti_path):
    logger.info("Generating Metadata for merged model...")
    netlist_names = []
    merged_md = {
                    "compiled_graph_state" : {
                        "ordered_input_names" : [],
                        "ordered_output_names" : [],
                        "ordered_constant_node_names" : [],
                        "ordered_parameter_node_names" : [],
                        "post_const_eval_parameters" : {},
                        "post_const_eval_constants" : {},
                        "ordered_input_runtime_tensor_transforms" : [],
                        "ordered_output_runtime_tensor_transforms" : []
                    },
                    "arch" : {},
                    "devtype" : {}
                }
    
    for (i, tti_path) in enumerate(unzipped_tti_paths):
        with open(os.path.join(tti_path, "unzipped_tti", "device.json"), "r") as file:
            device_md = json.load(file)
        for name in device_md["compiled_graph_state"]["ordered_constant_node_names"]:
            merged_md["compiled_graph_state"]["ordered_constant_node_names"].append("model_" + str(i) + "_" + name)
        for name in device_md["compiled_graph_state"]["ordered_parameter_node_names"]:
            merged_md["compiled_graph_state"]["ordered_parameter_node_names"].append("model_" + str(i) + "_" + name)
        for name in device_md["compiled_graph_state"]["ordered_input_names"]:
            merged_md["compiled_graph_state"]["ordered_input_names"].append("model_" + str(i) + "_" + name)
        for name in device_md["compiled_graph_state"]["ordered_output_names"]:
            merged_md["compiled_graph_state"]["ordered_output_names"].append("model_" + str(i) + "_" + name)
        for name in device_md["compiled_graph_state"]["post_const_eval_parameters"]:
            merged_md["compiled_graph_state"]["post_const_eval_parameters"]["model_" + str(i) + "_" + name] = device_md["compiled_graph_state"]["post_const_eval_parameters"][name]
            tensor_bin = merged_md["compiled_graph_state"]["post_const_eval_parameters"]["model_" + str(i) + "_" + name]["bin"].split("/")[1]
            merged_md["compiled_graph_state"]["post_const_eval_parameters"]["model_" + str(i) + "_" + name]["bin"] = "tensors/" + "model_" + str(i) + "_" + tensor_bin
        for name in device_md["compiled_graph_state"]["post_const_eval_constants"]:
            merged_md["compiled_graph_state"]["post_const_eval_constants"]["model_" + str(i) + "_" + name] = device_md["compiled_graph_state"]["post_const_eval_constants"][name]
            tensor_file = merged_md["compiled_graph_state"]["post_const_eval_constants"]["model_" + str(i) + "_" + name]["bin"].split("/")[1]
            merged_md["compiled_graph_state"]["post_const_eval_constants"]["model_" + str(i) + "_" + name]["bin"] = "tensors/" + "model_" + str(i) + "_" + tensor_file
        for transform in device_md["compiled_graph_state"]["ordered_input_runtime_tensor_transforms"]:
            merged_md["compiled_graph_state"]["ordered_input_runtime_tensor_transforms"].append(transform)
        for transform in device_md["compiled_graph_state"]["ordered_output_runtime_tensor_transforms"]:
            merged_md["compiled_graph_state"]["ordered_output_runtime_tensor_transforms"].append(transform)
        netlist_names.append(os.path.join(tti_path, "unzipped_tti/", os.path.basename(device_md["compiled_graph_state"]["netlist_filename"])))
        merged_md["compiled_graph_state"]["netlist_filename"] = "unzipped_tti/merged_netlist.yaml"
        if(type(device_md["arch"]) == str):
            merged_md["arch"]["__enum__"] = device_md["arch"]
        else:   
            merged_md["arch"]["__enum__"] = device_md["arch"]["__enum__"]
        if(type(device_md["devtype"]) == str):
            merged_md["devtype"]["__enum__"] = device_md["devtype"]
        else:
            merged_md["devtype"]["__enum__"] = device_md["devtype"]["__enum__"]
    
    with open(os.path.join(merged_tti_path, "unzipped_tti", "device.json"), "w+") as file:
        json.dump(merged_md, file, indent = 4)

    return netlist_names

def verify_and_copy_config_json(unzipped_tti_paths, merged_tti_dir):
    logger.info("Generating Device Config for Merged Model...")
    backend_config = {}
    overlay_blob_size_per_model = {}
    for (i, tti_path) in enumerate(unzipped_tti_paths):
        config_file = os.path.join(tti_path, "unzipped_tti", "compile_and_runtime_config.json")
        with open(config_file, "r") as file:
            config = json.load(file)    
        for cfg_entry in config:
            if "TT_BACKEND_" in cfg_entry:
                if cfg_entry == "TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE":
                    overlay_blob_size_per_model[i] = config[cfg_entry]
                if not cfg_entry in backend_config:
                    # Add first entry to map
                    backend_config[cfg_entry] = config[cfg_entry]
                    continue
                
                if cfg_entry == "TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE":
                    # Get max used extra overlay size across models
                    backend_config[cfg_entry] = str(max(int(backend_config[cfg_entry]), int(config[cfg_entry])))
                elif cfg_entry == "TT_BACKEND_HARVESTED_ROWS":
                    # Require grid sizes to be consistent
                    harv_mask_bin = bin(int(config[cfg_entry]))
                    existing_harv_mask_bin = bin(int(backend_config[cfg_entry]))
                    assert harv_mask_bin.count('1') == existing_harv_mask_bin.count('1'), "SOC Grid size mismatch across models"
                else:
                    assert config[cfg_entry] == backend_config[cfg_entry], "Config var: " + str(cfg_entry) + " mismatches across models."
                
    for be_cfg in backend_config:
        os.environ[be_cfg] = str(backend_config[be_cfg])
    device_json = os.path.join(merged_tti_dir, "unzipped_tti", "compile_and_runtime_config.json")
    with open(device_json, "w+") as file:
        json.dump(backend_config, file, indent = 4)
    return overlay_blob_size_per_model
                
def unzip_ttis_and_generate_output_dir(tti_file_paths, output_tti_dir):
    logger.info("Unzipping individual TTIs...")
    os.makedirs(os.path.join(output_tti_dir, "unzipped_tti"))
    os.makedirs(os.path.join(output_tti_dir, "unzipped_tti", "tensors"))
    
    unzipped_tti_directories = []
    for tti in tti_file_paths:
        unzipped_tti_directories.append(tempfile.mkdtemp())
        unzipped_tti_directory = unzipped_tti_directories[-1]
        temp_directories.append(unzipped_tti_directory)
        sp.run(['tar', '-xf', tti, '-C', unzipped_tti_directory])
    return unzipped_tti_directories

def merge_netlists(netlist_paths, merged_tti_path, unzipped_tti_paths, overlay_blob_size_per_model, switch_chans_if_capacity_hit, overlap_dynamic_queues):
    logger.info("Merging Netlists...")
    soc_descriptor = os.path.join(unzipped_tti_paths[0], "unzipped_tti/backend_build_binaries/device_desc_runtime/0.yaml")
    if not os.path.exists(soc_descriptor):
        soc_descriptor = os.path.join(unzipped_tti_paths[0], "unzipped_tti/backend_build_binaries/device_desc.yaml")
    uniquifed_netlist =  merge_unique_netlists(uniquify_global_structures(netlist_paths), overlay_blob_size_per_model)
    dynamic_queues, start_offset_to_queue_buf_per_model = get_dynamic_queue_info(uniquifed_netlist)
    merged_model = reallocate_queues(uniquifed_netlist, dynamic_queues, start_offset_to_queue_buf_per_model, soc_descriptor, switch_chans_if_capacity_hit, overlap_dynamic_queues)
    yaml_output = yaml.dump(merged_model, default_flow_style=False, sort_keys=False)
    netlist_path = os.path.join(merged_tti_path, "unzipped_tti/merged_netlist.yaml")
    with open(netlist_path, "w+") as file:
        file.write(yaml_output)
    return netlist_path
        
def compile_backend_binaries(merged_tti_path, netlist_path):
    logger.info("Compiling TT Binaries for merged model...")
    os.makedirs(os.path.join(merged_tti_path, "unzipped_tti/backend_build_binaries/"))
    bcfg = backend_api.BackendConfig(backend_api.BackendType.Silicon,
                              backend_api.BackendDevice.Wormhole_B0,
                              backend_api.DeviceMode.CompileOnly,
                              0,
                              os.path.join(merged_tti_path, "unzipped_tti/backend_build_binaries/"),
                              "",
                              "")
    be_api = backend_api.BackendApi(netlist_path, bcfg)
    compile_result = backend_api.BackendCompileResult()
    assert be_api.initialize(compile_result) == backend_api.BackendStatusCode.Success
    
def create_merged_tti(output_loc, merged_binary_dir):
    logger.info("Packaging Binaries...")
    result = sp.run(['tar', '-cf', output_loc, "-C", merged_binary_dir, "unzipped_tti"], stdout=sp.PIPE, text=True)

def cleanup():
    logger.info("Cleaning up intermediate state and exiting")
    for dir in temp_directories:
        shutil.rmtree(dir)
            
def merge_models(model_binaries, arch, merged_model_location = "", switch_chans_if_capacity_hit = True, overlap_dynamic_queues = True):
    # Main API that gets exported to other files
    try:
        assert arch == "grayskull" or arch == "wormhole_b0", "Expected arch to be Grayskull or Wormhole_B0"
        output_loc = merged_model_location
        if not output_loc:
            output_loc = "merged_model.tti"
        
        merged_binary_dir = tempfile.mkdtemp()
        temp_directories.append(merged_binary_dir)
        unzipped_tti_paths = unzip_ttis_and_generate_output_dir(model_binaries, merged_binary_dir)
        overlay_blob_size_per_model = verify_and_copy_config_json(unzipped_tti_paths, merged_binary_dir)
        netlist_names = merge_device_metadata(unzipped_tti_paths, merged_binary_dir)
        uniquify_tensor_bin_names(unzipped_tti_paths, merged_binary_dir)
        merged_netlist_path = merge_netlists(netlist_names, merged_binary_dir, unzipped_tti_paths, overlay_blob_size_per_model, switch_chans_if_capacity_hit, overlap_dynamic_queues)
        compile_backend_binaries(merged_binary_dir, merged_netlist_path)
        create_merged_tti(output_loc, merged_binary_dir)
        logger.info("Binaries for the merged model are stored in: " + output_loc)
        logger.info("Done!")
        cleanup()
    except Exception as e:
        logger.exception(e)
        cleanup()
    
if __name__ == "__main__":
    # Interface to run tool directly
    parser =  argparse.ArgumentParser()
    parser.add_argument("--model_binaries", type = str, help = "List of model binaries (tti files) to merge.", required = True, nargs = "*")
    parser.add_argument("--arch", type = str, help = "Target TT architecture.", default="wormhole_b0")
    parser.add_argument("--merged_model_location", type = str, help = "Filesystem location where the merged model binaries are stored.")
    parser.add_argument("--skip_channel_reallocation", type = bool, help = "Skip memory usage optimization that reallocates buffers on different DRAM channels, once channel capacity is hit.", default = False)
    parser.add_argument("--dynamic_queue_overlap_off", type = bool, help = "Turn off memory usage optimization that overlaps dynamic queues", default = False)
    args = parser.parse_args()
    merge_models(args.model_binaries, args.arch.lower(), args.merged_model_location, not args.skip_channel_reallocation, not args.dynamic_queue_overlap_off)
