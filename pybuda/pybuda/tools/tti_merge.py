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
    
def uniquify_and_merge_netlists(model_paths, inter_model_connections, consumer_to_producers_map, overlay_size_per_model):
    temp_dir = tempfile.mkdtemp()
    temp_directories.append(temp_dir)
    
    consumer_inputs = list(inter_model_connections.keys())
    producer_outputs = list(inter_model_connections.values())
    producer_queue_shapes = {}
    producer_data_format= {}
    producer_tile_dims = {}
    model_input_counts = {}
    fused_op_counter = 0
    merged_model = {"devices" : [],
                    "queues" : {},
                    "graphs" : {},
                    "programs" : []}
    # Track if the IO queues specified in the dependency list are actually present in the netlist
    sub_graph_nodes_visited = {}
    for queue in consumer_inputs + producer_outputs:
        sub_graph_nodes_visited[queue] = False
    
    for i in range(0, len(model_paths)):
        unique_queue_names = {}
        unique_graph_names = {}
        unique_op_names = {}
        fused_op_idx_updates = {}
        
        model_path = model_paths[i]
        with open(model_path, 'r') as file:
            netlist_data = yaml.load(file, Loader = yaml.FullLoader)
            merged_model["devices"] = netlist_data["devices"]
            for queue in netlist_data["queues"].keys():
                updated_queue_name = "model_" + str(i) + "_" + queue
                if updated_queue_name in sub_graph_nodes_visited:
                    # This queue was specified as a model-to-model queue in the dependency list
                    # Mark it as visited, since it was found
                    sub_graph_nodes_visited[updated_queue_name] = True
                    # Keep track of queue parameters for queues feeding downstream models
                    if updated_queue_name in producer_outputs:
                        num_tiles_y = netlist_data["queues"][queue]["ublock"][0] * netlist_data["queues"][queue]["mblock"][0] * netlist_data["queues"][queue]["grid_size"][0]
                        num_tiles_x = netlist_data["queues"][queue]["ublock"][1] * netlist_data["queues"][queue]["mblock"][1] * netlist_data["queues"][queue]["grid_size"][1]
                        producer_queue_shapes[updated_queue_name] = [netlist_data["queues"][queue]["entries"], netlist_data["queues"][queue]["t"], num_tiles_y, num_tiles_x]
                        producer_data_format[updated_queue_name] = netlist_data["queues"][queue]["df"]
                        if "tile_dim" in netlist_data["queues"][queue]["tile_dim"]:
                            producer_tile_dims[updated_queue_name] = netlist_data["queues"][queue]["tile_dim"]
                        else:
                            producer_tile_dims[updated_queue_name] = [32, 32]
                    
                if updated_queue_name in inter_model_connections:
                    # This queue is being tied to a queue from a previous model.
                    # Alias this queue with the feeder after ensuring that the producer and consumer are compatible.
                    # Since this queue is aliased with its producer queue (which has already been added to the merged netlist) don't add this queue again
                    num_tiles_y = netlist_data["queues"][queue]["ublock"][0] * netlist_data["queues"][queue]["mblock"][0] * netlist_data["queues"][queue]["grid_size"][0]
                    num_tiles_x = netlist_data["queues"][queue]["ublock"][1] * netlist_data["queues"][queue]["mblock"][1] * netlist_data["queues"][queue]["grid_size"][1]
                    consumer_queue_shape = [netlist_data["queues"][queue]["entries"], netlist_data["queues"][queue]["t"], num_tiles_y, num_tiles_x]
                    tile_dim = [32, 32]
                    if "tile_dim" in netlist_data["queues"][queue]["tile_dim"]:
                        tile_dim = netlist_data["queues"][queue]["tile_dim"]
                    assert consumer_queue_shape == producer_queue_shapes[inter_model_connections[updated_queue_name]], "Consumer " + queue + " shape is incompatible with the producer."
                    assert netlist_data["queues"][queue]["df"] == producer_data_format[inter_model_connections[updated_queue_name]], "Consumer " + queue + " data format is incompatible with the producer."
                    assert tile_dim == producer_tile_dims[inter_model_connections[updated_queue_name]], "Consumer " + queue + " tile dimensions are incompatible with the producer."
                    updated_queue_name = inter_model_connections[updated_queue_name]
                else:
                    # This queue is not tied to a queue from a previous model. Add an unaliased version of it to the merged netlist.
                    input_name = netlist_data["queues"][queue]["input"]
                    # Queues can only be fed by ops in the sane model or by host.
                    updated_input_name = "HOST" if (input_name.lower() == "host") else "model_" + str(i) + "_" + input_name    
                    merged_model["queues"][updated_queue_name] = netlist_data["queues"][queue]
                    merged_model["queues"][updated_queue_name]["input"] = updated_input_name
                # Track the updated queue name, for modifying graph and program structures.
                unique_queue_names[queue] = updated_queue_name
                
                
            for graph in netlist_data["graphs"].keys():
                updated_graph_name = "model_" + str(i) + "_" + graph
                unique_graph_names[graph] = updated_graph_name
                merged_model["graphs"][updated_graph_name] = {}
                for op in netlist_data["graphs"][graph]:
                    if(op == "target_device" or op == "input_count"):
                        if (op == "input_count"):
                            model_input_counts["model_" + str(i)] = netlist_data["graphs"][graph][op]
                            if "model_" + str(i) in consumer_to_producers_map:
                                # Model has producers
                                for producer in consumer_to_producers_map["model_" + str(i)]:
                                    assert netlist_data["graphs"][graph][op] == model_input_counts[producer], "The microbatch sizes across producers and consumers are not consistent."

                        merged_model["graphs"][updated_graph_name][op] = netlist_data["graphs"][graph][op]
                        
                    else:
                        for input_idx  in range(len(netlist_data["graphs"][graph][op]["inputs"])):
                            if netlist_data["graphs"][graph][op]["inputs"][input_idx] in unique_queue_names:
                                netlist_data["graphs"][graph][op]["inputs"][input_idx] = unique_queue_names[netlist_data["graphs"][graph][op]["inputs"][input_idx]]
                            elif netlist_data["graphs"][graph][op]["inputs"][input_idx] in unique_op_names:
                                netlist_data["graphs"][graph][op]["inputs"][input_idx] = unique_op_names[netlist_data["graphs"][graph][op]["inputs"][input_idx]]
                            else:
                                assert False, "Input to op " + op + " is not another op or a queue."
                        
                        if i in overlay_size_per_model:
                            netlist_data["graphs"][graph][op]["overlay_size"] = int(overlay_size_per_model[i])
                        
                        if netlist_data["graphs"][graph][op]["type"] == "fused_op":
                            local_id = netlist_data["graphs"][graph][op]["attributes"]["fused_op_id"]
                            if not local_id in fused_op_idx_updates:
                                fused_op_idx_updates[local_id] = fused_op_counter
                                fused_op_counter = fused_op_counter + 1
                            netlist_data["graphs"][graph][op]["attributes"]["fused_op_id"] = fused_op_idx_updates[local_id]
                        updated_op_name = "model_" + str(i) + "_" + op
                        unique_op_names[op] = updated_op_name
                        merged_model["graphs"][updated_graph_name][updated_op_name] = netlist_data["graphs"][graph][op]
            
            for prog_idx, program in enumerate(netlist_data["programs"]):
                program_name = list(program.keys())[0]
                updated_program_name = "model_" + str(i) + "_" + program_name
                for instrn_idx, instrn in enumerate(netlist_data["programs"][prog_idx][program_name]):
                    if(type(instrn) == dict):
                        instrn_code = list(instrn.keys())[0]
                        if instrn_code == "execute":
                            netlist_data["programs"][prog_idx][program_name][instrn_idx][instrn_code]["graph_name"] = unique_graph_names[netlist_data["programs"][prog_idx][program_name][instrn_idx][instrn_code]["graph_name"]]
                            queue_settings = netlist_data["programs"][prog_idx][program_name][instrn_idx][instrn_code]["queue_settings"]
                            netlist_data["programs"][prog_idx][program_name][instrn_idx][instrn_code]["queue_settings"] = {}
                            for queue in queue_settings:
                                updated_queue = unique_queue_names[queue]
                                netlist_data["programs"][prog_idx][program_name][instrn_idx][instrn_code]["queue_settings"][updated_queue] = queue_settings[queue]
                        if instrn_code == "allocate_queue" or instrn_code == "deallocate_queue":
                            for queue_idx in range(len(netlist_data["programs"][prog_idx][program_name][instrn_idx][instrn_code])):
                                queue_name = netlist_data["programs"][prog_idx][program_name][instrn_idx][instrn_code][queue_idx]
                                netlist_data["programs"][prog_idx][program_name][instrn_idx][instrn_code][queue_idx] = unique_queue_names[queue_name]
            
                merged_model["programs"].append({updated_program_name : netlist_data["programs"][prog_idx][program_name]})

            if "fused_ops" in netlist_data:
                if not "fused_ops" in merged_model:
                    merged_model["fused_ops"] = {}
                for group in netlist_data["fused_ops"].keys():
                    schedules = netlist_data["fused_ops"][group]["schedules"]
                    netlist_data["fused_ops"][group]["schedules"] = []
                    for sched_idx in range(len(schedules)):
                        netlist_data["fused_ops"][group]["schedules"].append([])
                        for op_idx, op in enumerate(schedules[sched_idx]):
                            updated_op_name = "model_" + str(i) + "_" + list(op.keys())[0]
                            netlist_data["fused_ops"][group]["schedules"][sched_idx].append({updated_op_name : schedules[sched_idx][op_idx][list(op.keys())[0]]})
                    merged_model["fused_ops"][fused_op_idx_updates[group]] = netlist_data["fused_ops"][group]
        
    # Assert if the queues specified in the dependency list are not found in the appropriate netlists
    for queue in sub_graph_nodes_visited:
        assert sub_graph_nodes_visited[queue], "Queue " + queue + " was specified in the dependency list but was not found in any netlist."    
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
    format = backend_api.get_format_from_string(netlist["queues"][queue]["df"])
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
    
def reallocate_queues(arch, merged_model, dynamic_queues, start_offset_to_queue_buf_per_model, soc_descriptor, switch_chans_if_capacity_hit, overlap_dynamic_queues):
    dev_cfg = backend_api.DeviceConfig(arch,
                             soc_descriptor,
                             "",
                             "",
                             "",
                             False,
                             [])
    
    max_reserved_backend_space = dev_cfg.get_dram_backend_reserved_max()
    # Constants derived from the SOC descriptor. These will be unchanged for the arch.
    max_dram_space = dev_cfg.get_dram_channel_capacity()
    backend_reserved_dram_memory = {}
    memory_consumed_per_host_channel = {}
    
    for chan in range(dev_cfg.get_dram_num_channels()):
        backend_reserved_dram_memory[chan] = max_reserved_backend_space
        
    for host_chan in range(dev_cfg.get_host_memory_num_channels()):
        memory_consumed_per_host_channel[host_chan] = dev_cfg.get_host_memory_channel_start_address()
    
    static_queue_dram_space = copy.copy(backend_reserved_dram_memory)
    
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
                if static_queue_dram_space[alloc[0]] + queue_size > max_dram_space:
                    if switch_chans_if_capacity_hit:
                        logger.info("DRAM Channel {} capacity hit. Bytes Used: {}. Reallocating queue to a different channel", alloc[0], static_queue_dram_space[alloc[0]])
                        for i in static_queue_dram_space:
                            if static_queue_dram_space[i] + queue_size <= max_dram_space:
                                alloc[0] = i
                            
                alloc[1] = static_queue_dram_space[alloc[0]]
                static_queue_dram_space[alloc[0]] += queue_size
                assert static_queue_dram_space[alloc[0]] <= max_dram_space, "DRAM space exceeded for DRAM channel " + str(alloc[0]) + " when trying to allocate memory for queue " + queue + " Bytes used: " + str(static_queue_dram_space[alloc[0]])
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


def merge_device_metadata(unzipped_tti_paths, merged_tti_path, inter_model_connections):
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
    
    intermediate_inputs = set()
    intermediate_outputs = set()
    
    for connection in inter_model_connections:
        intermediate_inputs.add(connection)
        intermediate_outputs.add(inter_model_connections[connection])
        
    for (i, tti_path) in enumerate(unzipped_tti_paths):
        with open(os.path.join(tti_path, "unzipped_tti", "device.json"), "r") as file:
            device_md = json.load(file)
        for name in device_md["compiled_graph_state"]["ordered_constant_node_names"]:
            merged_md["compiled_graph_state"]["ordered_constant_node_names"].append("model_" + str(i) + "_" + name)
        for name in device_md["compiled_graph_state"]["ordered_parameter_node_names"]:
            merged_md["compiled_graph_state"]["ordered_parameter_node_names"].append("model_" + str(i) + "_" + name)
        for name in device_md["compiled_graph_state"]["ordered_input_names"]:
            if not "model_" + str(i) + "_" + name in intermediate_inputs:
                merged_md["compiled_graph_state"]["ordered_input_names"].append("model_" + str(i) + "_" + name)
        for name in device_md["compiled_graph_state"]["ordered_output_names"]:
            if not "model_" + str(i) + "_" + name in intermediate_outputs:
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

def merge_netlists(arch, netlist_paths, merged_tti_path, unzipped_tti_paths, overlay_blob_size_per_model, switch_chans_if_capacity_hit, overlap_dynamic_queues, inter_model_connections, consumer_to_producers_map):
    logger.info("Merging Netlists...")

    wh_soc_desc_dir = os.path.join(unzipped_tti_paths[0], "unzipped_tti/backend_build_binaries/device_desc_runtime")
    gs_golden_soc_desc_dir = os.path.join(unzipped_tti_paths[0], "unzipped_tti/backend_build_binaries/device_descs")
    soc_descriptor = ""
    
    soc_desc_dir = wh_soc_desc_dir # Expect files in WH device desc location
    if not os.path.exists(wh_soc_desc_dir):
        # If WH device desc dir does not exist, check GS silicon or Golden (All archs) location
        soc_desc_dir = gs_golden_soc_desc_dir
    # If device desc dir does not exist, set it to default device_descs.yaml
    if os.path.exists(soc_desc_dir):
        soc_desc_files = os.listdir(soc_desc_dir)
        if len(soc_desc_files):
            soc_descriptor = os.path.join(soc_desc_dir, soc_desc_files[0])
    
    if not soc_descriptor:
        soc_descriptor = os.path.join(unzipped_tti_paths[0], "unzipped_tti/backend_build_binaries/device_desc.yaml")
    
    assert(os.path.exists(soc_descriptor), "Could not find SOC Descriptor in Unzipped TTI Files")
    
    merged_netlist = uniquify_and_merge_netlists(netlist_paths, inter_model_connections, consumer_to_producers_map, overlay_blob_size_per_model)
    dynamic_queues, start_offset_to_queue_buf_per_model = get_dynamic_queue_info(merged_netlist)
    merged_model = reallocate_queues(arch, merged_netlist, dynamic_queues, start_offset_to_queue_buf_per_model, soc_descriptor, switch_chans_if_capacity_hit, overlap_dynamic_queues)

    yaml_output = yaml.dump(merged_model, default_flow_style=False, sort_keys=False)
    netlist_path = os.path.join(merged_tti_path, "unzipped_tti/merged_netlist.yaml")
    with open(netlist_path, "w+") as file:
        file.write(yaml_output)
    return netlist_path
        
def compile_backend_binaries(arch, merged_tti_path, netlist_path):
    logger.info("Compiling TT Binaries for merged model...")
    os.makedirs(os.path.join(merged_tti_path, "unzipped_tti/backend_build_binaries/"))
    bcfg = backend_api.BackendConfig(backend_api.BackendType.Silicon,
                              backend_api.BackendDevice.from_string(arch.capitalize()),
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
        
def check_model_dep_constraints(models, dep_list):
    for model in dep_list:
        for input in dep_list[model]["inputs"]:
            if type(dep_list[model]["inputs"][input]) == str:
                assert dep_list[model]["inputs"][input].lower() == "host", "If input for model " + str(model) + " is not host, the feeder must be specified in format [feeder_model_name, feeder_queue_name]."
            else:
                assert type(dep_list[model]["inputs"][input]) == list, "The feeder for model " + str(model) + " must be specified in format [feeder_model_name, feeder_queue_name]."
                assert dep_list[model]["inputs"][input][0] in models, "Feeder model " + str(dep_list[model]["inputs"][input][0]) + " to consumer model " + model + " is not specified."
                
def parse_model_deps(models, dependency_list_file):
    ordered_models = []
    model_connections = {}
    model_name_remap = {}
    consumer_to_producers_map = {}
    with open(dependency_list_file, "r") as dep_file:
        dep_list = yaml.load(dep_file, Loader = yaml.FullLoader)
    
    check_model_dep_constraints(models, dep_list)
    while len(models) != len(ordered_models):
        for model in models:
            if not model in ordered_models:
                if not model in dep_list:
                    logger.warning("Could not find model {} in dependency list. Assuming that this model is fed by host.", model)
                    ordered_models.append(model)
                else:
                    feeders_found = True
                    for input in dep_list[model]["inputs"]:
                        if not type(dep_list[model]["inputs"][input]) == str:
                            feeder_model = dep_list[model]["inputs"][input][0]
                            feeders_found = feeder_model in ordered_models
                    if feeders_found:
                        ordered_models.append(model)
                        
    for model_idx in range(len(ordered_models)):
        model_name_remap[ordered_models[model_idx]] = "model_" + str(model_idx) 
        if ordered_models[model_idx] in dep_list:
            for input in dep_list[ordered_models[model_idx]]["inputs"]:
                if type(dep_list[ordered_models[model_idx]]["inputs"][input]) == str:
                    continue
                feeder_model = model_name_remap[dep_list[ordered_models[model_idx]]["inputs"][input][0]]
                feeder_queue = dep_list[ordered_models[model_idx]]["inputs"][input][1]
                model_connections["model_" + str(model_idx) + "_" + input] = feeder_model + "_" + feeder_queue
                if not "model_" + str(model_idx) in consumer_to_producers_map:
                    consumer_to_producers_map["model_" + str(model_idx)] = []
                consumer_to_producers_map["model_" + str(model_idx)].append(feeder_model)
    return ordered_models, model_connections, consumer_to_producers_map

def merge_models(model_bin_location, models, arch = "wormhole_b0", merged_model_location = "", switch_chans_if_capacity_hit = True,
                 overlap_dynamic_queues = True):       
    # Main API that gets exported to other files
    try:
        assert arch == "grayskull" or arch == "wormhole_b0", "Expected arch to be grayskull or wormhole_b0"
        output_loc = merged_model_location
        if not output_loc:
            output_loc = "merged_model.tti"
        merged_binary_dir = tempfile.mkdtemp()
        temp_directories.append(merged_binary_dir)
        
        # Parse dependency file, topologically sort models, and infer connections
        ordered_models = models
        inter_model_connections = {}
        consumer_to_producers_map = {}
        dependency_file = "" # Explicitly set dependency file to empty, since we don't have compiler support for pipelined models
        if dependency_file:
            ordered_models, inter_model_connections, consumer_to_producers_map = parse_model_deps(models, dependency_file)
        
        model_binaries = [os.path.join(model_bin_location, x + ".tti") for x in ordered_models]
        unzipped_tti_paths = unzip_ttis_and_generate_output_dir(model_binaries, merged_binary_dir)
        overlay_blob_size_per_model = verify_and_copy_config_json(unzipped_tti_paths, merged_binary_dir)
        netlist_names = merge_device_metadata(unzipped_tti_paths, merged_binary_dir, inter_model_connections)
        uniquify_tensor_bin_names(unzipped_tti_paths, merged_binary_dir)
        merged_netlist_path = merge_netlists(arch, netlist_names, merged_binary_dir, unzipped_tti_paths, overlay_blob_size_per_model, 
                                             switch_chans_if_capacity_hit, overlap_dynamic_queues, inter_model_connections, consumer_to_producers_map)
        compile_backend_binaries(arch, merged_binary_dir, merged_netlist_path)
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
    parser.add_argument("--model_binaries_location", "-mbl", type=str, help="Location of model binaries (tti files) to merge.", required=True)
    parser.add_argument("--models", "-mdl", type=str, help="List of models to merge", required=True, nargs="*")
    # Disable passing in dependency files for now, since we don't have compiler support for pipelined models
    # parser.add_argument("--dependency_file", "-df", type=str, help="YAML file describing IO dependencies between models")
    parser.add_argument("--arch", "-a", type=str, help="Target TT architecture.", default="wormhole_b0")
    parser.add_argument("--merged_model_location", "-mml", type=str, help="Filesystem location where the merged model binaries are stored.")
    parser.add_argument("--skip_channel_reallocation", "-scr", type=bool, help="Skip memory usage optimization that reallocates buffers on different DRAM channels, once channel capacity is hit.", default=False)
    parser.add_argument("--dynamic_queue_overlap_off", "-dqo", type=bool, help="Turn off memory usage optimization that overlaps dynamic queues", default=False)
    args = parser.parse_args()
    
    merge_models(args.model_binaries_location,
                 args.models,
                 args.arch.lower(),
                 args.merged_model_location,
                 not args.skip_channel_reallocation, 
                 not args.dynamic_queue_overlap_off)
