# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import re
import json
from collections import OrderedDict
import yaml
import os
from pybuda._C import DataFormat

def convert_data_format(str_format):
    if str_format is None:
        return None
    elif str_format == "Bfp4":
        return DataFormat.Bfp4
    elif str_format == "Bfp8":
        return DataFormat.Bfp8
    elif str_format == "Float16":
        return DataFormat.Float16
    elif str_format == "Float32":
        return DataFormat.Float32
    
    assert(False)


def manual_placer(config, filename, loops=1, non_fuse_offset=0):
    """ Parses file with the following format:
    {
        "epoch_breaks": [OP_NAME, ...],
        "chip_breaks": [OP_NAME, ...],
        "op_sizes": {
            OP_NAME: [ROW, COL],
            ...
        },
        "grid_locations": {
            OP_NAME: [START_ROW, START_COL],
            ...
        },
        "nop_buffer_insertions":{
            SRC_OP_OR_QUEUE_NAME: {"dest": DEST_OP_NAME, "hoist_tms": HOIST_TMS,
                "op_size": [ROWS, COLS], "grid_location": {"location":[START_ROW, START_COL]},
                "add_schedule_constraint": OP_NAME},
            ...
        },
        "insert_dram_queues": [ [SRC_OP_NAME, DEST_OP_NAME], ... ],
        "default_loop_offset": [53],
        "loop_offset_override": {
            "fused_blah_op": [5],
            "buffer_blah_1_fused.1_blah_3": [53, 0, 5],
            ...
        }
    }
    and uses config.override_op_size, config.override_op_placement,
    config.insert_nop, config.add_schedule_constraint,
    config.set_chip_break, config.set_epoch_break and INSERT_DRAM_QUEUES
    env variable. Loop offsets are used to apply the same placement to
    multiple epochs from a single file. YMMV. See example_placement.json.
    """
    with open(filename, "r") as f:
        text = f.read()
        data = json.loads(text)

    insert_dram_queues_env_variable = ""

    for loop in range(loops):
        if 'op_sizes' in data:
            for base_op_name, size in data["op_sizes"].items():
                op_name = apply_loop_offset(base_op_name, data, loop, non_fuse_offset)
                if op_name is None:
                    continue
                config.override_op_size(op_name, size)

        if 'grid_locations' in data:
            for base_op_name, location_attributes in data["grid_locations"].items():
                op_name = apply_loop_offset(base_op_name, data, loop, non_fuse_offset)
                if op_name is None:
                    continue

                location = tuple(location_attributes["location"])
                try:
                    transpose = location_attributes["transpose"]
                except KeyError:
                    transpose = False
                print("Override placement for op: {}, location: {}, transpose: {}".format(op_name, location, transpose))
                config.override_op_placement(op_name, start=location, transpose_op=transpose)

        if 'nop_buffer_insertions' in data:
            for base_op_name_src, buffer_attributes in data["nop_buffer_insertions"].items():
                op_name_src = apply_loop_offset(base_op_name_src, data, loop, non_fuse_offset)

                base_op_names_dest = buffer_attributes["dest"]
                if not isinstance(base_op_names_dest, list):
                    base_op_names_dest = [base_op_names_dest]

                op_names_dest = []
                for base_op_name_dest in base_op_names_dest:
                    op_name_dest = apply_loop_offset(base_op_name_dest, data, loop, non_fuse_offset)
                    op_names_dest.append(op_name_dest)
                
                if op_name_src is None or any(op_name_dest is None for op_name_dest in op_names_dest):
                    print("INFO: Skipping nop buffer insertion {} - {}, loop: {}".format(base_op_name_src, base_op_name_dest, loop))
                    continue

                buffer_node_name = "buffer_0_{}_{}".format(op_name_src, op_names_dest[0])

                size = buffer_attributes.get("op_size")
                hoist_tms = buffer_attributes.get("hoist_tms", False)
                add_schedule_constraint = buffer_attributes.get("add_schedule_constraint")
                grid_location_properties = buffer_attributes.get("grid_location")
                if grid_location_properties is not None:
                    location = grid_location_properties.get("location")
                    transpose = grid_location_properties.get("transpose", False)

                print("Insert nop buffer between {} and {}, hoist_tms: {}".format(op_name_src, op_name_dest, hoist_tms))
                config.insert_nop(op_name_src, op_names_dest, hoist_tms=hoist_tms)

                if size is not None:
                    config.override_op_size(buffer_node_name, size)

                if grid_location_properties is not None and location is not None:
                    config.override_op_placement(buffer_node_name, start=location, transpose_op=transpose)

                if add_schedule_constraint is not None:
                    add_schedule_constraint_name = apply_loop_offset(add_schedule_constraint, data, loop, non_fuse_offset)
                    print("Add schedule constraint: {} must be scheduled before {}".format(add_schedule_constraint_name, buffer_node_name))
                    config.add_schedule_constraint([add_schedule_constraint_name, buffer_node_name])

        if 'daisy_chain_nop_buffer_insertions' in data:
            for op_name_src, buffer_attributes in data["daisy_chain_nop_buffer_insertions"].items():

                base_op_name_dest = buffer_attributes["dest"]
                if isinstance(base_op_name_dest, list):
                    assert(False)

                # generate a list of all corresponding dest ops for all encoders
                all_op_names_dest = []
                for dest_loop in range(loops):
                    op_name_dest = apply_loop_offset(base_op_name_dest, data, dest_loop, non_fuse_offset)
                    if op_name_dest is None:
                        print("INFO: Skipping dest op {} - {}, loop: {} in daisy chain nop insertion".format(op_name_src, base_op_name_dest, dest_loop))
                        continue
                    all_op_names_dest.append(op_name_dest)

                size = buffer_attributes.get("op_size")
                hoist_tms = buffer_attributes.get("hoist_tms", False)
                add_schedule_constraint = buffer_attributes.get("add_schedule_constraint")
                grid_location_properties = buffer_attributes.get("grid_location")
                if grid_location_properties is not None:
                    location = grid_location_properties.get("location")
                    transpose = grid_location_properties.get("transpose", False)

                # build correct src op (buffer) name for current loop
                buffer_name = op_name_src
                for dest_loop in range(loop):
                    buffer_name = "buffer_0_{}_{}".format(buffer_name, all_op_names_dest[dest_loop])

                op_names_dest = all_op_names_dest[loop:]

                print("Insert daisy chain nop buffer between {} and {}, hoist_tms: {}".format(buffer_name, op_names_dest, hoist_tms))
                config.insert_nop(buffer_name, op_names_dest, hoist_tms=hoist_tms)

                new_buffer_name = "buffer_0_{}_{}".format(buffer_name, op_names_dest[0])

                if size is not None:
                    config.override_op_size(new_buffer_name, size)

                if grid_location_properties is not None and location is not None:
                    config.override_op_placement(new_buffer_name, start=location, transpose_op=transpose)

                if add_schedule_constraint is not None:
                    add_schedule_constraint_name = apply_loop_offset(add_schedule_constraint, data, loop, non_fuse_offset)
                    print("Add schedule constraint: {} must be scheduled before {}".format(add_schedule_constraint_name, new_buffer_name))
                    config.add_schedule_constraint([add_schedule_constraint_name, new_buffer_name])

                insert_dram_queue = buffer_attributes.get("insert_dram_queue", False)
                if insert_dram_queue and loop >= 1:
                    print("Insert dram queue between {} and {}".format(buffer_name, new_buffer_name))
                    insert_dram_queues_env_variable += "{}-{},".format(buffer_name, new_buffer_name)

        if 'epoch_breaks' in data:
            pass
            for base_op_name in data["epoch_breaks"]:
                op_name = apply_loop_offset(base_op_name, data, loop, non_fuse_offset)
                if op_name is None:
                    continue

                print("Insert epoch break at {}".format(op_name))
                config.set_epoch_break(op_name)

        if 'chip_breaks' in data:
            for base_op_name in data["chip_breaks"]:
                op_name = apply_loop_offset(base_op_name, data, loop, non_fuse_offset)
                if op_name is None:
                    continue
                config.set_chip_break(op_name)

        if 'insert_dram_queues' in data:
            for base_op_name_src, base_op_name_dest in data["insert_dram_queues"]:
                op_name_src = apply_loop_offset(base_op_name_src, data, loop, non_fuse_offset)
                op_name_dest = apply_loop_offset(base_op_name_dest, data, loop, non_fuse_offset)
                if op_name_src is None or op_name_dest is None:
                    continue
                print("Insert dram queue between {} and {}".format(op_name_src, op_name_dest))
                insert_dram_queues_env_variable += "{}-{},".format(op_name_src, op_name_dest)
        if 'data_format' in data:
            for base_op_name, data_format_properties in data["data_format"].items():
                op_name = apply_loop_offset(base_op_name, data, loop, non_fuse_offset)
                if op_name is None:
                    continue

                op_type = data_format_properties.get("op_type")

                epoch_type = data_format_properties.get("epoch_type")
                
                output_df = data_format_properties.get("output_df")
                output_df = convert_data_format(output_df)
                
                intermediate_df = data_format_properties.get("intermediate_df")
                intermediate_df = convert_data_format(intermediate_df)
                
                accumulate_df = data_format_properties.get("accumulate_df")
                accumulate_df = convert_data_format(accumulate_df)
                
                str_input_dfs_dict = data_format_properties.get("input_df")
                input_df = {}
                if str_input_dfs_dict is not None:
                    for idx in str_input_dfs_dict.keys():
                        str_in_df = str_input_dfs_dict[idx]
                        in_df = convert_data_format(str_in_df)
                        input_df[idx] = in_df

                print("Setting MP config for {} to:".format(op_name))
                print(data_format_properties)
                config.configure_mixed_precision(op_type=op_type,epoch_type=epoch_type, output_df=output_df,
                    intermediate_df=intermediate_df, accumulate_df=accumulate_df, name_regex=op_name, input_df=input_df)
    
    if insert_dram_queues_env_variable != "":
        insert_dram_queues_env_variable = insert_dram_queues_env_variable[:-1]
        env_export = "INSERT_DRAM_QUEUES={}".format(insert_dram_queues_env_variable)
    else:
        env_export = "INSERT_DRAM_QUEUES=\"\""
    print(env_export)
    with open("insert_dram_queues_env_variable.log", "w") as f:
        f.write(env_export)
    os.environ['INSERT_DRAM_QUEUES'] = insert_dram_queues_env_variable
            





def apply_loop_offset(base_op_name, data, loop_number, non_fuse_offset=0):
    """ Apply a loop offset to an op name, e.g. change ('matmul_42', 17) to 'matmul_59')
        default_loop_offset is applied if the name is not found in loop_offset_override
        If no offsets are provided, the name is returned unchanged.
    """
    offset = data.get("loop_offset_override", {}).get(base_op_name, data.get("default_loop_offset", 0))
    if offset == 0:
        return base_op_name

    if isinstance(offset, int):
        offset = [offset]

    # Each offset is a list of integer offests, one for each number found in the name
    # e.g. for "buffer_0_layernorm_38.dc.subtract.1__fused_op_3" and offset [0, 53, 5]
    # the offsets are applied to the numbers 0, 38, and 3 respectively
    # The result is "buffer_0_layernorm_91.dc.subtract.1__fused_op_8"

    # Find all numbers in the name
    numbers = [ int(x) for x in re.findall(r'(\d+)', base_op_name) ]
    # Apply the offsets
    nfo = non_fuse_offset if not base_op_name.startswith('_fused_op_') else 0
    try:
        numbers = [int(n) + nfo + offset[i] * loop_number for i, n in enumerate(numbers)]
    except:
        a = 12

    # Replace the numbers in the name
    op_name = re.sub(r'(\d+)', '{}', base_op_name).format(*['%d' % n for n in numbers])

    # check if op_name valid
    if 'last_valid_op_number' in data:
        last_valid_op_number = data.get("last_valid_op_number", {}).get(base_op_name, None)
        if last_valid_op_number is not None and any(i > last_valid_op_number for i in numbers):
            print("INFO: {} is not a valid op name, skipping placement.".format(op_name))
            return None

    return op_name


def generate_placement(netlist, filename):
    """ Generate a placement file using the op names and sizes from the netlist.
        Manual epoch breaks are also generated to match the original layout.
    """
    data = {
        "epoch_breaks": [],
        "op_sizes": OrderedDict()
    }

    with open(netlist, 'r') as f:
        netlist = yaml.safe_load(f)
        first_graph = True
        for graph in netlist['graphs'].values():
            first_op_in_graph = False if first_graph else True
            for op_name, op_params in graph.items():
                if isinstance(op_params, dict) and 'grid_size' in op_params:
                    data['op_sizes'][op_name] = op_params['grid_size']
                    if first_op_in_graph:
                        data['epoch_breaks'].append(op_name)
                        first_op_in_graph = False
            first_graph = False
    write_pretty_placement(filename, data)


def write_pretty_placement(filename, placement):
    with open(filename, "w") as f:
        text = json.dumps(placement)
        text = text.replace('{', '{\n    ')
        text = text.replace('}', '\n    }')
        text = text.replace('], ', '],\n        ')
        text = text.replace('        "op_sizes": {\n    ', '    "op_sizes": {\n        ')
        text = text.replace('    }\n    }', '    }\n}\n')
        f.write(text)


def generate_maximum_placement(legal_filename, output_filename):
    """ Read a legal placement file and generate a placement file with the maximum size for each op.
    """
    with open(legal_filename, "r") as f:
        data = json.load(f)

    # format of legal_grids.json is "matmul_2": [ {"r": 2, "c": 4} ],
    placement = {
        "epoch_breaks": [],
        "op_sizes": OrderedDict()
    }

    for op_name, sizes in data.items():
        placement[op_name] = sizes[-1]["r"], sizes[-1]["c"]

    write_pretty_placement(output_filename, placement)
