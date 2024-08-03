#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import subprocess
import sys


def style(string, color=None, bold=False):
    color_codes = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "purple": "35",
        "cyan": "36",
    }
    escape = "\033["
    clr = f";{color_codes[color]}" if color is not None else ""
    fmt = f"{escape}{1 if bold else 0}{clr}m"
    reset = "\033[0m"
    return f"{fmt}{string}{reset}"



def generate_blobgen_cmd(
    root,
    device_yaml,
    blob_yaml,
    blob_output_dir,
    temporal_epoch,
    chip_ids,
):
    # TODO: This blobgen is deprecated. Use src/blobgen2 c++ code.
    # Even further, this whole file shouldn't exist. There are exactly the same python
    # tools located in third_party/budabackend/verif/common
    blobgen_exe = root + "/tb/llk_tb/overlay/blob_gen.rb"
    temporal_epoch_graph_name = "pipegen_epoch" + str(temporal_epoch)
 
    # parse general spec
    import yaml
    with open(device_yaml) as fd:
        device_descriptor_yaml = yaml.load(fd, Loader=yaml.Loader)

        grid_size_x = int(device_descriptor_yaml["grid"]["x_size"])
        grid_size_y = int(device_descriptor_yaml["grid"]["y_size"])
        physical_grid_size_x = grid_size_x
        physical_grid_size_y = grid_size_y
        if "physical" in device_descriptor_yaml:
            if "x_size" in device_descriptor_yaml["physical"]:
                physical_grid_size_x = int(device_descriptor_yaml["physical"]["x_size"])
            if "y_size" in device_descriptor_yaml["physical"]:
                physical_grid_size_y = int(device_descriptor_yaml["physical"]["y_size"])
        arch_name = str(device_descriptor_yaml["arch_name"]).lower()
        overlay_version = int(device_descriptor_yaml["features"]["overlay"]["version"])
        tensix_memsize = 1499136 if "wormhole_b0" == arch_name else 1024 * 1024
        noc_translation_id_enabled = False
        if "noc" in device_descriptor_yaml["features"] and "translation_id_enabled" in device_descriptor_yaml["features"]["noc"]:
            noc_translation_id_enabled = bool(device_descriptor_yaml["features"]["noc"]["translation_id_enabled"])

        blobgen_cmd = [
            "ruby", blobgen_exe,
            "--blob_out_dir", blob_output_dir,
            "--graph_yaml 1",
            "--graph_input_file", blob_yaml,
            "--graph_name", temporal_epoch_graph_name,
            "--noc_x_size", str(physical_grid_size_x),
            "--noc_y_size", str(physical_grid_size_y),
            "--noc_x_logical_size", str(grid_size_x),
            "--noc_y_logical_size", str(grid_size_y),
            "--chip", arch_name,
            "--noc_version", str(overlay_version),
            "--tensix_mem_size", str(tensix_memsize),
            "--noc_translation_id_enabled", str(int(noc_translation_id_enabled)),
        ]

        # parse eth spec
        eth_cores = device_descriptor_yaml["eth"]
        if len(eth_cores) > 0:
            l1_overlay_blob_base = 128 + 140 * 1024
            if "wormhole_b0" == arch_name:
                eth_max_memsize = 256 * 1024
                eth_overlay_blob_base = 0x9000 + 92 * 1024 + 128
                eth_data_buffer_space_base = 0x9000 + 124 * 1024
            else: # grayskull
                eth_max_memsize = 0 
                eth_overlay_blob_base = 0
                eth_data_buffer_space_base = 0
            eth_cores_swap = []
            for e_core in eth_cores:
                cord = e_core.split('-')
                swapped_cord = str(cord[1]) + '-' + str(cord[0])
                eth_cores_swap.append(swapped_cord)
            eth_cores_str = ",".join(eth_cores_swap)
            chip_ids_str = ",".join([str(idx) for idx in chip_ids])
            extra_cmds = [
                "--tensix_mem_size_eth", str(eth_max_memsize),
                "--eth_cores", eth_cores_str,
                "--blob_section_start", str(l1_overlay_blob_base),
                "--blob_section_start_eth", str(eth_overlay_blob_base),
                "--data_buffer_space_base_eth", str(eth_data_buffer_space_base),
                "--chip_ids", chip_ids_str
            ]
            blobgen_cmd.extend(extra_cmds)

        return blobgen_cmd

def pipegen(
    net2pipe_output_dir,
    device_yaml="third_party/budabackend/device/grayskull_120_arch.yaml",
    run_blobgen=False,
    verbose=False,
):
    if "BUDA_HOME" not in os.environ:
        pipegen_root_path = "./third_party/budabackend/"
    else:
        pipegen_root_path = os.environ["BUDA_HOME"] + "/"
    pipegen_path = pipegen_root_path + "build/bin/pipegen2"
    assert os.path.isdir(net2pipe_output_dir)
    root = net2pipe_output_dir
    error_msgs = []
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)) and "temporal_epoch" in d:
            temporal_epoch = int(d.split("_")[-1])
            base_path = os.path.join(root, d, "overlay")
            pipegen_yaml = os.path.join(base_path, "pipegen.yaml")
            blob_yaml = os.path.join(base_path, "blob.yaml")
            perf_level = "0"
            cmd = [
                pipegen_path,
                pipegen_yaml,
                device_yaml,
                blob_yaml,
                str(temporal_epoch),
                perf_level,
            ]

            p = subprocess.run(cmd, capture_output=True)
            pipegen_fails = (p.returncode)
            if verbose:
                sys.stdout.buffer.write(p.stdout)
            if p.returncode != 0:
                error_message = f"repro: {' '.join(cmd)}\n"
                for l in p.stdout.decode("utf-8").split("\n"):
                    if "ERROR" in l:
                        error_message += l
                error_msgs.append(error_message)

            if run_blobgen and not pipegen_fails:
                blobgen_cmd = generate_blobgen_cmd(pipegen_root_path, device_yaml, blob_yaml, base_path, temporal_epoch, [0])
                blobgen_cmd_conn = ' '.join(blobgen_cmd)
                p = subprocess.Popen(blobgen_cmd_conn, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, errors = p.communicate()
                if verbose:
                    sys.stdout.write(stdout.decode("utf-8"))
                if p.returncode != 0:
                    error_message = f"repro: {blobgen_cmd_conn}\n"
                    for l in errors.decode("utf-8").split("\n"):
                        if "Error" in l:
                            error_message += l
                    error_msgs.append(error_message)

    if len(error_msgs) > 0:
        return 1, "\n".join(error_msgs)
    else:
        return 0, ""


def net2pipe_stats(net2pipe_output_dir):
    import yaml
    from collections import defaultdict

    class CoreInfo:
        def __init__(self):
            self.coord = None
            self.bytes = defaultdict(lambda: 0)
            self.op = None

        def total_bytes(self):
            return sum(self.bytes.values())

        def __repr__(self):
            return f"Core({self.coord[0]:2}, {self.coord[1]:2}) - {self.op}"

    def decode_buffer(bid):
        if bid < 8:
            return f"input[{bid}]"
        elif bid < 16:
            return f"param[{bid - 8}]"
        elif bid < 24:
            return f"output[{bid - 16}]"
        elif bid < 32:
            return f"inter[{bid - 24}]"
        else:
            return f"unknown[{bid}]"

    def print_pipegen_yaml(pipegen_yaml):
        print("Stats:", pipegen_yaml)
        with open(pipegen_yaml) as fd:
            specs = list(yaml.load_all(fd, Loader=yaml.Loader))

        chip_info = defaultdict(lambda: defaultdict(CoreInfo))
        dram_info = defaultdict(lambda: 0)
        for spec in specs:
            for k, v in spec.items():
                if "graph_name" in k:
                    print(f"  {k}: {v}")
                if "buffer" in k and v["core_coordinates"][0] != 255:
                    coords = tuple(v["core_coordinates"])
                    core_info = chip_info[v["chip_id"][0]][coords]
                    core_info.coord = coords
                    if not v["dram_buf_flag"]:
                        core_info.op = v["md_op_name"]
                        bid = v["id"]
                    else:
                        bid = 8
                    core_info.bytes[bid] += v["size_tiles"] * v["tile_size"]

        for chip_id in sorted(chip_info.keys()):
            print(f"  Chip[{chip_id}]:")
            for core in sorted(chip_info[chip_id].keys()):
                info = chip_info[chip_id][core]
                print(f"    {info}:")
                for k in sorted(info.bytes.keys()):
                    print(f"      {decode_buffer(k):>9} bytes: {info.bytes[k]}")
                print(f"      {'total':>9} bytes: {info.total_bytes()}")

    assert os.path.isdir(net2pipe_output_dir)
    root = net2pipe_output_dir
    for d in os.listdir(root):
        if os.path.isdir(os.path.join(root, d)) and "temporal_epoch" in d:
            base_path = os.path.join(root, d, "overlay")
            pipegen_yaml = os.path.join(base_path, "pipegen.yaml")
            print_pipegen_yaml(pipegen_yaml)


def net2pipe(
    netlist,
    device_yaml="third_party/budabackend/device/grayskull_120_arch.yaml",
    cluster_cfg_yaml=None,
    stats=False,
    run_pipegen=False,
    run_blobgen=False,
    verbose=False,
):
    net2pipe_output_dir = "net2pipe_output"
    stdout = f"{net2pipe_output_dir}/net2pipe.stdout"
    stderr = f"{net2pipe_output_dir}/net2pipe.stderr"
    subprocess.run(["rm", "-rf", net2pipe_output_dir])
    subprocess.run(["mkdir", "-p", net2pipe_output_dir])
    if "BUDA_HOME" not in os.environ:
        net2pipe_root_path = "./third_party/budabackend/"
    else:
        net2pipe_root_path = os.environ["BUDA_HOME"] + "/"
    net2pipe_path = net2pipe_root_path + "build/bin/net2pipe"
    cmd = [
        net2pipe_path,
        netlist,
        net2pipe_output_dir,
        "0",
        device_yaml,
    ]

    if cluster_cfg_yaml:
        cmd.append(cluster_cfg_yaml)

    p = subprocess.run(cmd, capture_output=True)
    error_message = ""
    try:
        with open(stdout, "wb") as fd:
            fd.write(p.stdout)
        with open(stderr, "wb") as fd:
            fd.write(p.stderr)
    except:
        pass
    if verbose:
        sys.stdout.buffer.write(p.stdout)
        sys.stderr.buffer.write(p.stderr)
    if p.returncode != 0:
        pytest = ""
        repro = " ".join(cmd)
        with open(netlist) as fd:
            for l in fd.readlines():
                if "pytest" in l:
                    pytest = l.strip()
                    break
        error_message += f"{pytest}\n# {repro}\n"
        found = False
        for l in p.stdout.decode("utf-8").split("\n"):
            found |= "ERROR" in l
            if found and l != "":
                error_message += l
        for l in p.stderr.decode("utf-8").split("\n"):
            if l != "":
                error_message += l

    if p.returncode == 0 and stats:
        net2pipe_stats(net2pipe_output_dir)

    if p.returncode == 0 and run_pipegen:
        return pipegen(net2pipe_output_dir, device_yaml=device_yaml, run_blobgen=run_blobgen, verbose=verbose)

    return p.returncode, error_message


def print_status(netlist, returncode, error_message):
    if returncode == 0:
        print(
            style(f"{netlist}:", bold=True),
            style("OK", color="green", bold=True),
        )
    else:
        print(
            style(f"{netlist}:", bold=True),
            style("FAIL", color="red", bold=True),
            f"({returncode}):",
        )
        print(error_message)


def main():
    parser = argparse.ArgumentParser(description="Batch run net2pipe")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--device-yaml",
        type=str,
        default="third_party/budabackend/device/grayskull_120_arch.yaml",
        help="Device yaml to use",
    )
    parser.add_argument(
        "--cluster-cfg-yaml",
        type=str,
        default=None,
        help="Cluster config yaml to use",
    )
    parser.add_argument(
        "-s",
        "--skip",
        type=str,
        nargs="+",
        default=[],
        help="Regex to skip certain netlist files",
    )
    parser.add_argument(
        "--fail-on-first",
        action="store_true",
        help="Early out on first failure",
    )
    parser.add_argument(
        "--run-pipegen",
        action="store_true",
        help="Run pipegen after net2pipe success",
    )
    parser.add_argument(
        "--run-blobgen",
        action="store_true",
        help="Run blob after pipegen success",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print stats after net2pipe success",
    )
    parser.add_argument("netlist_path", help="Directory of ci outputs or netlist yaml")
    state = parser.parse_args()

    def run_net2pipe(netlist):
        returncode, error_message = net2pipe(
            netlist,
            device_yaml=state.device_yaml,
            cluster_cfg_yaml=state.cluster_cfg_yaml,
            run_pipegen=(state.run_pipegen or state.run_blobgen),
            run_blobgen=state.run_blobgen,
            stats=state.stats,
            verbose=state.verbose,
        )
        print_status(netlist, returncode, error_message)
        return returncode

    fail = False
    if os.path.isdir(state.netlist_path):
        for root, dirs, files in os.walk(state.netlist_path):
            for netlist in files:
                if not netlist.endswith(".yaml"):
                    continue
                netlist = os.path.join(root, netlist)
                if any(re.search(skip, netlist) for skip in state.skip):
                    print(
                        style(f"{netlist}:", bold=True),
                        style("SKIP", color="blue", bold=True),
                    )
                    continue
                fail |= run_net2pipe(netlist)
                if state.fail_on_first and fail:
                    return int(fail)
    else:
        fail |= run_net2pipe(state.netlist_path)
    return int(fail)


if __name__ == "__main__":
    sys.exit(main())
