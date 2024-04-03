#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import sys
import yaml


TILE_DIM = 32


def write_reportify_graph(netlist_name, graph, report_name, verbose=False):
    summary_dir = os.path.join(os.environ.get("HOME"), "testify", "ll-sw", netlist_name)
    reportify_path = os.path.join(summary_dir, report_name)
    os.makedirs(os.path.dirname(reportify_path), exist_ok=True)
    summary_path = os.path.join(summary_dir, "summary.yaml")
    if not os.path.exists(summary_path):
        summary = {
            "content": {"name": f"ll-sw.{netlist_name}", "output-dir": summary_dir},
            "type": "summary",
        }
        with open(summary_path, "w") as fd:
            yaml.dump(summary, fd)
    if verbose:
        print("Writing graph:", reportify_path)
    with open(reportify_path, "w") as fd:
        json.dump(graph, fd, indent=4)


def net2reportify(netlist_name, netlist, extract_graphs=[], verbose=False):
    if bool(int(os.environ.get("PYBUDA_DISABLE_REPORTIFY_DUMP", "0"))):
        return
    if type(netlist) is str:
        with open(netlist) as fd:
            netlist = yaml.load(fd, yaml.SafeLoader)

    def node_shape(node):
        w = 1
        z = node["t"]
        r = node["grid_size"][0] * node["mblock"][0] * node["ublock"][0] * TILE_DIM
        c = node["grid_size"][1] * node["mblock"][1] * node["ublock"][1] * TILE_DIM
        return [w, z, r, c]

    def emit_queue(reportify_graph, queue_name, queue):
        reportify_graph["nodes"][queue_name] = queue
        reportify_graph["nodes"][queue_name]["pybuda"] = 1
        reportify_graph["nodes"][queue_name]["epoch"] = 0
        reportify_graph["nodes"][queue_name]["name"] = queue_name
        reportify_graph["nodes"][queue_name]["cache"] = {"shape": node_shape(queue)}
        if queue["input"] == "HOST":
            reportify_graph["nodes"][queue_name]["class"] = "Input::"
            reportify_graph["nodes"][queue_name]["input_nodes"] = []
        else:
            reportify_graph["nodes"][queue_name]["class"] = "Output"
            reportify_graph["nodes"][queue_name]["input_nodes"] = [queue["input"]]
        reportify_graph["nodes"][queue_name]["output_nodes"] = []

    def emit_queues(reportify_graph, node_name, node):
        for queue_name in node["inputs"]:
            if queue_name in netlist["queues"]:
                emit_queue(reportify_graph, queue_name, netlist["queues"][queue_name])
        for queue_name, queue in netlist["queues"].items():
            if node_name == queue["input"]:
                emit_queue(reportify_graph, queue_name, queue)

    def emit_op(reportify_graph, node_name, node, epoch, epoch_type):
        def get_ublock_order(input_idx):
            if node["type"] == "matmul" and input_idx == 0:
                return "c"
            elif node["type"] == "matmul" and input_idx == 1:
                return "r"
            return node["ublock_order"]

        reportify_graph["nodes"][node_name] = node
        reportify_graph["nodes"][node_name]["class"] = node["type"]
        reportify_graph["nodes"][node_name]["pybuda"] = 1
        reportify_graph["nodes"][node_name]["epoch"] = epoch
        reportify_graph["nodes"][node_name]["epoch_type"] = epoch_type
        reportify_graph["nodes"][node_name]["name"] = node_name
        reportify_graph["nodes"][node_name]["cache"] = {"shape": node_shape(node)}
        reportify_graph["nodes"][node_name]["input_nodes"] = node["inputs"]
        reportify_graph["nodes"][node_name]["input_node_to_edge_type"] = {
            k: "Data" for k in node["inputs"]
        }
        reportify_graph["nodes"][node_name]["incoming_edge_port_info"] = [
            f"Data: {n} (port_{i}) ublock_order({get_ublock_order(i)})"
            for i, n in enumerate(node["inputs"])
        ]
        reportify_graph["nodes"][node_name]["output_nodes"] = []
        emit_queues(reportify_graph, node_name, node)

    def get_epoch_type(graph_name):
        s = graph_name.split("_")
        epoch_type, epoch = (s[0], s[-1])
        return (
            int(epoch),
            {"fwd": "Forward", "bwd": "Backward", "opt": "Optimizer"}[epoch_type],
        )

    reportify_graph = {"graph": {}, "nodes": {}}

    for graph_name, graph in netlist["graphs"].items():
        if extract_graphs and graph_name not in extract_graphs:
            continue

        epoch, epoch_type = get_epoch_type(graph_name)
        for node_name, node in graph.items():
            if node_name in {"target_device", "input_count"}:
                continue
            emit_op(reportify_graph, node_name, node, epoch, epoch_type)

    write_reportify_graph(
        netlist_name,
        reportify_graph,
        f"buda_reports/Passes/{netlist_name}.buda",
        verbose=verbose,
    )


def net2placement(
    netlist_name,
    netlist,
    device_yaml=None,
    verbose=False,
):
    if bool(int(os.environ.get("PYBUDA_DISABLE_REPORTIFY_DUMP", "0"))):
        return
    if type(netlist) is str:
        with open(netlist) as fd:
            netlist = yaml.load(fd, yaml.SafeLoader)

    if device_yaml is None:
        if netlist["devices"]["arch"] == "grayskull":
            device_yaml = "third_party/budabackend/device/grayskull_120_arch.yaml"
        elif netlist["devices"]["arch"] == "wormhole_b0":
            device_yaml = "third_party/budabackend/device/wormhole_b0_80_arch.yaml"
        else:
            raise RuntimeError(f"Unknown device type {netlist['devices']['arch']}")

    placement_json = {}
    placement_json["netlist"] = netlist
    with open(device_yaml) as fd:
        device_info = yaml.load(fd, yaml.SafeLoader)
    placement_json["device_info"] = device_info
    write_reportify_graph(
        netlist_name,
        placement_json,
        "placement_reports/placement.json",
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate reportify graph from netlist"
    )
    parser.add_argument("netlist_path", help="path to netlist yaml")
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=[],
        help="A list of graphs to extract from the netlist",
    )
    parser.add_argument(
        "--device-yaml",
        type=str,
        default=None,
        help="Device yaml to use",
    )
    parser.add_argument(
        "--cluster-cfg-yaml",
        type=str,
        default=None,
        help="Cluster config yaml to use",
    )
    state = parser.parse_args()

    assert type(state.netlist_path) is str
    with open(state.netlist_path) as fd:
        netlist = yaml.load(fd, yaml.SafeLoader)
    netlist_name = os.path.splitext(os.path.basename(state.netlist_path))[0]

    net2reportify(netlist_name, netlist, state.graphs, verbose=True)
    net2placement(netlist_name, netlist, device_yaml=state.device_yaml, verbose=True)


if __name__ == "__main__":
    sys.exit(main())
