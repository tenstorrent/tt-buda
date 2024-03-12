# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#
# Various utility functions for working with FX graphs
#
from typing import List, Tuple, Union

import torch
from loguru import logger


def flatten_args(args: List) -> List:
    # Flatten inputs into a list
    node_args = []
    for arg in args:
        if isinstance(arg, torch.fx.Node):
            node_args.append(arg)
        elif isinstance(arg, (list, tuple)):
            node_args.extend(arg)
        # Explicitly allow types to make sure we don't miss anything important
        elif isinstance(arg, (int, float)):
            continue
        else:
            assert False, f"Unsupported argument type {type(arg)}"
    return node_args

            
def reduce_graph(module_or_graph: Union[torch.fx.Graph, torch.fx.GraphModule]):
    # Reduce the graph to only the nodes that are used
    
    # Traverse up the graph from output nodes to populate consumed nodes set
    graph = module_or_graph.graph if isinstance(module_or_graph, torch.fx.GraphModule) else module_or_graph
    consumed = set()
    working_nodes = []
    for node in graph.nodes:
        if node.op == "output":
            working_nodes.append(node)
            consumed.add(node)

    while len(working_nodes) > 0:
        node = working_nodes.pop(0)
        if not isinstance(node, torch.fx.Node):
            continue
        args = flatten_args(node.args)
        for arg in args:
            if arg not in consumed:
                consumed.add(arg)
                working_nodes.append(arg)

    for node in reversed(graph.nodes):
        if node not in consumed:
            logger.debug(f"Removing node {node.name}")
            graph.erase_node(node)

    if len(graph.nodes) == 1:
        for node in graph.nodes:
            if node.op == "output":
                # Remove the output node if it's the only one
                graph.erase_node(node)

def get_output_node(graph: torch.fx.Graph) -> torch.fx.Node:
    # Find the output node of the graph - any faster way to do this?
    for node in reversed(graph.nodes):
        if node.op == "output":
            return node
    return None

def append_to_output(graph: torch.fx.Graph, src: torch.fx.Node) -> Tuple[torch.fx.Node, int]:
    # Append a src node to the output of the graph, and return the index of the output
    output_node = get_output_node(graph)
    if output_node is None:
        # Create a new one
        output_node = graph.output((src,))
        output_node.meta["tensor_meta"] = (src.meta["tensor_meta"],)
        return (output_node, 0)

    output_node.args = ((*output_node.args[0], src),)
    output_node.meta["tensor_meta"] = (*output_node.meta["tensor_meta"] , src.meta["tensor_meta"])
    return (output_node, len(output_node.args) - 1)

def move_output_to_end(graph: torch.fx.Graph):
    if len(graph.nodes) == 0:
        return

    # Output should be at the end, topologically, if it's not there already
    output_node = None
    first = True
    for node in reversed(graph.nodes):
        if node.op == "output":
            if first:
                return # already last
            output_node = node
            break
        first = False

    assert output_node is not None
    graph.node_copy(output_node, lambda x : x)
    graph.erase_node(output_node)

def remove_output_index(node: torch.fx.Node, idx: int):
    # Remove the output index for the list of outputs
    print(node, node.op, node.meta, node.args)
    args = list(node.args[0])
    del args[idx]
    node.args = (tuple(args),)
    
    meta = list(node.meta["tensor_meta"])
    del meta[idx]
    node.meta["tensor_meta"] = tuple(meta)

def graph_lint(graph: torch.fx.Graph, graph_name: str = "graph"):
    # Check that the graph is well formed. Built-in lint doesn't do enough
    # Run built-in first
    graph.lint()

    def check(cond, msg):
        if not cond:
            logger.error(f"Lint error in {graph_name}: {msg}")
            graph.print_tabular()
            raise RuntimeError(f"Lint error in {graph_name}: {msg}")

    # Check that there's only one output node, and that tensor_meta list is matching output list
    found_output = False
    for node in graph.nodes:
        if node.op == "output":
            check(not found_output, f"Multiple output nodes found")
            check(len(node.args) == 1, f"Output node {node} has more than one argument")
            check(len(node.meta["tensor_meta"]) == len(node.args[0]), f"Output node {node} in has mismatched tensor meta and args: {node.meta['tensor_meta']} vs {node.args[0]}")
    

