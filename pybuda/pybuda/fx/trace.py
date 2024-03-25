# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict, Set, Optional
from collections import defaultdict

from loguru import logger
import torch

class IOTracer:
    # For a list of graphs, find which inputs affect which outputs. Cache results to make tracing faster.
    def __init__(self, graphs: List[torch.fx.Graph]):
        self.graphs = graphs
        self.input_to_output_map : Dict[torch.fx.Node, List[torch.fx.Node]] = {}

    def remove_graph(self, graph: torch.fx.Graph):
        assert graph in self.graphs
        self.graphs.remove(graph)
        to_remove = []
        for input in self.input_to_output_map:
            if input.graph == graph:
                to_remove.append(input)

    def add_graph(self, graph: torch.fx.Graph):
        assert graph not in self.graphs
        self.graphs.append(graph)

    def get_output_nodes(self, input: torch.fx.Node) -> List[torch.fx.Node]:
        if input not in self.input_to_output_map:
            self._trace_graph(input.graph)

        print(" - get_output_nodes", input.name, self.input_to_output_map[input])
        return self.input_to_output_map[input]

    def _trace_graph(self, graph: torch.fx.Node):
        # Trace all input to outputs

        # Keep track of visited noted, and which outputs they lead to, to avoid tracing the whole graph again for other inputs
        node_to_output : Dict[torch.fx.Node, Set[torch.fx.Node]] = defaultdict(set)

        def trace(node: torch.fx.Node) -> Set[torch.fx.Node]:
            for user in node.users:
                if user.op == "output":
                    node_to_output[node].add(node)
                    print(f" {node.name} -> direct output")
                elif user in node_to_output: # depth-first, so we should have already reached the outputs if we hit the node again - no cycles
                    node_to_output[node].update(node_to_output[user])
                    print(f" {node.name} -> extended output to {node_to_output[user]}")
                else:
                    node_to_output[node].update(trace(user))
            return node_to_output[node]
        
        for node in graph.nodes:
            if node.op == "placeholder":
                self.input_to_output_map[node] = list(trace(node))

    def trace_for_cycle(self, input_node: torch.fx.Node, outputs_to_dest_node: Dict[torch.fx.Node, Set[torch.fx.Node]]) -> Optional[torch.fx.Node]:
        # Given an input node, and mapping of outputs to inputs in other graphs, trace through to see if a cycle exists, i.e.
        # if we reach the original graph again. Return the output node through which we reached the cycle, or None if there are none

        logger.trace(f"Trace for cycle from {input_node}")
        def trace(output_node: torch.fx.Node, target_graph: torch.fx.Graph) -> torch.fx.Node:
            # Trace output node to other graphs, and see if target graph is reached
            for input_node in outputs_to_dest_node[output_node]:
                if input_node.graph == target_graph:
                    return input_node

                # Trace further
                for output_node in self.get_output_nodes(input_node):
                    node = trace(output_node, target_graph)
                    if node:
                        return node

            return None

        for output_node in self.get_output_nodes(input_node):
            node = trace(output_node, input_node.graph)
            if node:
                logger.trace(f" -- tracing found cycle through output {output_node} to {node}")
                return node

        return None
