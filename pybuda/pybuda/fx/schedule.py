# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#
# Schedule describes the order of execution of FX and Buda graphs, and the mapping of inputs and outputs
#

from typing import List, Dict, Set, Tuple, Optional

import torch
from loguru import logger

from pybuda.fx.graph_utils import get_output_node
    
# Convenience tuple indicating whether tensor is intermediate or input/output
class TensorState:
    def __init__(self, intermediate: bool, node: torch.fx.Node):
        self.intermediate = intermediate
        self.node = node

    def __repr__(self):
        return f"{'Intermediate' if self.intermediate else 'Input/Output'} {self.node.name}"

# Define where the input is coming from for each schedule item - from main inputs, or from other graphs
# If it's from main inputs, then index is the index of the inputs, otherwise it's index into list of intermediates
class InputSource:
    def __init__(self, intermediate: bool, index: int):
        self.intermediate = intermediate
        self.index = index 

    def __repr__(self):
        return f"{'Intermediate' if self.intermediate else 'Input'} {self.index}"

# Define where the output goes, to main outputs or to intermediates
class OutputDest:
    def __init__(self, intermediate: bool, index: int):
        self.intermediate = intermediate
        self.index = index

    def __repr__(self):
        return f"{'Intermediate' if self.intermediate else 'Output'} {self.index}"

# Single graph call, with mappings of inputs and outputs
class ScheduleItem:
    def __init__(self, inputs: List[InputSource], outputs: List[OutputDest], graph: Optional[torch.fx.Graph], graph_index: int):
        self.graph = graph
        self.graph_index = graph_index
        self.inputs = inputs
        self.outputs = outputs

    def is_fallback_graph(self) -> bool:
        return self.graph is not None

    def __repr__(self) -> str:
        if self.is_fallback_graph():
            return f"ScheduleItem(fallback graph={self.graph_index}, inputs={self.inputs}, outputs={self.outputs})"
        else:
            return f"ScheduleItem(device graph, inputs={self.inputs}, outputs={self.outputs})"

class Schedule:

    def __init__(self, 
            subgraph_index: int, 
            inputs: List[torch.fx.Node], 
            outputs: List[torch.fx.Node],
            aten_graph: torch.fx.Graph,
            fallback_graphs: List[torch.fx.Graph],
            mappings: Dict[str, Dict[torch.fx.Node, torch.fx.Node]]):

        new_io_mapping = mappings["new_io_mapping"]
        placeholder_map = mappings["placeholder_map"]
        #copied_node_mapping = mappings["copied_node_mapping"]
        moved_output_mapping = mappings["moved_output_mapping"]

        intermediate_valid: Set[torch.fx.Node] = set() # Set of valid intermediate ids, after a graph has been executed

        # For each graph, figure out where the inputs are coming from, and which of the outputs it creates
        input_mappings = {}
        for i, graph in enumerate(fallback_graphs):
            # Find inputs
            input_mappings[i] = []
            for node in graph.nodes:
                if node.op != "placeholder":
                    continue;

                # Find the original node
                if node in placeholder_map:
                    # Original input
                    input_mappings[i].append(TensorState(False, node))
                    continue

                if node in new_io_mapping:
                    # Intermediate from another graph
                    input_mappings[i].append(TensorState(True, new_io_mapping[node]))
                    continue

                # No other option is legal
                assert False, f"Placeholder {node} not found in any mapping"

        # The original graph, the one that will run on the device
        aten_mappings = []
        for node in aten_graph.nodes:
            if node.op != "placeholder":
                continue;

            # Original input
            if node in inputs:
                aten_mappings.append(TensorState(False, node))
                continue

            # Intermediate from another graph
            if node in new_io_mapping:
                # Intermediate from another graph
                aten_mappings.append(TensorState(True, new_io_mapping[node]))
                continue

            # No other option is legal
            assert False, f"Placeholder {node} not found in any mapping"

        # Keep figuring out which graphs we can run, i.e. we have all inputs available, until we're done with all of them
        self.schedule : List[ScheduleItem] = []
        to_schedule : List[Tuple[bool, int]] = [(True, i) for i in range(len(fallback_graphs))]
        to_schedule.append((False, 0)) # The original graph

        # Map intermediate to unique IDs that we can put in the schedule
        self.next_intermediate_id = 0
        intermediate_ids : Dict[torch.fx.Node, int] = {}

        # Figure out where outputs go, and set intermediate IDs and valids
        def record_outputs(graph: torch.fx.Graph, fallback_graph: bool) -> List[OutputDest]:
            output_index = 0
            output_list = []
            output_node = get_output_node(graph)

            for arg in output_node.args[0]:

                # Figure out where this output goes - to an intermediate, or to the main outputs
                if arg in outputs:
                    output_list.append(OutputDest(False, outputs.index(arg)))
                    output_index += 1
                    continue

                # Intermediate output, assign new ID and record
                intermediate_ids[arg] = self.next_intermediate_id
                output_list.append(OutputDest(True, self.next_intermediate_id))
                self.next_intermediate_id += 1
                        
                # Record that the intermediate is valid, for scheduling purposes
                intermediate_valid.add(arg)

            return output_list

        def generate_inputs(graph: torch.fx.Graph, fallback_graph: bool) -> List[InputSource]:
            # Generate list of input sources for this graph
            input_list = []
            for node in graph.nodes:
                if node.op != "placeholder":
                    continue

                if not fallback_graph and node in inputs:
                    input_list.append(InputSource(False, inputs.index(node)))
                    continue

                if fallback_graph and node in placeholder_map:
                    input_list.append(InputSource(False, inputs.index(placeholder_map[node])))
                    continue

                # Coming from intermediates
                input_list.append(InputSource(True, intermediate_ids[new_io_mapping[node]]))

            return input_list

        while len(to_schedule) > 0:
            progress = False
            for fallback, index in to_schedule:
                if fallback:
                    if all([t.node in intermediate_valid for t in input_mappings[index] if t.intermediate]):
                        # We can run this graph
                        logger.trace(f"Scheduling fallback graph {index}")
                        self.schedule.append(
                                ScheduleItem(generate_inputs(fallback_graphs[index], True), 
                                        record_outputs(fallback_graphs[index], True), fallback_graphs[index], index))
                        to_schedule.remove((True, index))
                        progress = True
                else:
                    if all([t.node in intermediate_valid for t in aten_mappings if t.intermediate]):
                        # We can run device graph
                        logger.trace(f"Scheduling device graph")
                        self.schedule.append(
                                ScheduleItem(generate_inputs(aten_graph, False),
                                    record_outputs(aten_graph, False), None, 0))
                        to_schedule.remove((False, 0))
                        progress = True

            assert progress, "No progress made in scheduling"

        logger.trace(f"Schedule: {self}")
        self.validate(len(inputs), len(outputs))

    def __iter__(self):
        return iter(self.schedule)

    def __repr__(self):
        ret = "Schedule:\n"
        for item in self.schedule:
            ret += f" - {item}\n"
        return ret

    def validate(self, num_inputs: int, num_outputs: int):
        # Check that all inputs are used, and all outputs are generated
        unused_inputs = set(range(num_inputs))
        unused_outputs = set(range(num_outputs))
        for item in self.schedule:
            for input_source in item.inputs:
                if input_source.intermediate:
                    continue
                if input_source.index in unused_inputs: # it's ok if the input is used multiple times
                    unused_inputs.remove(input_source.index)

            for output_dest in item.outputs:
                if output_dest.intermediate:
                    continue
                assert output_dest.index in unused_outputs, f"Output {output_dest.index} used multiple times, or beyond the number of outputs"
                unused_outputs.remove(output_dest.index)

        assert len(unused_inputs) == 0, f"Inputs {unused_inputs} are not used"
        assert len(unused_outputs) == 0, f"Outputs {unused_outputs} are not generated"

