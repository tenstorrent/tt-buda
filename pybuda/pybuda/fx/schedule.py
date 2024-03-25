# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#
# Schedule describes the order of execution of FX and Buda graphs, and the mapping of inputs and outputs
#

from typing import List, Dict, Set, Tuple, Optional
from enum import Enum

import torch
from loguru import logger

from pybuda.fx.graph_utils import get_output_node

# Enum to hold the source of a tensor
class TensorSource(Enum):
    INPUT = 1
    INTERMEDIATE = 2
    OUTPUT = 3
    
# Convenience tuple indicating whether tensor is intermediate or input/output
class TensorState:

    def __init__(self, src: TensorSource, node: torch.fx.Node):
        self.src = src
        self.node = node

    def __repr__(self):
        return f"TensorState({self.src}, {self.node})"

# Define where the input is coming from for each schedule item - from main inputs, or from other graphs
# If it's from main inputs, then index is the index of the inputs, otherwise it's index into list of intermediates
class InputSource:

    def __init__(self, src: TensorSource, index: int):
        self.src = src
        self.index = index 

    def __repr__(self):
        return f"InputSource({self.src}, {self.index})"

# Define where the output goes, to main outputs or to intermediates
class OutputDest:
    def __init__(self, intermediate: bool, index: int):
        self.intermediate = intermediate
        self.index = index

    def __repr__(self):
        return f"{'Intermediate' if self.intermediate else 'Output'} {self.index}"

# Single graph call, with mappings of inputs and outputs
class ScheduleItem:
    def __init__(self, fallback: bool, inputs: List[InputSource], outputs: List[OutputDest], graph: Optional[torch.fx.Graph], graph_index: int):
        self.fallback = fallback
        self.graph = graph
        self.graph_index = graph_index
        self.inputs = inputs
        self.outputs = outputs

    def is_fallback_graph(self) -> bool:
        return self.fallback

    def get_subgraph_input_index(self, index: int) -> int:
        # filter graph inputs
        graph_inputs = [i.index for i in self.inputs if i.src == TensorSource.INPUT]
        assert index < len(graph_inputs), f"Index {index} out of range for graph inputs {graph_inputs}"
        return graph_inputs[index]

    def __repr__(self) -> str:
        if self.is_fallback_graph():
            return f"ScheduleItem(fallback graph={self.graph_index}, inputs={self.inputs}, outputs={self.outputs})"
        else:
            return f"ScheduleItem(device graph={self.graph_index}, inputs={self.inputs}, outputs={self.outputs})"

class Schedule:

    def __init__(self, 
            inputs: List[torch.fx.Node], 
            outputs: List[torch.fx.Node],
            device_graphs: List[torch.fx.Graph],
            fallback_graphs: List[torch.fx.Graph],
            mappings: Dict[str, Dict[torch.fx.Node, torch.fx.Node]]):

        new_io_mapping = mappings["new_io_mapping"]
        placeholder_map = mappings["placeholder_map"]
        #copied_node_mapping = mappings["copied_node_mapping"]
        #moved_output_mapping = mappings["moved_output_mapping"]

        intermediate_valid: Set[torch.fx.Node] = set() # Set of valid intermediate nodes, after a graph has been executed
        outputs_valid: Dict[torch.fx.Node, int] = {} # Map of valid outputs, and their index

        # For each graph, figure out where the inputs are coming from, and which of the outputs it creates
        input_mappings: Dict[int, List[TensorState]] = {} # list per subgraph
        for i, graph in enumerate(fallback_graphs):
            # Find inputs
            input_mappings[i] = []
            for node in graph.nodes:
                if node.op != "placeholder":
                    continue;
            
                # Find the original node
                if node in placeholder_map or node in inputs:
                    # Original input
                    input_mappings[i].append(TensorState(TensorSource.INPUT, node))
                    continue

                # Intermediate or output from another graph
                if node in new_io_mapping:
                    src = new_io_mapping[node]
                    if src in outputs:
                        # Output from another graph
                        input_mappings[i].append(TensorState(TensorSource.OUTPUT, src))
                    else:
                        input_mappings[i].append(TensorState(TensorSource.INTERMEDIATE, src))
                    continue

                # No other option is legal
                assert False, f"Placeholder {node} not found in any mapping"

        # The device graphs
        device_mappings: Dict[int, List[TensorState]] = {} # list per subgraph
        for i, device_graph in enumerate(device_graphs):
            device_mappings[i] = []
            for node in device_graph.nodes:
                if node.op != "placeholder":
                    continue;

                # Original input
                if node in inputs:
                    device_mappings[i].append(TensorState(TensorSource.INPUT, node))
                    continue

                # Intermediate or output from another graph
                if node in new_io_mapping:
                    src = new_io_mapping[node]
                    if src in outputs:
                        device_mappings[i].append(TensorState(TensorSource.OUTPUT, src))
                    else:
                        device_mappings[i].append(TensorState(TensorSource.INTERMEDIATE, src))
                    continue

                # No other option is legal
                assert False, f"Placeholder {node} not found in any mapping"

        # Keep figuring out which graphs we can run, i.e. we have all inputs available, until we're done with all of them
        self.schedule : List[ScheduleItem] = []
        to_schedule : List[Tuple[bool, int]] = [(True, i) for i in range(len(fallback_graphs))]

        for i in range(len(device_graphs)):
            if len(device_graphs[i].nodes) > 0:
                to_schedule.append((False, i))

        # Map intermediate to unique IDs that we can put in the schedule
        self.next_intermediate_id = 0
        intermediate_ids : Dict[torch.fx.Node, int] = {}

        # Figure out where outputs go, and set intermediate IDs and valids
        def record_outputs(graph: torch.fx.Graph) -> List[OutputDest]:
            output_list = []
            output_node = get_output_node(graph)

            for arg in output_node.args[0]:

                # Figure out where this output goes - to an intermediate, or to the main outputs
                if arg in outputs:
                    output_list.append(OutputDest(False, outputs.index(arg)))
                    outputs_valid[arg] = outputs.index(arg)
                    continue

                # Intermediate output, assign new ID and record
                intermediate_ids[arg] = self.next_intermediate_id
                output_list.append(OutputDest(True, self.next_intermediate_id))
                self.next_intermediate_id += 1
                        
                # Record that the intermediate is valid, for scheduling purposes
                intermediate_valid.add(arg)

            return output_list

        def generate_inputs(graph: torch.fx.Graph)  -> List[InputSource]:
            # Generate list of input sources for this graph
            input_list = []
            for node in graph.nodes:
                if node.op != "placeholder":
                    continue

                if node in inputs:
                    input_list.append(InputSource(TensorSource.INPUT, inputs.index(node)))
                    continue

                if node in placeholder_map:
                    input_list.append(InputSource(TensorSource.INPUT, inputs.index(placeholder_map[node])))
                    continue

                assert node in new_io_mapping
                src = new_io_mapping[node]
                if src in outputs:
                    input_list.append(InputSource(TensorSource.OUTPUT, outputs.index(src)))
                else:
                    input_list.append(InputSource(TensorSource.INTERMEDIATE, intermediate_ids[src]))

            return input_list

        while len(to_schedule) > 0:
            progress = False
            for fallback, index in to_schedule:
                if fallback:
                    if all([t.node in intermediate_valid for t in input_mappings[index] if t.src == TensorSource.INTERMEDIATE]) and \
                            all([t.node in outputs_valid for t in input_mappings[index] if t.src == TensorSource.OUTPUT]):
                        # We can run this graph
                        logger.trace(f"Scheduling fallback graph {index}")
                        self.schedule.append(
                                ScheduleItem(fallback, generate_inputs(fallback_graphs[index]), 
                                        record_outputs(fallback_graphs[index]), fallback_graphs[index], index))
                        to_schedule.remove((True, index))
                        progress = True
                else:
                    if all([t.node in intermediate_valid for t in device_mappings[index] if t.src == TensorSource.INTERMEDIATE]) and \
                            all([t.node in outputs_valid for t in device_mappings[index] if t.src == TensorSource.OUTPUT]):
                        # We can run device graph
                        logger.trace(f"Scheduling device graph")
                        self.schedule.append(
                                ScheduleItem(fallback, generate_inputs(device_graphs[index]),
                                    record_outputs(device_graphs[index]), device_graphs[index], index))
                        to_schedule.remove((False, index))
                        progress = True

            if not progress:
                print("Intermediate valids", intermediate_valid)
                print("Outputs valids", outputs_valid)
                print("To schedule", to_schedule)
                for i, im in enumerate(input_mappings):
                    print(f"Input mappings {i}", im)
                print("inputs (aten): ", device_mappings)
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

    def get_device_schedule_item(self, index: int) -> ScheduleItem:
        return next(filter(lambda x: x.graph_index == index and not x.fallback, self.schedule))

    def get_device_program_ids(self) -> List[int]:
        return [i.graph_index for i in self.schedule if not i.fallback]

    def validate(self, num_inputs: int, num_outputs: int):
        # Check that all inputs are used, and all outputs are generated
        unused_inputs = set(range(num_inputs))
        unused_outputs = set(range(num_outputs))
        for item in self.schedule:
            for input_source in item.inputs:
                if input_source.src != TensorSource.INPUT:
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

