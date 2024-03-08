# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Generic test model randomizer


from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
import yaml

from pybuda.op_repo import OperatorDefinition
from test.conftest import TestDevice


@dataclass
class RandomizerNode:
    index: Optional[int] = None
    out_value: Optional[str] = None
    operator: Optional[OperatorDefinition] = None
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    inputs: Optional[List] = None

    def operator_name(self):
        return f"op{self.index}"

    def layer_name(self):
        return f"l{self.index}"

    def node_name(self):
        return self.operator_name() if self.operator.is_operator() else self.layer_name()

    def get_name(self):
        return self.operator.name

    def node_info(self):
        return f"{self.node_name()} {self.get_name()}"


@dataclass
class ExecutionContext:
    values: Dict
    last_value: torch.Tensor
    node: Optional[RandomizerNode] = None
    inputs: Optional[List[torch.Tensor]] = None


@dataclass
class RandomizerParameters:
    test_index: int
    random_seed: int
    test_device: TestDevice
    framework_name: str
    graph_builder_name: str


# TODO load from file
@dataclass
class RandomizerGraph:
    # parameters: RandomizerParameters
    nodes: List[RandomizerNode]
    input_shape: Optional[Tuple[int, ...]] = None  # will be set by the graph_builder later
    # graph_builder: Optional[str] = None

    def to_ops_str(self):
        ops = [node.get_name() for node in self.nodes]
        ops_str = " -> ".join(ops)
        return ops_str

    def short_description(self):
        return f"ops: ({self.to_ops_str()}) input_shape: {self.input_shape}"

    # TODO support serialization/deserialization of RandomizerGraph
    def to_str(self):
        graph_dict = asdict(self)
        # Serialize dictionary to YAML string
        yaml_str = yaml.dump(graph_dict)
        # yaml_str = json.dumps(graph.__dict__)
        return yaml_str


@dataclass
class RandomizerConfig:
    print_graph: bool = True
    print_code: bool = False
    # debug_forward: bool = True  # TODO remove obsoleted
    run_test: bool = True
    test_dir:str = "pybuda/test/random_tests"
    save_tests: bool = False
    # build_model_from_code: bool = False  # TODO remove obsoleted
    min_op_size: int = 16
    max_op_size: int = 512
    num_of_nodes: int = 10


@dataclass
class RandomizerTestContext:
    randomizer_config: RandomizerConfig
    parameters: RandomizerParameters
    # framework: Framework
    # graph_builder: GraphBuilder
    graph: Optional[RandomizerGraph]  # graph will be constructed later during test processing
