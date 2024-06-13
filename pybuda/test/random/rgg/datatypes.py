# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Generic test model randomizer


from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torch

from pybuda.op_repo import OperatorDefinition
from test.conftest import TestDevice


# Defining a type for tensor shape
TensorShape = Tuple[int, ...]

@dataclass
class RandomizerInputNode:
    out_value: str
    input_shape: TensorShape


@dataclass
class RandomizerNode:
    index: Optional[int] = None
    out_value: Optional[str] = None
    operator: Optional[OperatorDefinition] = None
    inputs: List['RandomizerNode'] = field(default_factory=list)
    constructor_kwargs: Dict[str, object] = field(default_factory=dict)
    forward_kwargs: Dict[str, object] = field(default_factory=dict)
    input_shapes: List[TensorShape] = field(default_factory=list)
    output_shape: TensorShape = None

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
    nodes: List[RandomizerNode] = field(default_factory=list)
    input_nodes: List[RandomizerInputNode] = field(default_factory=list)
    # graph_builder: Optional[str] = None


@dataclass
class RandomizerConfig:
    print_graph: bool = True
    print_code: bool = False
    # debug_forward: bool = True  # TODO remove obsoleted
    run_test: bool = True
    test_dir:str = "pybuda/test/random_tests"
    save_tests: bool = False
    # build_model_from_code: bool = False  # TODO remove obsoleted
    verify_shapes: bool = False,
    dim_min: int = 3
    dim_max: int = 4
    op_size_per_dim_min: int = 16
    op_size_per_dim_max: int = 512
    microbatch_size_min: int = 1
    microbatch_size_max: int = 8
    num_of_nodes: int = 10


@dataclass
class RandomizerTestContext:
    randomizer_config: RandomizerConfig
    parameters: RandomizerParameters
    # framework: Framework
    # graph_builder: GraphBuilder
    graph: Optional[RandomizerGraph]  # graph will be constructed later during test processing
