# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Generic test model randomizer


from typing import Dict, List, Optional, Final
from dataclasses import dataclass, field
import random
import torch

from pybuda.op_repo import TensorShape
from pybuda.op_repo import OperatorDefinition
from pybuda.op_repo import ShapeCalculationContext
from test.conftest import TestDevice


@dataclass
class RandomizerInputNode:
    constant: Final[bool] = field(default=False, init=False)
    out_value: str
    input_shape: TensorShape


@dataclass
class RandomizerConstantNode:
    constant: Final[bool] = field(default=True, init=False)
    out_value: str
    input_shape: TensorShape


@dataclass
class RandomizerNode:
    constant: Final[bool] = field(default=False, init=False)
    index: Optional[int] = None
    out_value: Optional[str] = None
    operator: Optional[OperatorDefinition] = None
    input_num: int = field(init=False)
    inputs: List['RandomizerNode'] = field(init=False)
    constructor_kwargs: Dict[str, object] = field(default_factory=dict)
    forward_kwargs: Dict[str, object] = field(default_factory=dict)
    input_shapes: List[TensorShape] = field(default_factory=list)
    output_shape: TensorShape = None

    def __post_init__(self):
        # List of input nodes is initialized with None values for each input
        # Inputs will be set later during graph construction
        self.input_num = self.operator.input_num_range.operands_min
        self.init_inputs()

    def init_inputs(self):
        self.inputs = [None for _ in range(self.input_num)]

    @property
    def operator_name(self):
        return f"op{self.index}"

    @property
    def layer_name(self):
        return f"l{self.index}"

    @property
    def node_name(self):
        return self.operator_name if self.operator.is_operator else self.layer_name

    @property
    def name(self):
        return self.operator.name

    @property
    def node_info(self):
        return f"{self.node_name} {self.name}"


class NodeShapeCalculationContext(ShapeCalculationContext):

    def __init__(self, node: RandomizerNode, test_context: 'RandomizerTestContext'):
        self.node = node
        self.test_context = test_context

    @property
    def operator(self) -> OperatorDefinition:
        return self.node.operator
    
    @property
    def input_num(self) -> int:
        return self.node.input_num

    @property
    def constructor_kwargs(self) -> Dict[str, object]:
        return self.node.constructor_kwargs

    @property
    def forward_kwargs(self) -> Dict[str, object]:
        return self.node.forward_kwargs

    @property
    def output_shape(self) -> TensorShape:
        return self.node.output_shape

    @property
    def rng_shape(self) -> random.Random:
        return self.test_context.rng_shape


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
    constant_nodes: List[RandomizerConstantNode] = field(default_factory=list)
    # graph_builder: Optional[str] = None


@dataclass
class RandomizerConfig:
    print_graph: bool = True
    print_code: bool = False
    run_test: bool = True
    test_dir:str = "pybuda/test/random_tests"
    save_tests: bool = False
    save_failing_tests: bool = False
    # build_model_from_code: bool = False  # TODO remove obsoleted
    debug_shapes: bool = False,
    verify_shapes: bool = False,
    verification_timeout: int = 60
    dim_min: int = 3
    dim_max: int = 4
    op_size_per_dim_min: int = 16
    op_size_per_dim_max: int = 512
    microbatch_size_min: int = 1
    microbatch_size_max: int = 8
    num_of_nodes_min: int = 5
    num_of_nodes_max: int = 10
    num_fork_joins_max: int = 50
    constant_input_rate: int = 20
    same_inputs_percent_limit: int = 10


@dataclass
class RandomizerTestContext:
    randomizer_config: RandomizerConfig
    parameters: RandomizerParameters
    # framework: Framework
    # graph_builder: GraphBuilder
    graph: Optional[RandomizerGraph]  # graph will be constructed later during test processing
    test_name: str = "Default"

    # random number generators for graph building
    rng_graph: Optional[random.Random] = None
    # random number generators for shape generation
    rng_shape: Optional[random.Random] = None
    # random number generators for parameters
    rng_params: Optional[random.Random] = None


class InvalidShape(Exception):

    def __init__(self, message):
        super().__init__(message)
