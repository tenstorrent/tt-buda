# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Utility functions


import random
from typing import Callable, List, Dict
from dataclasses import asdict
from loguru import logger
import re
import yaml

import torch
import pybuda

from pybuda.op_repo import OperatorParam, OperatorDefinition, OperatorParamNumber

from .datatypes import TensorShape
from .datatypes import RandomizerConfig, RandomizerTestContext, RandomizerNode, RandomizerGraph


class StrUtils:

    @staticmethod
    def kwargs_str(**kwargs):
        s = ', '.join([f"{key}= {value}" for key, value in kwargs.items()])
        return s

    @staticmethod
    def args_str(*args):
        s = ', '.join([f"{value}" for value in args])
        if s:
            s = ", " + s
        return s

    @staticmethod
    def camel_case_to_snake_case(camel_case: str) -> str:
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        snake_case = re.sub(pattern, '_', camel_case).lower()
        return snake_case

    @classmethod
    def test_id(cls, test_context: RandomizerTestContext) -> str:
        parameters = test_context.parameters
        graph_builder_snake_case = cls.camel_case_to_snake_case(parameters.graph_builder_name)
        test_id = f"{parameters.framework_name}_{graph_builder_snake_case}_{parameters.test_index}_{parameters.random_seed}"
        return test_id

    @staticmethod
    def nodes_to_str(nodes: List[RandomizerNode]) -> str:
        nodes_str = "\n".join([f"    {node}" for node in nodes])
        return nodes_str


class RandomUtils:

    @classmethod
    def random_value_for_param(cls, param: OperatorParam, rng_params: random.Random):
        if isinstance(param, OperatorParamNumber):
            return cls.random_value_for_number_param(param, rng_params)
        else:
            raise ValueError(f"Unsupported param type {type(param)}")

    @classmethod
    def random_value_for_number_param(cls, param: OperatorParamNumber, rng_params: random.Random) -> int:
        # TODO: support open intervals
        # TODO: store rng_params in test_context
        if param.type == float:
            return rng_params.uniform(param.min_value, param.max_value)
        elif param.type == int:
            return rng_params.randint(param.min_value, param.max_value)
        else:
            raise ValueError(f"Unsupported type {param.type}")

    @classmethod
    def constructor_kwargs(cls, operator: OperatorDefinition, constructor_kwargs: Dict[str, object], rng_params: random.Random) -> Dict:
        return {param.name: cls.random_value_for_param(param, rng_params) if param.name not in constructor_kwargs else constructor_kwargs[param.name] for param in operator.constructor_params}

    @classmethod
    def forward_kwargs(cls, operator: OperatorDefinition, forward_kwargs: Dict[str, object], rng_params: random.Random) -> Dict:
        return {param.name: cls.random_value_for_param(param, rng_params) if param.name not in forward_kwargs else forward_kwargs[param.name] for param in operator.forward_params}

    @classmethod
    def random_shape(cls,
                     rng_shape: random.Random,
                     dim_min: int,
                     dim_max: int,
                     op_size_min: int,
                     op_size_max: int,
                     microbatch_size_min: int,
                     microbatch_size_max: int,
        ) -> TensorShape:
        shape = [rng_shape.randint(op_size_min, op_size_max) for _ in range(rng_shape.randint(dim_min - 1, dim_max - 1))]
        microbatch_size = rng_shape.randint(microbatch_size_min, microbatch_size_max)
        shape.insert(0, microbatch_size)
        shape = tuple(shape)

        return shape

    @classmethod
    def random_shape_from_config(cls, randomizer_config: RandomizerConfig, rng_shape: random.Random) -> TensorShape:
        op_size_min = randomizer_config.op_size_per_dim_min
        op_size_max = randomizer_config.op_size_per_dim_max

        dim_min = randomizer_config.dim_min
        dim_max = randomizer_config.dim_max

        microbatch_size_min = randomizer_config.microbatch_size_min
        microbatch_size_max = randomizer_config.microbatch_size_max

        return cls.random_shape(
            rng_shape,
            dim_min=dim_min,
            dim_max=dim_max,
            op_size_min=op_size_min,
            op_size_max=op_size_max,
            microbatch_size_min=microbatch_size_min,
            microbatch_size_max=microbatch_size_max,
        )


class GraphUtils:

    @classmethod
    def get_input_shapes(cls, graph: RandomizerGraph) -> List[TensorShape]:
        input_shapes = [input_node.input_shape for input_node in graph.input_nodes]
        return input_shapes

    @classmethod
    def to_ops_str(cls, graph: RandomizerGraph) -> str:
        ops = [node.get_name() for node in graph.nodes]
        ops_str = " -> ".join(ops)
        return ops_str

    @classmethod
    def short_description(cls, graph: RandomizerGraph):
        return f"ops: ({cls.to_ops_str(graph)}) input_shapes: {cls.get_input_shapes(graph)}"

    # TODO support serialization/deserialization of RandomizerGraph
    @classmethod
    def to_str(cls, graph: RandomizerGraph):
        graph_dict = asdict(graph)
        # Serialize dictionary to YAML string
        yaml_str = yaml.dump(graph_dict)
        # yaml_str = json.dumps(graph.__dict__)
        return yaml_str


class NodeUtils:

    @staticmethod
    def is_previous_node(node: RandomizerNode, previous_node: RandomizerNode) -> bool:
        return node.index == previous_node.index + 1

    @staticmethod
    def is_open(node: RandomizerNode) -> bool:
        return (node.inputs is None or len(node.inputs) == 0) or (node.operator.input_num > 1 and len(node.inputs) < node.operator.input_num)

    @classmethod
    def get_open_nodes(cls, nodes: List[RandomizerNode]) -> List[RandomizerNode]:
        return [node for node in nodes if cls.is_open(node)]

    @classmethod
    def get_open_nodes_with_input_shape(cls, nodes: List[RandomizerNode], input_shape: TensorShape) -> List[RandomizerNode]:
        # TODO support checking not just next operand but all not connected operands
        return [node for node in nodes if cls.is_open(node) and node.input_shapes[len(node.inputs)] == input_shape]

    @classmethod
    def calc_input_shapes(cls, node: RandomizerNode, rng_shape: random.Random) -> List[TensorShape]:
        return node.operator.calc_input_shapes(node.operator, node.output_shape, rng_shape)


class DebugUtils:

    @classmethod
    def format_tensors(cls, tensors: List[pybuda.Tensor]):
        if isinstance(tensors[0], pybuda.Tensor):
            format_tensor: Callable[[pybuda.Tensor], str] = lambda t: f'{t.data_format}:{t.shape}'
        elif isinstance(tensors[0], torch.Tensor):
            format_tensor: Callable[[pybuda.Tensor], str] = lambda t: f'{t.type()}:{t.shape}'
        return [format_tensor(t) for t in tensors]
    
    @classmethod
    def debug_inputs(cls, inputs: List[pybuda.Tensor]):
        logger.info(f"inputs: {cls.format_tensors(inputs)}")
