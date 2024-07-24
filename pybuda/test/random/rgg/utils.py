# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Utility functions


import random
import signal
from typing import Callable, Generator, List, Dict
from dataclasses import asdict
from loguru import logger
import re
import yaml

import torch
import pybuda

from pybuda.op_repo import OperatorParam, OperatorDefinition, OperatorParamNumber

from .datatypes import TensorShape
from .datatypes import RandomizerConfig, RandomizerTestContext, RandomizerNode, RandomizerGraph
from .datatypes import NodeShapeCalculationContext


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

    @staticmethod
    def text_to_snake_case(text: str) -> str:
        text = text.lower()
        pattern = re.compile(r'\ +')
        snake_case = re.sub(pattern, '_', text).lower()
        return snake_case

    @classmethod
    def test_id(cls, test_context: RandomizerTestContext) -> str:
        parameters = test_context.parameters
        graph_builder_snake_case = cls.camel_case_to_snake_case(parameters.graph_builder_name)
        test_name = cls.text_to_snake_case(test_context.test_name)
        test_id = f"{parameters.framework_name}_{graph_builder_snake_case}_{test_name}_{parameters.test_index}_{parameters.random_seed}"
        return test_id

    @staticmethod
    def nodes_to_str(nodes: List[RandomizerNode]) -> str:
        '''Converts list of nodes to string representation
        Used for debugging purposes
        
        Args:
            nodes (List[RandomizerNode]): list of nodes

        Returns:
            str: string representation of nodes
        '''
        # TODO Very slow -> implement in a faster way
        # nodes_str = "\n".join([f"    {node}" for node in nodes])
        nodes_str = ""
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
    def quantize(cls, value: int, quantization: int = 2) -> int:
        '''Quantize the value to the nearest multiple of quantization

        Args:
            value (int): value to quantize
            quantization (int, optional): quantization factor. Defaults to 2.

        Returns:
            int: quantized value
        '''
        # Using max to avoid quantizing to 0
        return max(round(value / quantization) * quantization, quantization)

    @classmethod
    def random_shape(cls,
                     rng_shape: random.Random,
                     dim_min: int,
                     dim_max: int,
                     op_size_min: int,
                     op_size_max: int,
                     quantization: int,
                     microbatch_size_min: int,
                     microbatch_size_max: int,
        ) -> TensorShape:
        shape = [cls.quantize(rng_shape.randint(op_size_min, op_size_max), quantization) for _ in range(rng_shape.randint(dim_min - 1, dim_max - 1))]
        microbatch_size = rng_shape.randint(microbatch_size_min, microbatch_size_max)
        shape.insert(0, microbatch_size)
        shape = tuple(shape)

        return shape

    @classmethod
    def random_shape_from_config(cls, randomizer_config: RandomizerConfig, rng_shape: random.Random) -> TensorShape:
        op_size_min = randomizer_config.op_size_per_dim_min
        op_size_max = randomizer_config.op_size_per_dim_max
        op_size_quantization = randomizer_config.op_size_quantization

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
            quantization=op_size_quantization,
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
        ops = [node.name for node in graph.nodes]
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

    @classmethod
    def num_of_open_inputs(cls, node: RandomizerNode) -> int:
        return node.inputs.count(None)

    @classmethod
    def num_of_closed_inputs(cls, node: RandomizerNode) -> int:
        return node.input_num - cls.num_of_open_inputs(node)

    @classmethod
    def is_open(cls, node: RandomizerNode) -> bool:
        return cls.num_of_open_inputs(node) > 0

    # TODO replace list with generator
    @classmethod
    def get_open_nodes(cls, nodes: List[RandomizerNode]) -> List[RandomizerNode]:
        return [node for node in nodes if cls.is_open(node)]

    @classmethod
    def has_open_input_with_input_shape(cls, node: RandomizerNode, input_shape: TensorShape) -> bool:
        for i, open_input in enumerate(node.inputs):
            if open_input is None:
                if input_shape == node.input_shapes[i]:
                    return True
        return False

    @classmethod
    def get_open_input_indices(cls, node: RandomizerNode) -> Generator[int, None, None]:
        for i, open_input in enumerate(node.inputs):
            if open_input is None:
                yield i

    # TODO replace list with generator
    @classmethod
    def get_open_nodes_with_input_shape(cls, nodes: List[RandomizerNode], input_shape: TensorShape) -> List[RandomizerNode]:
        return [node for node in nodes if cls.is_open(node) and cls.has_open_input_with_input_shape(node, input_shape)]

    @classmethod
    def calc_input_shapes(cls, node: RandomizerNode, shape_calculation_context: NodeShapeCalculationContext) -> List[TensorShape]:
        return node.operator.calc_input_shapes(shape_calculation_context)

    @classmethod
    def get_random_input_num(cls, node: RandomizerNode, test_context: RandomizerTestContext) -> int:
        input_num_range = node.operator.input_num_range
        return test_context.rng_graph.randint(input_num_range.operands_min, input_num_range.operands_max)

    @classmethod
    def init_random_inputs(cls, node: RandomizerNode, test_context: RandomizerTestContext) -> None:
        node.input_num = cls.get_random_input_num(node, test_context)
        node.init_inputs()


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


class RateLimitter:
    '''Rate limitter class to limit the number of allowed operations by a rate limit factor'''

    def __init__(self, rng: random.Random, max_limit: int, current_limit: int):
        self.rng = rng
        self.max_limit = max_limit
        self.current_limit = current_limit
        self.current_value: int = None

    def is_allowed(self) -> bool:
        '''Check if the operation is allowed by the rate limit factor and current random value'''
        self.current_value = self.rng.randint(0, self.max_limit)
        return self.current_value < self.current_limit
    
    def limit_info(self) -> str:
        '''Return the rate limit info for previous operation'''
        if self.current_value < self.current_limit:
            return f"{self.current_value} < {self.current_limit}"
        else:
            return f"{self.current_value} >= {self.current_limit}"


class TimeoutException(Exception):
    pass


# Handler for timeout signal
def timeout_handler(signum, frame):
    raise TimeoutException


# Decorator for time limiting
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            # Set alarm
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Shutdown alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

