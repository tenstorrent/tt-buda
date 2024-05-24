# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Implementation of randomization algrorithms


import random
from loguru import logger

from pybuda.op_repo import OperatorDefinition

from .datatypes import RandomizerGraph, RandomizerTestContext
from .datatypes import RandomizerInputNode
from .base import RandomizerNode, GraphBuilder
from .base import Framework
from .utils import RandomUtils, StrUtils, NodeUtils


class NodesUtils:

    # Whether to always generate unique variables for each node
    always_unique_variables = False

    @classmethod
    def init_nodes(cls, graph: RandomizerGraph):
        """
        Initializes the nodes of a graph. 

        This method does three main things:
        1. Sets the index for each node.
        2. Stores output values if they are needed as explicit input for a later operator.
        3. Setting input nodes for open nodes
        4. Validates the input configuration.

        Args:
            graph (RandomizerGraph): The graph nodes to initialize.

        Raises:
            Exception: If the number of inputs for a node does not match the configured input number.
            Exception: If the node operator is not of type RandomizerOperator.

        Returns:
            None
        """
        nodes = graph.nodes

        # Setting node.index
        op_index_cnt = 0
        for node in nodes:
            op_index_cnt += 1
            node.index = op_index_cnt

        # Storing output values if needed as explicit input for later operator
        for node in nodes:
            # setting default output variable name
            node.out_value = "v"
            for input_node in node.inputs:
                if not NodeUtils.is_previous_node(node, input_node) or cls.always_unique_variables:
                    # overriding default output variable name
                    input_node.out_value = input_node.operator_name()
                    logger.trace(f"Set out_value = {input_node.out_value}")

        open_nodes = NodeUtils.get_open_nodes(nodes)
        logger.trace(f"Open nodes {StrUtils.nodes_to_str(open_nodes)}")

        # Setting input nodes for open nodes
        for node in open_nodes:
            for _ in range(node.operator.input_num - len(node.inputs)):
                input_node = RandomizerInputNode(out_value=f"in_value{len(graph.input_nodes)+1}")
                graph.input_nodes.append(input_node)
                node.inputs.append(input_node)

        # Validation of input configuration
        for node in nodes:
            if node.operator.input_num and node.operator.input_num > 1:
                if len(node.inputs) != node.operator.input_num:
                    raise Exception(f"Expected {node.operator.input_num} number of inputs but configured {node.inputs}")

        # Validation of operator and layer types
        for node in nodes:
            if node.operator and not isinstance(node.operator, OperatorDefinition):
                raise Exception(f"Step operator is wrong type {node.node_info()} expected RandomizerOperator got {type(node.operator)}")

        nodes_str = StrUtils.nodes_to_str(nodes)
        logger.trace(f"Nodes: \n{nodes_str}")


class RandomGraphAlgorithm(GraphBuilder):

    SKIP_OPERATORS = (
        "matmul",  # skip matmul until shape calculation support is added
    )

    def __init__(self, framework: Framework, randomizer_config):
        super(RandomGraphAlgorithm, self).__init__(randomizer_config)
        self.framework = framework
        self.operators = [
            op for op in framework.operator_repository.operators
            if
                not op.is_layer() and
                op.name not in self.SKIP_OPERATORS
        ]

    def get_single_input_operator(self, rng):
        return rng.choice(self.operators)

    def build_graph(self, test_context: RandomizerTestContext):
        parameters = test_context.parameters
        graph = test_context.graph
        nodes = graph.nodes

        # Initialize random number generators for graph building
        rng_graph = random.Random(parameters.random_seed)

        num_of_nodes = self.randomizer_config.num_of_nodes

        for _ in range(rng_graph.randint(int(num_of_nodes/2), num_of_nodes)):
            # Choose operator randomly based on rng
            op1 = self.get_single_input_operator(rng_graph)

            # if op1.is_layer:
            #     # Layers require number of input and output features
            #     nodes.append(RandomizerNode(operator=op1, in_features=cols1, out_features=cols2))
            # else:
            #     nodes.append(RandomizerNode(operator=op1))

            open_nodes = NodeUtils.get_open_nodes(nodes)

            last_node: RandomizerNode = None
            random_node: RandomizerNode = None

            if len(nodes) > 0:
                last_node = nodes[0]

            if len(open_nodes) > 0:
                random_node = rng_graph.choice(open_nodes)

            if last_node is not None and random_node is not None and last_node == random_node:
                random_node = None

            closing_nodes = [closing_node for closing_node in [last_node, random_node] if closing_node is not None]

            node = RandomizerNode(operator=op1)

            for closing_node in closing_nodes:
                for _ in range(rng_graph.randint(1, closing_node.operator.input_num - len(closing_node.inputs))):
                    closing_node.inputs.append(node)

            open_nodes.append(node)
            nodes.insert(0, node)

        NodesUtils.init_nodes(graph)

        # Initialize random number generators for shape generation
        rng_shape = random.Random(test_context.parameters.random_seed)

        # Provide input shapes for validation
        # TODO support operands with different shapes
        input_shape = RandomUtils.random_shape_from_config(self.randomizer_config, rng_shape)
        for input_node in test_context.graph.input_nodes:
            input_node.input_shape = input_shape
