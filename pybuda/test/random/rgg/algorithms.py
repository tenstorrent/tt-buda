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


class GraphNodeSetup:
    '''Common step for completion of setting up and validating of the graph
    after it's built by a graph builder algorithm'''

    # Whether to always generate unique variables for each node
    always_unique_variables = False

    @classmethod
    def init_nodes(cls, graph: RandomizerGraph, rng_params: random.Random):
        """
        Initializes the nodes of a graph. 

        This method does three main things:
        1. Sets the index for each node.
        2. Stores output values if they are needed as explicit input for a later operator.
        3. Setting input nodes for open nodes.
        4. Generates random settings for operator parameters.

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
            input_shapes = node.input_shapes
            for i in range(len(node.inputs), node.operator.input_num):
                input_nodes_with_same_shape = [input_node for input_node in graph.input_nodes if input_node.input_shape == input_shapes[i] and input_node not in node.inputs]
                if len(input_nodes_with_same_shape) > 0:
                    # reuse existing input node with the same shape that is not already connected to the node
                    input_node = input_nodes_with_same_shape[0]
                else:
                    input_node = RandomizerInputNode(out_value=f"in_value{len(graph.input_nodes)+1}", input_shape=input_shapes[i])
                    graph.input_nodes.append(input_node)
                node.inputs.append(input_node)

        # Generate random values for operator parameters
        for node in nodes:
            node.constructor_kwargs = RandomUtils.constructor_kwargs(node.operator, node.constructor_kwargs, rng_params)
            node.forward_kwargs = RandomUtils.forward_kwargs(node.operator, node.forward_kwargs, rng_params)

    @classmethod
    def validate_graph(cls, graph: RandomizerGraph):
        '''Validates the graph
        1. Validates the number of inputs for each node
        2. Validates operator class type

        Args:
            graph (RandomizerGraph): The graph to validate

        Raises:
            Exception: If the number of inputs for a node does not match the configured input number.
            Exception: If the node operator is not of type RandomizerOperator.
        '''
        nodes = graph.nodes

        # Validation of input configuration
        for node in nodes:
            if node.operator.input_num and node.operator.input_num > 1:
                if len(node.inputs) != node.operator.input_num:
                    raise Exception(f"Expected {node.operator.input_num} number of inputs but configured {node.inputs}")

        # Validation of operator and layer types
        for node in nodes:
            if node.operator and not isinstance(node.operator, OperatorDefinition):
                raise Exception(f"Step operator is wrong type {node.node_info()} expected RandomizerOperator got {type(node.operator)}")

    @classmethod
    def prepare_graph(cls, graph: RandomizerGraph, rng_params: random.Random):
        cls.init_nodes(graph, rng_params)
        cls.validate_graph(graph)

        nodes_str = StrUtils.nodes_to_str(graph.nodes)
        logger.trace(f"Nodes: \n{nodes_str}")


class RandomGraphAlgorithm(GraphBuilder):
    '''Implementation of the random graph building algorithm'''

    SKIP_OPERATORS = (
        "sqrt",  # skip because it's failing for negative values
        # "linear",
        "conv2d",  # skip until calc_input_shapes is properly implemented
    )

    def __init__(self, framework: Framework, randomizer_config):
        super(RandomGraphAlgorithm, self).__init__(randomizer_config)
        self.framework = framework
        self.operators = [
            op for op in framework.operator_repository.operators
            if op.name not in self.SKIP_OPERATORS
        ]

    def _get_random_operator(self, rng):
        return rng.choice(self.operators)

    def _init_default_constructor_params(self, node: RandomizerNode):
        '''Initializing default constructor parameters based on input and output shapes'''
        # Operator specific settings
        # TODO abstract this
        if len([param for param in node.operator.constructor_params if param.name == "in_features"]) == 1:
            node.constructor_kwargs["in_features"] = node.input_shapes[0][-1]
        if len([param for param in node.operator.constructor_params if param.name == "out_features"]) == 1:
            node.constructor_kwargs["out_features"] = node.output_shape[-1]
        if len([param for param in node.operator.constructor_params if param.name == "in_channels"]) == 1:
            node.constructor_kwargs["in_channels"] = node.input_shapes[0][1]
        if len([param for param in node.operator.constructor_params if param.name == "out_channels"]) == 1:
            node.constructor_kwargs["out_channels"] = node.output_shape[1]

    # Build graph of random operators via random graph building algorithm
    # Graph contains between num_of_nodes/2 and num_of_nodes nodes
    # Graph is constructed backwards starting from end node
    # In each step a random operator is selected and a new node is created
    # New node is connected to the last node and optionally to a random node with the same input shape
    # When new node is connected to 2 nodes graph contains a fork join
    # Input shapes for each node are calculated based on output shape of the node
    def build_graph(self, test_context: RandomizerTestContext):
        '''Implementation of the random graph building algorithm'''
        parameters = test_context.parameters
        graph = test_context.graph
        nodes = graph.nodes

        # Initialize random number generators for graph building
        rng_graph = random.Random(parameters.random_seed)

        # Initialize random number generators for shape generation
        rng_shape = random.Random(test_context.parameters.random_seed)

        # Initialize random number generators for parameters
        rng_params = random.Random(test_context.parameters.random_seed)

        num_of_nodes = self.randomizer_config.num_of_nodes

        # Building the graph with number of nodes between n/2 and n
        # num_of_nodes defines max number of nodes in the graph
        for _ in range(rng_graph.randint(int(num_of_nodes/2), num_of_nodes)):
            # Choose operator randomly based on rng
            op1 = self._get_random_operator(rng_graph)

            # Last node defines output shape for next node to create
            last_node: RandomizerNode = None
            # Random node is selected by matching the same input shape to support fork joins
            # TODO random_node -> random_nodes, select all random_nodes instead of just one
            # TODO: obsolete last_node in flavor of random_nodes
            random_node: RandomizerNode = None

            if len(nodes) > 0:
                # If graph is not empty find previusly added node
                last_node = nodes[0]

            if len(nodes) == 0:
                # Setting output shape for the first node
                output_shape = RandomUtils.random_shape_from_config(self.randomizer_config, rng_shape)
            else:
                # Setting output shape based on last node input shapes
                input_shapes = last_node.input_shapes
                output_shape = input_shapes[len(last_node.inputs)]

            # Find open nodes with input shape mathing the output shape of new node
            open_nodes = NodeUtils.get_open_nodes_with_input_shape(nodes, output_shape)

            if len(open_nodes) > 0:
                # Randomly selecting one of the open nodes
                random_node = rng_graph.choice(open_nodes)

            if last_node is not None and random_node is not None and last_node == random_node:
                # Skip random_node if it's the same as last_node
                random_node = None

            # Closing nodes are last_node and optionally random_node
            closing_nodes = [closing_node for closing_node in [last_node, random_node] if closing_node is not None]

            # Creating new node
            node = RandomizerNode(operator=op1, output_shape=output_shape)
            # Saving input shapes for the new node
            node.input_shapes = NodeUtils.calc_input_shapes(node, rng_shape)

            # Initializing default constructor parameters based on input and output shapes
            self._init_default_constructor_params(node)

            for closing_node in closing_nodes:
                for _ in range(rng_graph.randint(1, closing_node.operator.input_num - len(closing_node.inputs))):
                    # currently only if next input of closing node matches the output shape a closing node will be actually closed
                    # TODO check all inputs for matching shapes not just next one
                    # if second operands is different shape than first one it will most likely not be closed with an internal node but with external input
                    # e.x. second operand of matmul usually connect to external input instead of an internal node
                    if closing_node.input_shapes[len(closing_node.inputs)] == node.output_shape:
                        closing_node.inputs.append(node)

            open_nodes.append(node)
            nodes.insert(0, node)

        GraphNodeSetup.prepare_graph(graph, rng_params)
