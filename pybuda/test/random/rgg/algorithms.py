# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Implementation of randomization algrorithms


from typing import List
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
    def init_nodes(cls, test_context: RandomizerTestContext):
        """
        Initializes the nodes of a graph. 

        This method does three main things:
        1. Sets the index for each node.
        2. Stores output values if they are needed as explicit input for a later operator.
        3. Setting input nodes for open nodes.
        4. Generates random settings for operator parameters.

        Args:
            test_context (RandomizerTestContext): The test context.

        Raises:
            Exception: If the number of inputs for a node does not match the configured input number.
            Exception: If the node operator is not of type RandomizerOperator.

        Returns:
            None
        """
        graph = test_context.graph
        nodes = test_context.graph.nodes

        rng_shape = test_context.rng_shape
        rng_params = test_context.rng_params

        # Setting node.index
        op_index_cnt = 0
        for node in nodes:
            op_index_cnt += 1
            node.index = op_index_cnt

        # Storing output values if needed as explicit input for later operator
        logger.trace("Setting out_value for nodes")
        for node in nodes:
            # setting default output variable name
            node.out_value = "v"
            for input_node in node.inputs:
                if not NodeUtils.is_previous_node(node, input_node) or cls.always_unique_variables:
                    # overriding default output variable name
                    input_node.out_value = input_node.operator_name()
                    logger.trace(f"Set out_value = {input_node.out_value}")

        logger.trace("Setting input nodes for open nodes")
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

        logger.trace("Generating random settings for operator parameters")
        # Generate random values for operator parameters
        for node in nodes:
            node.constructor_kwargs = RandomUtils.constructor_kwargs(node.operator, node.constructor_kwargs, rng_params)
            node.forward_kwargs = RandomUtils.forward_kwargs(node.operator, node.forward_kwargs, rng_params)
        logger.trace("Random settings for operator parameters generated")

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
    def prepare_graph(cls, test_context: RandomizerTestContext):

        graph = test_context.graph

        logger.trace("Initializing nodes")
        cls.init_nodes(test_context)
        logger.trace("Nodes initialized")

        logger.trace("Validating graph")
        cls.validate_graph(graph)
        logger.trace("Graph validated")

        logger.trace("Serializing nodes")
        nodes_str = StrUtils.nodes_to_str(graph.nodes)
        logger.trace("Nodes serialized")
        logger.trace(f"Nodes: \n{nodes_str}")


class RandomGraphAlgorithm(GraphBuilder):
    '''Implementation of the random graph building algorithm'''

    def __init__(self, framework: Framework, randomizer_config):
        super(RandomGraphAlgorithm, self).__init__(randomizer_config)
        self.framework = framework
        self.operators = framework.operator_repository.operators

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
    # Graph contains between num_of_nodes_min and num_of_nodes_max nodes
    # Graph is constructed backwards starting from end node
    # In each step a random operator is selected and a new node is created
    # Output of new node is connected as input to the multiple open nodes randomly selected which has the same input shape
    # When new node is connected to more than one node, graph constructs a fork join
    # Output shape of first node is random
    # Output shape of other nodes is based on next input shape of a randomly picked open node
    # Input shapes for each node are calculated based on output shape of the node
    def build_graph(self, test_context: RandomizerTestContext):
        '''Implementation of the random graph building algorithm'''

        graph = test_context.graph
        nodes = graph.nodes

        rng_graph = test_context.rng_graph
        rng_shape = test_context.rng_shape

        fork_join_counter = 0
        fork_join_max = test_context.randomizer_config.num_fork_joins_max

        # Building the graph with number of nodes between num_of_nodes_min and num_of_nodes_max
        num_of_nodes = rng_graph.randint(self.randomizer_config.num_of_nodes_min, self.randomizer_config.num_of_nodes_max) 
        for node_index in range(num_of_nodes, 0, -1):
            first_node = node_index == num_of_nodes

            # Choose operator randomly based on rng
            op1 = self._get_random_operator(rng_graph)

            # Find all open nodes
            open_nodes = NodeUtils.get_open_nodes(nodes)

            # Select output shape for the new node
            if first_node:
                # For the first node set output shape as random shape
                output_shape = RandomUtils.random_shape_from_config(self.randomizer_config, rng_shape)
            else:
                # For other nodes, output shape is based on input shapes of a random open node
                # Select one of open nodes randomly
                random_open_node: RandomizerNode = rng_graph.choice(open_nodes)
                # Setting output shape based on input shapes of the random open node
                input_shapes = random_open_node.input_shapes
                output_shape = input_shapes[len(random_open_node.inputs)]

            # Find all other open nodes with input shape mathing the output shape of new node
            open_nodes = NodeUtils.get_open_nodes_with_input_shape(nodes, output_shape)

            # Random nodes are selected by matching the same input shape as new node
            # Closing multiple nodes will construct fork joins
            random_nodes: List[RandomizerNode]

            if not first_node:
                # There must be at least one node to close
                subset_count_min = max(1, len(open_nodes) // 2)
                subset_count_max = len(open_nodes)
                # Choose a random number of nodes to close
                subset_count = rng_graph.randint(subset_count_min, subset_count_max)

                # Limit number of fork joins
                subset_count = min(subset_count, fork_join_max - fork_join_counter + 1)

                # Increase fork join counter
                new_fork_join = subset_count - 1
                if new_fork_join > 0:
                    logger.trace(f"Constructing {new_fork_join} new fork join(s) from operator op{node_index} {op1.name}")
                fork_join_counter += new_fork_join

                # Select random subset of open nodes to close
                random_nodes = rng_graph.sample(open_nodes, subset_count)
            else:
                random_nodes = []

            # Closing nodes are all random open nodes
            closing_nodes = random_nodes

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
                    if closing_node.input_shapes[len(closing_node.inputs)] == node.output_shape:
                        closing_node.inputs.append(node)

            open_nodes.append(node)
            nodes.insert(0, node)

        logger.trace(f"Graph built with {len(nodes)} nodes")

        logger.trace("Preparing graph")
        GraphNodeSetup.prepare_graph(test_context)
        logger.trace("Graph prepared")
