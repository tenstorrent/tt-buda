# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Implementation of randomization algrorithms


from typing import List
from loguru import logger

from pybuda.op_repo import OperatorDefinition

from .datatypes import RandomizerGraph, RandomizerTestContext
from .datatypes import NodeShapeCalculationContext
from .datatypes import RandomizerInputNode
from .datatypes import RandomizerConstantNode
from .base import RandomizerNode, GraphBuilder
from .base import Framework
from .utils import RandomUtils, StrUtils, NodeUtils
from .utils import RateLimitter


class GraphNodeSetup:
    '''Common step for completion of setting up and validating of the graph
    after it's built by a graph builder algorithm'''

    # Whether to always generate unique variables for each node
    always_unique_variables = False

    @classmethod
    def init_nodes_names(cls, test_context: RandomizerTestContext):
        """
        Initializes the nodes names of a graph. 

        This method does following things:
        1. Sets the index for each node.
        2. Stores output values if they are needed as explicit input for a later operator.
        """

        nodes = test_context.graph.nodes

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
                if input_node is not None and not input_node.constant and (not NodeUtils.is_previous_node(node, input_node) or cls.always_unique_variables):
                    # overriding default output variable name
                    input_node.out_value = input_node.operator_name
                    logger.trace(f"Set out_value = {input_node.out_value}")

    @classmethod
    def init_nodes_inputs(cls, test_context: RandomizerTestContext):
        """
        Setting input and contant nodes for open nodes.

        Args:
            test_context (RandomizerTestContext): The test context.

        Returns:
            None
        """
        graph = test_context.graph
        nodes = test_context.graph.nodes

        rng_shape = test_context.rng_shape

        constant_input_rate_limitter = RateLimitter(rng_shape, 100, test_context.randomizer_config.constant_input_rate)
        same_inputs_rate_limitter = RateLimitter(rng_shape, 100, test_context.randomizer_config.same_inputs_percent_limit)

        logger.trace("Setting input nodes for open nodes")
        open_nodes = NodeUtils.get_open_nodes(nodes)
        logger.trace(f"Open nodes {StrUtils.nodes_to_str(open_nodes)}")

        # Setting input nodes for open nodes
        for node in open_nodes:
            input_shapes = node.input_shapes
            # list of input nodes that are already connected to the node
            used_input_nodes: List[RandomizerInputNode] = []
            for open_input_index in NodeUtils.get_open_input_indices(node):
                input_shape = input_shapes[open_input_index]

                # There must be at least one input node for forward method
                if len(graph.input_nodes) > 0 and constant_input_rate_limitter.is_allowed():
                    # Creates a new constant node with the same shape
                    constant_node = RandomizerConstantNode(out_value=None, input_shape=input_shape)
                    logger.trace(f"Allowed constant input {constant_node.out_value} -> {node.name}[{open_input_index}] due to rate limit not exceeded: {constant_input_rate_limitter.limit_info()}")
                    # Stores the new constant node in the graph constant nodes
                    graph.constant_nodes.append(constant_node)
                    input_node = constant_node
                else:
                    # list of all graph input nodes with the same shape as the input shape
                    input_nodes_with_same_shape = [input_node for input_node in graph.input_nodes if input_node.input_shape == input_shape]
                    # list of input nodes with the same shape that are not already connected to the node
                    input_nodes_with_same_shape_unused = [input_node for input_node in input_nodes_with_same_shape if input_node not in used_input_nodes]
                    if len(input_nodes_with_same_shape_unused) > 0:
                        # reuse existing input node with the same shape that is not already connected to the node
                        input_node = input_nodes_with_same_shape_unused[0]
                        used_input_nodes.append(input_node)
                    else:
                        # there are no input nodes with the same shape that are not already connected to the node
                        # check if same input value is allowed
                        # there must be at least one input node with the same shape to allow repeat
                        allow_repeat = len(input_nodes_with_same_shape) > 0

                        if allow_repeat:
                            if not same_inputs_rate_limitter.is_allowed():
                                logger.trace(f"Not allowed same input value {input_node.out_value} -> {node.name}[{open_input_index}] due to rate limit exceeded: {same_inputs_rate_limitter.limit_info()}")
                                allow_repeat = False

                        if allow_repeat:
                            input_node = rng_shape.choice(input_nodes_with_same_shape)
                            logger.trace(f"Allowed same input value {input_node.out_value} -> {node.name}[{open_input_index}] due to rate limit not exceeded: {same_inputs_rate_limitter.limit_info()}")
                        
                        else:
                            # create a new input node with the same shape since there are no unused input nodes with the same shape or repeat is not allowed
                            input_node = RandomizerInputNode(out_value=f"in_value{len(graph.input_nodes)+1}", input_shape=input_shape)
                            used_input_nodes.append(input_node)
                            # store the new input node in the graph input nodes
                            graph.input_nodes.append(input_node)
                    
                # connect the input node to the open node input
                node.inputs[open_input_index] = input_node

        # Assign constant node values after connecting inputs
        iconst_index = 0
        for i, constant_node in enumerate(graph.constant_nodes):
            if constant_node.out_value is None:
                iconst_index += 1
                constant_node.out_value = f"iconst{iconst_index}"

    @classmethod
    def init_node_params(cls, node: RandomizerNode, test_context: RandomizerTestContext):
        """
        Generates random parameters for specified node.

        Args:
            node (RandomizerNode): The node.
            test_context (RandomizerTestContext): The test context.

        Returns:
            None
        """
        rng_params = test_context.rng_params

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
            if node.input_num and node.input_num > 1:
                if NodeUtils.num_of_open_inputs(node) > 0:
                    raise Exception(f"Closed {NodeUtils.num_of_closed_inputs(node)}/{node.input_num} inputs, missing {NodeUtils.num_of_open_inputs(node)} inputs for node {node.node_info}")

        # Validation of operator and layer types
        for node in nodes:
            if node.operator and not isinstance(node.operator, OperatorDefinition):
                raise Exception(f"Step operator is wrong type {node.node_info} expected RandomizerOperator got {type(node.operator)}")

    @classmethod
    def prepare_graph(cls, test_context: RandomizerTestContext):

        graph = test_context.graph

        logger.trace("Initializing nodes")
        cls.init_nodes_names(test_context)
        cls.init_nodes_inputs(test_context)
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

    @classmethod
    def _init_default_constructor_params(cls, node: RandomizerNode):
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

        constant_input_rate_limitter = RateLimitter(rng_shape, 100, test_context.randomizer_config.constant_input_rate)
        same_inputs_rate_limitter = RateLimitter(rng_shape, 100, test_context.randomizer_config.same_inputs_percent_limit)

        # Context object for shape calculation, node will be set later in the loop
        shape_calculation_context = NodeShapeCalculationContext(node=None, test_context=test_context)

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
                open_input_indices = [i for i in NodeUtils.get_open_input_indices(random_open_node)]
                open_input_index = open_input_indices[rng_graph.randint(0, len(open_input_indices) - 1)]
                output_shape = input_shapes[open_input_index]

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

                if len(random_nodes) > 1:
                    for random_node in random_nodes[1:]:
                        logger.trace(f"Constructing new fork join from operator op{node_index} {op1.name} -> {random_node.name}")

            else:
                random_nodes = []

            # Closing nodes are all random open nodes
            closing_nodes = random_nodes

            # Creating new node
            node = RandomizerNode(operator=op1, output_shape=output_shape)

            # Initializing node parameters
            # Calculating input shapes may require input parameters for its calculation
            GraphNodeSetup.init_node_params(node, test_context)

            # Saving input shapes for the new node
            shape_calculation_context.node = node
            node.input_shapes = NodeUtils.calc_input_shapes(node, shape_calculation_context)

            # Initializing default constructor parameters based on input and output shapes
            self._init_default_constructor_params(node)

            for closing_node in closing_nodes:
                node_connected = False
                for open_input_index in NodeUtils.get_open_input_indices(closing_node):
                    # check input shape of a closing node open input
                    if closing_node.input_shapes[open_input_index] == node.output_shape:

                        # Limit number of same inputs on same node
                        if node_connected:
                            if not same_inputs_rate_limitter.is_allowed():
                                logger.trace(f"Skipping same input node connection op{node_index} {node.name} -> {closing_node.name}[{open_input_index}] due to rate limit exceeded: {same_inputs_rate_limitter.limit_info()}")
                                continue
                            else:
                                logger.trace(f"Allowed same input node connection op{node_index} {node.name} -> {closing_node.name}[{open_input_index}] due to rate limit not exceeded: {same_inputs_rate_limitter.limit_info()}")
                        closing_node.inputs[open_input_index] = node
                        node_connected = True

            nodes.insert(0, node)

            # Connecting constants randomly to current node inputs
            open_nodes = NodeUtils.get_open_nodes(nodes)
            open_nodes_count = len(open_nodes)
            input_shapes = node.input_shapes
            for open_input_index in NodeUtils.get_open_input_indices(node):
                input_shape = input_shapes[open_input_index]
                # Skip connecting constant input for last open input to avoid disconnected graph
                if open_nodes_count > 1 or NodeUtils.num_of_open_inputs(node) > 1:
                    if constant_input_rate_limitter.is_allowed():
                        # Creates a new constant node with the same shape
                        constant_node = RandomizerConstantNode(out_value=None, input_shape=input_shape)
                        logger.trace(f"Allowed constant input {constant_node.out_value} -> {node.name}[{open_input_index}] due to rate limit not exceeded: {constant_input_rate_limitter.limit_info()}")
                        # Stores the new constant node in the graph constant nodes
                        graph.constant_nodes.insert(0, constant_node)
                        # Connects the input node to the open node input
                        node.inputs[open_input_index] = constant_node

        # Assign constant node values
        for i, constant_node in enumerate(graph.constant_nodes):
            constant_node.out_value = f"nconst{i+1}"

        logger.trace(f"Graph built with {len(nodes)} nodes")

        logger.trace("Preparing graph")
        GraphNodeSetup.prepare_graph(test_context)
        logger.trace("Graph prepared")
