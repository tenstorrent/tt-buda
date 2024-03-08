# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Base classes for random randomizer generator


from typing import Type
from loguru import logger
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader
import os

from pybuda import PyBudaModule
from pybuda.verify import verify_module, VerifyConfig
from pybuda.op_repo import OperatorDefinition, OperatorRepository
from test.conftest import TestDevice
from .datatypes import RandomizerNode, RandomizerGraph, RandomizerParameters, RandomizerConfig, ExecutionContext
from .datatypes import RandomizerTestContext
from .utils import RandomUtils, StrUtils


class GraphBuilder:
    """
    GraphBuilder is an interface that each graph building algorithm should implement.
    GraphBuilder encapsulates the logic for generating random graphs.
    """
    def __init__(self, randomizer_config: RandomizerConfig):
        self.randomizer_config = randomizer_config

    def build_graph(self, parameters: RandomizerParameters) -> RandomizerGraph:
        """
        Generate test graph with input shape needed for validation.

        Args:
            parameters (RandomizerParameters): The parameters for the randomizer.

        Returns:
            RandomizerGraph: The randomized graph model with input shape.

        Raises:
            Exception: This method is not implemented.
        """
        raise Exception("Method build_graph() not implemented")

    def get_name(self):
        return self.__class__.__name__


class ModelBuilder:
    '''
    ModelBuilder is an interface that each framework should implement for instantiated model instances from a previously generated test model class.
    '''

    def build_model(self, graph: RandomizerGraph, GeneratedTestModel: Type) -> PyBudaModule:
        raise Exception("Method build_model() not implemented")


@dataclass
class Framework:

    framework_name: str
    ModelBuilderType: Type[ModelBuilder]
    operator_repository: OperatorRepository


# Translates randomized graph into framework NN model code
class RandomizerCodeGenerator:

    def __init__(self, template_dir: str):
        self.template = Environment(loader=FileSystemLoader(template_dir)).get_template('generated_model.jinja2')

    def constructor_kwargs(self, node: RandomizerNode):
        constructor_kwargs = RandomUtils.constructor_kwargs(node.operator)
        return StrUtils.kwargs_str(**constructor_kwargs)

    def forward_args(self, node: RandomizerNode) -> str:
        args_str = ",".join([f"inputs[{i}]" for i in range(node.operator.input_num)])
        return args_str
    
    def forward_kwargs(self, node: RandomizerNode) -> str:
        forward_kwargs = RandomUtils.forward_kwargs(node.operator)
        return StrUtils.kwargs_str(**forward_kwargs)

    # TODO obsolete by constructor_kwargs
    def build_layer(self, node: RandomizerNode) -> str:
        if node.operator.is_layer() and node.operator.full_name is not None:
            return f"{node.operator.full_name}({node.in_features}, {node.out_features} {self.constructor_kwargs(node)})"
        else:
            raise Exception(f"Unsupported layer building for {node.node_info()}")

    # def call_operator(self, ctx: ExecutionContext) -> str:
    #     if ctx.node.operator.is_operator() and ctx.node.operator.full_name is not None:
    #         v = f"{ctx.node.operator.full_name}('{ctx.node.node_name()}', {self.forward_args(ctx.node)} {self.forward_kwargs(ctx.node)})"
    #     else:
    #         raise Exception(f"Unsupported operator call for {ctx.node.node_info()}")
    #     return v

    def generate_code(self, test_context: RandomizerTestContext, test_format: bool = True) -> str:
        # TODO setup random seed in generated test function

        parameters = test_context.parameters
        template = self.template

        code_str = template.render(
            graph_builder_name = parameters.graph_builder_name,
            test_id = StrUtils.test_id(test_context),
            test_format = test_format,
            test_index = parameters.test_index,
            random_seed = parameters.random_seed,
            graph=test_context.graph,
            nodes=test_context.graph.nodes,
            build_layer=self.build_layer,
            # call_operator=self.call_operator,
            constructor_kwargs=self.constructor_kwargs,
            forward_args=self.forward_args,
            forward_kwargs=self.forward_kwargs,
            ExecutionContext=ExecutionContext,
            )

        return code_str


class RandomizerModelProviderFromSourceCode:

    def __init__(self, code_generator: RandomizerCodeGenerator, model_builder: ModelBuilder):
        self.code_generator = code_generator
        self.model_builder = model_builder

    def build_model(self, test_context: RandomizerTestContext) -> PyBudaModule:
        '''
        Build model from generated test model class.

        Args:
            test_context (RandomizerTestContext): The context for the randomizer test.

        Returns:
            PyBudaModule: The PyBuda model.
        '''
        GeneratedTestModel = self._get_model_class(test_context)
        model = self.model_builder.build_model(test_context, GeneratedTestModel)
        return model

    def _get_model_class(self, test_context: RandomizerTestContext) -> Type:
        class_name = self._get_model_class_name(test_context)
        test_code_str = self.code_generator.generate_code(test_context, test_format=False)

        GeneratedTestModel = self._get_model_class_from_code(class_name, test_code_str)
        return GeneratedTestModel

    def _get_model_class_name(self, test_context: RandomizerTestContext):
        parameters = test_context.parameters
        class_name = f"GeneratedTestModel_{parameters.test_index}_{parameters.random_seed}"
        return class_name

    def _get_model_class_from_code(self, class_name: str, class_string: str) -> Type:
        # python magic, create class from class code string
        exec(class_string, globals())

        # renaming of random class name to GeneratedTestModel
        global GeneratedTestModel
        GeneratedTestModel = None
        exec(f"GeneratedTestModel = {class_name}", globals())
        return GeneratedTestModel


class RandomizerRunner:
    """
    The RandomizerRunner class is used for processing randomized tests.

    Attributes:
        test_context (RandomizerTestContext): The context for the randomizer test.
        model_provider (RandomizerModelProviderFromSourceCode): The model provider for generating tests.

    Methods:
        init_nodes(): Initializes the nodes for generating tests. Sets the index for each node and
                      stores output values if they are needed as explicit input for a later operator.
    """
    def __init__(self, test_context: RandomizerTestContext, modelBuilder: ModelBuilder):
        self.test_context = test_context
        self.code_generator = RandomizerCodeGenerator(f"pybuda/test/random/rgg/{test_context.parameters.framework_name.lower()}")
        self.model_provider = RandomizerModelProviderFromSourceCode(self.code_generator, modelBuilder)

    def init_nodes(self, graph: RandomizerGraph):
        """
        Initializes the nodes for generating tests. 

        This method does three main things:
        1. Sets the index for each node.
        2. Stores output values if they are needed as explicit input for a later operator.
        3. Validates the input configuration.

        Args:
            graph (RandomizerGraph): The model configuration for generating tests.

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
            if node.inputs:
                for input_node in node.inputs:
                    input_node: RandomizerNode = input_node
                    input_node.out_value = input_node.operator_name()
                    logger.trace(f"Set out_value = {input_node.out_value}")

        # Validation of input configuration
        for node in nodes:
            if node.operator.input_num and node.operator.input_num > 1:
                if len(node.inputs) != node.operator.input_num:
                    raise Exception(f"Expected {node.operator.input_num} number of inputs but configured {node.inputs}")

        # Validation of operator and layer types
        for node in nodes:
            if node.operator and not isinstance(node.operator, OperatorDefinition):
                raise Exception(f"Step operator is wrong type {node.node_info()} expected RandomizerOperator got {type(node.operator)}")

        nodes_str = "\n".join([f"    {node}" for node in nodes])
        logger.debug(f"Nodes: \n{nodes_str}")

    def generate_code(self) -> str:
        """
        Generates a test source code with test function for the randomized graph.

        Returns:
            str: The generated code.
        """
        return self.code_generator.generate_code(self.test_context, test_format=True)

    def build_model(self) -> PyBudaModule:
        model = self.model_provider.build_model(self.test_context)
        return model

    def verify(self, model: PyBudaModule) -> None:
        """
        Verify the model by building it and performing validation via PyBuda.
        The method is usually implemented once per framework.

        Args:
            test_context (RandomizerTestContext): The context for the randomizer test.
            model (PyBudaModule): The PyBuda model to verify.

        Raises:
            Exception: This method is not implemented.
        """

        parameters = self.test_context.parameters
        input_shape = self.test_context.graph.input_shape

        # verify PyBuda model
        verify_module(model, [input_shape],
                      VerifyConfig(devtype=parameters.test_device.devtype, arch=parameters.test_device.arch))

    def save_test(self, test_code_str: str):
        test_dir = self.test_context.randomizer_config.test_dir
        test_code_file_name = f"{test_dir}/test_gen_model_{StrUtils.test_id(self.test_context)}.py"

        if not os.path.exists(test_dir):
            logger.info(f"Creating test directory {test_dir}")
            os.makedirs(test_dir)

        logger.info(f"Saving test to {test_code_file_name}")
        with open(test_code_file_name, "w") as f:
            f.write(test_code_str)

    def run(self, graph_builder: GraphBuilder):
        """
        Process the randomizer generator.
        Usually the only method from this class that is called from the test.
        
        This method generates randomizer model config, initializes nodes, and performs verification via PyBuda.
        
        Args:
            test_context (RandomizerTestContext): The context for the randomizer test.
            graph_builder (GraphBuilder): The graph builder for generating tests.
        """
        logger.debug("-------------- Process Randomizer Generator -------------------")
        randomizer_config = self.test_context.randomizer_config
        parameters = self.test_context.parameters
        logger.debug(f"Parameters test_index: {parameters.test_index} random_seed: {parameters.random_seed} test_device: {parameters.test_device}")

        # build random graph for the specified parameters
        graph = graph_builder.build_graph(parameters)
        self.test_context.graph = graph
        # initialize nodes attributes
        self.init_nodes(graph)
        logger.debug(f"Generating graph model {graph.short_description()}")
        if randomizer_config.print_graph:
            # printing generated graph to console for debugging purposes
            logger.debug(f"Graph config:\n{graph.to_str()}")

        # generate test source code with test function
        test_code_str = self.generate_code()

        if randomizer_config.print_code:
            # printing generated test source code to console for debugging purposes
            logger.debug(f"Generated code: \n{test_code_str}")

        if randomizer_config.save_tests:
            # saving test source code to file for debugging purposes
            self.save_test(test_code_str)

        if randomizer_config.run_test:
            # instantiate PyBuda model
            model = self.build_model()
            # perform model validation
            self.verify(model)
        else:
            logger.info("Skipping test run")


def process_test(test_index: int, random_seed: int, test_device: TestDevice, randomizer_config: RandomizerConfig, graph_builder_type: Type[GraphBuilder], framework: Framework):
    '''
    Process a single randomizer test.

    Args:
        test_index (int): The index of the test.
        random_seed (int): The random seed for the test.
        test_device (TestDevice): The device for the test.
        randomizer_config (RandomizerConfig): The configuration for the randomizer.
        graph_builder_type (Type[GraphBuilder]): The graph builder type (algorithm) for the test.
        framework (Framework): The test framework for the test.
    '''
    # TODO read framwework from randomizer_config

    # instantiate graph_builder
    graph_builder = graph_builder_type(framework, randomizer_config)
    # instantiate parameters
    parameters = RandomizerParameters(test_index, random_seed, test_device, framework_name=framework.framework_name.lower(), graph_builder_name=graph_builder.get_name())
    # instantiate test_context
    test_context = RandomizerTestContext(randomizer_config=randomizer_config, parameters=parameters, graph=None)
    # instantiate graph_builder
    model_builder = framework.ModelBuilderType()
    # instantiate runner
    runner = RandomizerRunner(test_context, model_builder)
    # process test
    runner.run(graph_builder)
