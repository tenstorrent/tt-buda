# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Base classes for random randomizer generator


from typing import Type
from loguru import logger
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader
import os
import random

from pybuda import PyBudaModule
from pybuda.verify import verify_module, VerifyConfig
from pybuda.op_repo import OperatorRepository
from test.operators.utils import ShapeUtils
from test.conftest import TestDevice
from test.utils import Timer
from .datatypes import RandomizerNode, RandomizerGraph, RandomizerParameters, RandomizerConfig, ExecutionContext
from .datatypes import RandomizerTestContext
from .utils import StrUtils, GraphUtils
from .utils import timeout, TimeoutException


class GraphBuilder:
    """
    GraphBuilder is an interface that each graph building algorithm should implement.
    GraphBuilder encapsulates the logic for generating random graphs.
    """
    def __init__(self, randomizer_config: RandomizerConfig):
        self.randomizer_config = randomizer_config

    def build_graph(self, test_context: RandomizerTestContext) -> None:
        """
        Generate test graph with input shape needed for validation.

        Args:
            test_context (RandomizerTestContext): The context for the randomizer test.

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
        return StrUtils.kwargs_str(**node.constructor_kwargs)

    def forward_args(self, node: RandomizerNode) -> str:
        args_str = ", ".join([f"inputs[{i}]" for i in range(node.input_num)])
        return args_str
    
    def forward_kwargs(self, node: RandomizerNode) -> str:
        return StrUtils.kwargs_str(**node.forward_kwargs)

    def generate_code(self, test_context: RandomizerTestContext, test_format: bool = True) -> str:
        # TODO setup random seed in generated test function

        parameters = test_context.parameters
        template = self.template

        code_str = template.render(
            randomizer_config = test_context.randomizer_config,
            graph_builder_name = parameters.graph_builder_name,
            test_id = StrUtils.test_id(test_context),
            test_format = test_format,
            test_index = parameters.test_index,
            random_seed = parameters.random_seed,
            graph=test_context.graph,
            constructor_kwargs=self.constructor_kwargs,
            forward_args=self.forward_args,
            forward_kwargs=self.forward_kwargs,
            reduce_microbatch_size=ShapeUtils.reduce_microbatch_size,
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


# TODO move RandomizerRunner and process_test to runner.py
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

    def generate_code(self) -> str:
        """
        Generates a test source code with test function for the randomized graph.

        Returns:
            str: The generated code.
        """
        return self.code_generator.generate_code(self.test_context, test_format=True)

    def build_graph(self, graph_builder: GraphBuilder) -> None:
        self.test_context.graph = RandomizerGraph()

        # Initialize random number generators for graph building
        self.test_context.rng_graph = random.Random(self.test_context.parameters.random_seed)
        # Initialize random number generators for shape generation
        self.test_context.rng_shape = random.Random(self.test_context.parameters.random_seed)
        # Initialize random number generators for parameters
        self.test_context.rng_params = random.Random(self.test_context.parameters.random_seed)

        graph_builder.build_graph(self.test_context)

    def build_model(self) -> PyBudaModule:
        model = self.model_provider.build_model(self.test_context)
        return model

    def verify(self, model: PyBudaModule) -> None:

        verification_timeout = self.test_context.randomizer_config.verification_timeout

        try:
            @timeout(verification_timeout)
            def verify_model_timeout() -> None:
                self.verify_model(model)

            verify_model_timeout()
        except TimeoutException as e:
            logger.error(f"Module verification takes too long {e}.")
            raise e

    def verify_model(self, model: PyBudaModule) -> None:
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
        input_shapes = GraphUtils.get_input_shapes(self.test_context.graph)

        # verify PyBuda model
        verify_module(model, input_shapes,
                      VerifyConfig(devtype=parameters.test_device.devtype, arch=parameters.test_device.arch))

    def save_test(self, test_code_str: str, failing_test: bool = False):
        test_dir = self.test_context.randomizer_config.test_dir
        if failing_test:
            test_dir = f"{test_dir}/failing_tests"
            test_code_str = test_code_str.replace("# @pytest.mark.xfail", "@pytest.mark.xfail") 
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
        logger.trace("Building graph started")
        graph_duration = Timer()
        try:
            self.build_graph(graph_builder)
        except Exception as e1:
            # Try to save test source code to file for debugging purposes if an error occurs
            try:
                test_code_str = self.generate_code()
                if randomizer_config.save_tests:
                    # Saving test source code to file for debugging purposes
                    self.save_test(test_code_str, failing_test=True)
            except Exception as e2:
                logger.error(f"Error while saving test: {e2}")
            # Re-raise the original exception from graph building
            raise e1
        logger.trace("Building graph completed")
        graph = self.test_context.graph
        logger.debug(f"Generating graph model {GraphUtils.short_description(graph)}")
        if randomizer_config.print_graph:
            # printing generated graph to console for debugging purposes
            logger.debug(f"Graph config:\n{StrUtils.to_str(graph)}")

        # generate test source code with test function
        test_code_str = self.generate_code()

        if randomizer_config.print_code:
            # printing generated test source code to console for debugging purposes
            logger.debug(f"Generated code: \n{test_code_str}")

        if randomizer_config.save_tests:
            # saving test source code to file for debugging purposes
            self.save_test(test_code_str, failing_test=False)

        logger.info(f"Graph built in: {graph_duration.get_duration():.4f} seconds")

        if randomizer_config.run_test:
            # instantiate PyBuda model
            model = self.build_model()
            # perform model validation
            try:
                verify_duration = Timer()
                verify_successful = False
                self.verify(model)
                verify_successful = True
            finally:
                if not verify_successful:
                    if randomizer_config.save_failing_tests:
                        # saving error test source code to file for debugging purposes
                        self.save_test(test_code_str, failing_test=True)
                logger.debug(f"Test verified in: {verify_duration.get_duration():.4f} seconds")
        else:
            logger.info("Skipping test run")


def process_test(test_name: str, test_index: int, random_seed: int, test_device: TestDevice, randomizer_config: RandomizerConfig, graph_builder_type: Type[GraphBuilder], framework: Framework):
    '''
    Process a single randomizer test.

    Args:
        test_name (str): The name of the test used for generating test code, test file name, etc.
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
    test_context = RandomizerTestContext(randomizer_config=randomizer_config, parameters=parameters, graph=None, test_name=test_name)
    # instantiate graph_builder
    model_builder = framework.ModelBuilderType()
    # instantiate runner
    runner = RandomizerRunner(test_context, model_builder)
    # process test
    runner.run(graph_builder)
