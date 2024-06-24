# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Test random graph configurations by utilizing Random Graph Generator Algorithm and targeting PyBuda and PyTorch frameworks

from enum import Enum
import pytest

from typing import Tuple
from copy import copy

from pybuda.op_repo import OperatorParamNumber
from pybuda.op_repo import OperatorDefinition

from test.random.rgg import Framework
from test.random.rgg import Frameworks
from test.random.rgg import RandomGraphAlgorithm
from test.random.rgg import RandomizerConfig
from test.random.rgg import process_test


class FrameworkTestUtils:

    @classmethod
    def copy_framework(cls, framework: Framework, skip_operators: Tuple[str] = []) -> Framework:
        framework0 = framework
        framework = copy(framework)
        framework.operator_repository = copy(framework.operator_repository)
        cls.skip_operators(framework, skip_operators)
        assert len(framework.operator_repository.operators) + len(skip_operators) == len(framework0.operator_repository.operators), "Operators count should match after skipping operators"
        return framework

    @classmethod
    def skip_operators(cls, framework: Framework, skip_operators: Tuple[str] = []) -> None:
        initial_operator_count = len(framework.operator_repository.operators)
        framework.operator_repository.operators = [op for op in framework.operator_repository.operators if op.name not in skip_operators]
        assert len(framework.operator_repository.operators) + len(skip_operators) == initial_operator_count, "Operators count should match after skipping operators"

    @classmethod
    def allow_operators(cls, framework: Framework, allow_operators: Tuple[str] = []) -> None:
        framework.operator_repository.operators = [op for op in framework.operator_repository.operators if op.name in allow_operators]
        assert len(allow_operators) == len(framework.operator_repository.operators), "Operators count should match allowing skipping operators"

    @classmethod
    def copy_operator(cls, framework: Framework, operator_name: str) -> OperatorDefinition:
        operators = framework.operator_repository.operators

        i, operator = next(((i, operator) for i, operator in enumerate(operators) if operator.name == operator_name), (None, None))
        if not operator:
            return None

        operator = copy(operator)
        operators[i] = operator
        return operator


class FrameworksHealthy(Enum):
    ''' Adjust repositories to test healthy operators '''

    @staticmethod
    def healty_pybuda():
        SKIP_OPERATORS = (
            # Unary operators
            "exp",  # pcc?
            "sqrt",  # skip because it's failing for negative values
            "cumsum",  # bug
            "argmax",  # shape calc is wrong
            "logical_not",  # bug
            "dropout",  # pcc?
            "tilizer",  # bug

            # Binary operators
            "divide",  # bug
            "binary_stack",  # bug
            "power",  # occasionally fails
            "logical_and",  # bug
        )

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYBUDA.value, SKIP_OPERATORS)

        pow_operator = FrameworkTestUtils.copy_operator(framework, "pow")
        if pow_operator:
            pow_operator.forward_params = [
                # float exponent is currently not supported due to issue #2592
                # OperatorParamNumber("exponent", float, 0, 100),
                # OperatorParamNumber("exponent", int, 0, 100),
                OperatorParamNumber("exponent", int, 0, 4),  # pcc for higher numbers fails
            ]

        return framework

    @staticmethod
    def pybuda_matmul_joins():
        SKIP_OPERATORS = (
        )

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYBUDA.value, SKIP_OPERATORS)

        ALLOW_OPERATORS = (
            "relu",
            "tanh",
            "add",
            "matmul",
        )

        FrameworkTestUtils.allow_operators(framework, ALLOW_OPERATORS)

        return framework

    @staticmethod
    def healty_pytorch():
        SKIP_OPERATORS = (
            "sqrt",  # skip because it's failing for negative values
            # "linear",
            "conv2d",  # skip until calc_input_shapes is properly implemented
        )

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYTORCH.value, SKIP_OPERATORS)

        return framework
    
    PYBUDA = healty_pybuda()
    PYBUDA_MATMUL_JOINS = pybuda_matmul_joins()
    PYTORCH = healty_pytorch()


@pytest.mark.parametrize("framework", [
    FrameworksHealthy.PYBUDA.value,
])
def test_random_graph_algorithm_pybuda(test_index, random_seeds, test_device, randomizer_config: RandomizerConfig, framework):
    # adjust randomizer_config
    randomizer_config = copy(randomizer_config)
    # randomizer_config.debug_shapes = True
    # randomizer_config.verify_shapes = True
    randomizer_config.dim_min = 3
    randomizer_config.dim_max = 4
    randomizer_config.op_size_per_dim_min = 4
    # randomizer_config.op_size_per_dim_min = 16
    randomizer_config.op_size_per_dim_max = 8
    # randomizer_config.op_size_per_dim_max = 64
    # randomizer_config.op_size_per_dim_max = 256
    randomizer_config.microbatch_size_min = 1
    randomizer_config.microbatch_size_max = 8
    randomizer_config.num_of_nodes_min = 5
    randomizer_config.num_of_nodes_max = 10
    randomizer_config.num_fork_joins_max = 5

    # TODO random_seed instead of random_seeds
    random_seed = random_seeds[test_index]
    process_test("Default", test_index, random_seed, test_device, randomizer_config, graph_builder_type=RandomGraphAlgorithm, framework=framework)


@pytest.mark.parametrize("framework", [
    FrameworksHealthy.PYBUDA_MATMUL_JOINS.value,
])
def test_random_graph_algorithm_pybuda_matmul_joins(test_index, random_seeds, test_device, randomizer_config: RandomizerConfig, framework):
    # adjust randomizer_config
    randomizer_config = copy(randomizer_config)
    # randomizer_config.debug_shapes = True
    # randomizer_config.verify_shapes = True
    randomizer_config.dim_min = 3
    randomizer_config.dim_max = 4
    randomizer_config.op_size_per_dim_min = 4
    # randomizer_config.op_size_per_dim_min = 16
    randomizer_config.op_size_per_dim_max = 8
    # randomizer_config.op_size_per_dim_max = 64
    # randomizer_config.op_size_per_dim_max = 256
    randomizer_config.microbatch_size_min = 1
    randomizer_config.microbatch_size_max = 8
    randomizer_config.num_of_nodes_min = 10
    randomizer_config.num_of_nodes_max = 15
    randomizer_config.num_fork_joins_max = 10

    # TODO random_seed instead of random_seeds
    random_seed = random_seeds[test_index]
    process_test("Matmul Joins", test_index, random_seed, test_device, randomizer_config, graph_builder_type=RandomGraphAlgorithm, framework=framework)


@pytest.mark.parametrize("framework", [
    FrameworksHealthy.PYTORCH.value,
])
def test_random_graph_algorithm_pytorch(test_index, random_seeds, test_device, randomizer_config: RandomizerConfig, framework):
    # adjust randomizer_config
    randomizer_config = copy(randomizer_config)
    # randomizer_config.debug_shapes = True
    # randomizer_config.verify_shapes = True
    randomizer_config.dim_min = 4
    randomizer_config.dim_max = 4
    randomizer_config.op_size_per_dim_min = 4
    # randomizer_config.op_size_per_dim_min = 16
    randomizer_config.op_size_per_dim_max = 8
    # randomizer_config.op_size_per_dim_max = 64
    # randomizer_config.op_size_per_dim_max = 256
    randomizer_config.microbatch_size_min = 1
    randomizer_config.microbatch_size_max = 8
    randomizer_config.num_of_nodes_min = 3
    randomizer_config.num_of_nodes_max = 5
    randomizer_config.num_fork_joins_max = 5

    # TODO random_seed instead of random_seeds
    random_seed = random_seeds[test_index]
    process_test("Default", test_index, random_seed, test_device, randomizer_config, graph_builder_type=RandomGraphAlgorithm, framework=framework)
