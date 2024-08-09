# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# In depth testing of PyBuda models with one randomly selected operation


from enum import Enum

from loguru import logger
from typing import Tuple, Type
from copy import copy

from .base import Framework, ModelBuilder
from .shapes import OperatorShapes

from .pybuda.model import PyBudaModelBuilder
from .pytorch.model import PyTorchModelBuilder

from pybuda.op_repo import pybuda_operator_repository
from pybuda.op_repo import pytorch_operator_repository
from pybuda.op_repo import OperatorDefinition
from pybuda.op_repo import OperatorRepository


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

    @classmethod
    def set_calc_input_shapes(cls, framework: Framework, allow_operators: Tuple[str] = []) -> None:
        ''' Implicitly set calc_input_shapes for all operators in the framework '''
        logger.debug(f"Setting calc_input_shapes for framework {framework.framework_name}")
        for operator in framework.operator_repository.operators:
            function_name = f"{operator.name}_inputs"
            if function_name in OperatorShapes.__dict__:
                logger.debug(f"Found method {function_name} for {operator.name}")
                operator.calc_input_shapes = OperatorShapes.__dict__[function_name]
            else:
                operator.calc_input_shapes = OperatorShapes.same_input_shapes


def build_framework(framework_name: str, ModelBuilderType: Type[ModelBuilder], operator_repository: OperatorRepository):
    framework = Framework(
        framework_name=framework_name,
        ModelBuilderType=ModelBuilderType,
        operator_repository=operator_repository,
    )

    framework = FrameworkTestUtils.copy_framework(framework=framework, skip_operators=())

    FrameworkTestUtils.set_calc_input_shapes(framework)

    return framework


class Frameworks(Enum):
    ''' Register of all frameworks '''

    PYBUDA = build_framework(
        framework_name="PyBuda",
        ModelBuilderType=PyBudaModelBuilder,
        operator_repository=pybuda_operator_repository,
    )
    PYTORCH = build_framework(
        framework_name="PyTorch",
        ModelBuilderType=PyTorchModelBuilder,
        operator_repository=pytorch_operator_repository,
    )
