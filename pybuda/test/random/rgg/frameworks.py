# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# In depth testing of PyBuda models with one randomly selected operation


from enum import Enum

from typing import Tuple
from copy import copy

from .base import Framework

from .pybuda.model import PyBudaModelBuilder
from .pytorch.model import PyTorchModelBuilder

from pybuda.op_repo import pybuda_operator_repository
from pybuda.op_repo import pytorch_operator_repository
from pybuda.op_repo import OperatorDefinition


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


class Frameworks(Enum):
    ''' Register of all frameworks '''

    PYBUDA = Framework(
        framework_name="PyBuda",
        ModelBuilderType=PyBudaModelBuilder,
        operator_repository=pybuda_operator_repository,
    )
    PYTORCH = Framework(
        framework_name="PyTorch",
        ModelBuilderType=PyTorchModelBuilder,
        operator_repository=pytorch_operator_repository,
    )
