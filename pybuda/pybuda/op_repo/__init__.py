# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Operator repository datatypes
#
# Central place for defining all PyBuda, PyTorch, ... operators
#
# Usage of repository:
#  - RGG (Random Graph Generator)
#  - Single operator tests
#  - TVM python_codegen.py


from .datatypes import TensorShape, OperatorParam, OperatorParamNumber, OperatorDefinition, OperatorRepository
from .datatypes import ShapeCalculationContext
from .pybuda_operators import pybuda_operator_repository
from .pytorch_operators import pytorch_operator_repository

__ALL__ = [
    "TensorShape",
    "OperatorParam",
    "OperatorParamNumber",
    "OperatorDefinition",
    "OperatorRepository",
    "ShapeCalculationContext",
    "pybuda_operator_repository",
    "pytorch_operator_repository",
]
