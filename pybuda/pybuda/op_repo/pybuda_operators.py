# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# PyBuda repostiory operators


from .datatypes import OperatorDefinition, OperatorRepository
from .datatypes import OperatorParamNumber


# TODO describe operand and shapes
_OPERATORS = [
    OperatorDefinition("relu", "pybuda.op.Relu", 1),
    OperatorDefinition("sqrt", "pybuda.op.Sqrt", 1),
    OperatorDefinition("tanh", "pybuda.op.Tanh", 1),
    OperatorDefinition("exp", "pybuda.op.Exp", 1),
    OperatorDefinition("pow", "pybuda.op.Pow", 1, forward_params=[
        # float exponent is currently not supported due to issue #2592
        # OperatorParamNumber("exponent", float, 0, 100),
        OperatorParamNumber("exponent", int, 0, 100),
    ]),
    OperatorDefinition("add", "pybuda.op.Add", 2),

    OperatorDefinition("matmul", "pybuda.op.Matmul", 2),
    OperatorDefinition("eltwise", "pybuda.op.Add", 2),
]


pybuda_operator_repository = OperatorRepository([op for op in _OPERATORS])
