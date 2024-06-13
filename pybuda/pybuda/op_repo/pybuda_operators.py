# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# PyBuda repostiory operators


from .datatypes import OperatorDefinition, OperatorRepository
from .datatypes import OperatorParamNumber
from .shapes import same_input_shapes
from .shapes import matmul_inputs


# TODO describe operand and shapes
_OPERATORS = [
    OperatorDefinition("relu", "pybuda.op.Relu", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("tanh", "pybuda.op.Tanh", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("exp", "pybuda.op.Exp", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("pow", "pybuda.op.Pow", 1, forward_params=[
        # float exponent is currently not supported due to issue #2592
        # OperatorParamNumber("exponent", float, 0, 100),
        OperatorParamNumber("exponent", int, 0, 100),
    ], calc_input_shapes=same_input_shapes),
    OperatorDefinition("add", "pybuda.op.Add", 2, calc_input_shapes=same_input_shapes),

    OperatorDefinition("matmul", "pybuda.op.Matmul", 2, calc_input_shapes=matmul_inputs),
    OperatorDefinition("eltwise", "pybuda.op.Add", 2, calc_input_shapes=same_input_shapes),
]


pybuda_operator_repository = OperatorRepository([op for op in _OPERATORS])
