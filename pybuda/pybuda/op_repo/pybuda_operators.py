# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# PyBuda repostiory operators


from .datatypes import OperatorDefinition, OperatorRepository
from .datatypes import OperatorParamNumber
from .shapes import same_input_shapes
from .shapes import matmul_inputs


# TODO describe operand and shapes
_OPERATORS = [

    # Unary operators
    OperatorDefinition("exp", "pybuda.op.Exp", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("reciprocal", "pybuda.op.Reciprocal", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("buffer", "pybuda.op.Buffer", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("sqrt", "pybuda.op.Sqrt", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("relu", "pybuda.op.Relu", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("leaky_relu", "pybuda.op.LeakyRelu", 1, forward_params=[
        OperatorParamNumber("alpha", float, 0, 100),
    ], calc_input_shapes=same_input_shapes),
    OperatorDefinition("nop", "pybuda.op.Identity", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("gelu", "pybuda.op.Gelu", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("log", "pybuda.op.Log", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("sigmoid", "pybuda.op.Sigmoid", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("clip", "pybuda.op.Clip", 1, forward_params=[
        OperatorParamNumber("min", float, 0, 100),
        OperatorParamNumber("max", float, 0, 100),
    ], calc_input_shapes=same_input_shapes),
    OperatorDefinition("sine", "pybuda.op.Sine", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("cosine", "pybuda.op.Cosine", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("abs", "pybuda.op.Abs", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("tanh", "pybuda.op.Tanh", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("cumsum", "pybuda.op.CumSum", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("argmax", "pybuda.op.Argmax", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("logical_not", "pybuda.op.LogicalNot", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("dropout", "pybuda.op.Dropout", 1, calc_input_shapes=same_input_shapes),
    OperatorDefinition("pow", "pybuda.op.Pow", 1, forward_params=[
        OperatorParamNumber("exponent", float, 0, 100),
    ], calc_input_shapes=same_input_shapes),
    OperatorDefinition("tilizer", "pybuda.op.Tilize", 1, calc_input_shapes=same_input_shapes),

    # Binary operators
    OperatorDefinition("add", "pybuda.op.Add", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("divide", "pybuda.op.Divide", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("subtract", "pybuda.op.Subtract", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("multiply", "pybuda.op.Multiply", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("maximum", "pybuda.op.Max", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("minimum", "pybuda.op.Min", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("heaviside", "pybuda.op.Heaviside", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("binary_stack", "pybuda.op.BinaryStack", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("power", "pybuda.op.Power", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("greater", "pybuda.op.Greater", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("greater_equal", "pybuda.op.GreaterEqual", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("less", "pybuda.op.Less", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("less_equal", "pybuda.op.LessEqual", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("equal", "pybuda.op.Equal", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("not_equal", "pybuda.op.NotEqual", 2, calc_input_shapes=same_input_shapes),
    OperatorDefinition("logical_and", "pybuda.op.LogicalAnd", 2, calc_input_shapes=matmul_inputs),

    OperatorDefinition("matmul", "pybuda.op.Matmul", 2, calc_input_shapes=matmul_inputs),
]


pybuda_operator_repository = OperatorRepository([op for op in _OPERATORS])
