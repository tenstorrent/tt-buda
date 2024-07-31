# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# PyBuda repostiory operators


from .datatypes import OperatorDefinition, OperatorRepository
from .datatypes import OperatorParamNumber


# TODO describe operand and shapes
_OPERATORS = [

    # Unary operators
    OperatorDefinition("exp", "pybuda.op.Exp", 1),
    OperatorDefinition("reciprocal", "pybuda.op.Reciprocal", 1),
    OperatorDefinition("buffer", "pybuda.op.Buffer", 1),
    OperatorDefinition("sqrt", "pybuda.op.Sqrt", 1),
    OperatorDefinition("relu", "pybuda.op.Relu", 1),
    OperatorDefinition("leaky_relu", "pybuda.op.LeakyRelu", 1, forward_params=[
        OperatorParamNumber("alpha", float, 0, 100),
    ]),
    OperatorDefinition("nop", "pybuda.op.Identity", 1),
    OperatorDefinition("gelu", "pybuda.op.Gelu", 1),
    OperatorDefinition("log", "pybuda.op.Log", 1),
    OperatorDefinition("sigmoid", "pybuda.op.Sigmoid", 1),
    OperatorDefinition("clip", "pybuda.op.Clip", 1, forward_params=[
        OperatorParamNumber("min", float, 0, 100),
        OperatorParamNumber("max", float, 0, 100),
    ]),
    OperatorDefinition("sine", "pybuda.op.Sine", 1),
    OperatorDefinition("cosine", "pybuda.op.Cosine", 1),
    OperatorDefinition("abs", "pybuda.op.Abs", 1),
    OperatorDefinition("tanh", "pybuda.op.Tanh", 1),
    OperatorDefinition("cumsum", "pybuda.op.CumSum", 1),
    OperatorDefinition("argmax", "pybuda.op.Argmax", 1),
    OperatorDefinition("logical_not", "pybuda.op.LogicalNot", 1),
    OperatorDefinition("dropout", "pybuda.op.Dropout", 1),
    OperatorDefinition("pow", "pybuda.op.Pow", 1, forward_params=[
        OperatorParamNumber("exponent", float, 0, 100),
    ]),
    OperatorDefinition("tilizer", "pybuda.op.Tilize", 1),

    # Binary operators
    OperatorDefinition("add", "pybuda.op.Add", 2),
    OperatorDefinition("divide", "pybuda.op.Divide", 2),
    OperatorDefinition("subtract", "pybuda.op.Subtract", 2),
    OperatorDefinition("multiply", "pybuda.op.Multiply", 2),
    OperatorDefinition("maximum", "pybuda.op.Max", 2),
    OperatorDefinition("minimum", "pybuda.op.Min", 2),
    OperatorDefinition("heaviside", "pybuda.op.Heaviside", 2),
    OperatorDefinition("binary_stack", "pybuda.op.BinaryStack", 2),
    OperatorDefinition("power", "pybuda.op.Power", 2),
    OperatorDefinition("greater", "pybuda.op.Greater", 2),
    OperatorDefinition("greater_equal", "pybuda.op.GreaterEqual", 2),
    OperatorDefinition("less", "pybuda.op.Less", 2),
    OperatorDefinition("less_equal", "pybuda.op.LessEqual", 2),
    OperatorDefinition("equal", "pybuda.op.Equal", 2),
    OperatorDefinition("not_equal", "pybuda.op.NotEqual", 2),
    OperatorDefinition("logical_and", "pybuda.op.LogicalAnd", 2),

    # Nary operators
    OperatorDefinition("where", "pybuda.op.Where", 3),
    # OperatorDefinition("index_copy", "pybuda.op.IndexCopy", 3),  # Bug #2705
    OperatorDefinition("interleave", "pybuda.op.Interleave", (1,10), forward_params=[
        OperatorParamNumber("axis", int, -3, -3),
        OperatorParamNumber("stride", int, 1, 1),
    ]),
    OperatorDefinition("concatenate", "pybuda.op.Concatenate", (1, 10), forward_params=[
        OperatorParamNumber("axis", int, -10, 10),
    ]),
    OperatorDefinition("stack", "pybuda.op.Stack", (2,4), forward_params=[
        OperatorParamNumber("axis", int, 1, 10),
    ]),

    OperatorDefinition("matmul", "pybuda.op.Matmul", 2),
    # OperatorDefinition("sparse_matmul", "pybuda.op.SparseMatmul", 2),
]


pybuda_operator_repository = OperatorRepository([op for op in _OPERATORS])
