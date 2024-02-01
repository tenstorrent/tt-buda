# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union

from ..tensor import Tensor
from ..parameter import Parameter
from .common import PyBudaOp as op

def Add(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise add of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "add")

def Subtract(
        name: str, 
        operandA: Tensor, 
        operandB: Tensor) -> Tensor:

    """
    Elementwise subtraction of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "subtract")

def Multiply(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter]) -> Tensor:

    """
    Elementwise multiply of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "multiply")

def Divide(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter]) -> Tensor:

    """
    Elementwise divide of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "divide")

def Max(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise max of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "maximum")

def Min(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise min of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "minimum")

def Heaviside(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise max of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "heaviside")

def BinaryStack(
        name: str,
        operandA: Tensor,
        operandB: Union[Tensor, Parameter],
        dim: int) -> Tensor:
    """
    Elementwise max of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    dim: int
        Dimention on which to stack

    Returns
    -------
    Tensor
        Buda tensor

    """

    return op("binary_stack", name, operandA, operandB, attrs=(dim, )).get_tensor()

def Power(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    OperandA to the power of OperandB

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "power")


def Equal(
    name: str,
    operandA: Tensor,
    operandB: Union[Tensor, Parameter]
    ) -> Tensor:
    """
    Elementwise equal of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "equal")


def NotEqual(
    name: str,
    operandA: Tensor,
    operandB: Union[Tensor, Parameter]
    ) -> Tensor:
    """
    Elementwise equal of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "not_equal")


def Greater(
    name: str,
    operandA: Tensor,
    operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise greater of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "greater")


def Less(
    name: str,
    operandA: Tensor,
    operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise less of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "less")


def GreaterEqual(
    name: str, 
    operandA: Tensor, 
    operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise greater or equal of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "greater_equal")


def LessEqual(
    name: str, 
    operandA: Tensor, 
    operandB: Union[Tensor, Parameter]) -> Tensor:
    """
    Elementwise less or equal of two tensors

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor

    """

    return _Eltwise(name, operandA, operandB, "less_equal")


def _Eltwise(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter],
        op_type: str) -> Tensor:

    """
    Common implementation for eltwise ops.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    op_type: str
        Operation name (add, subtract, multiply...)

    Returns
    -------
    Tensor
        Buda tensor
    """

    result: Tensor = op(op_type, name, operandA, operandB).get_tensor()
    return result


def LogicalAnd(
        name: str,
        operandA: Tensor,
        operandB: Union[Tensor, Parameter]) -> Tensor:

    """
    Logical and operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operandB: Tensor
        Second operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("logical_and", name, operandA, operandA).get_tensor()
