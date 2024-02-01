# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union, List

from ..tensor import Tensor
from ..parameter import Parameter
from .common import PyBudaOp as op

def Concatenate(
        name: str, 
        *operands: Tensor,
        axis: int) -> Tensor:

    """
    Concatenate tensors along axis

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operands: Tuple[Tensor, ...]
        tensors to be concatenated

    axis: int
        concatenate axis


    Returns
    -------
    Tensor
        Buda tensor
    """

    result: Tensor = op("concatenate", name, *operands, attrs=(axis,)).get_tensor()
    return result

def Where(
        name: str,
        condition: Tensor,
        x: Tensor,
        y: Tensor) -> Tensor:

    """
    Returns elements selected from either x or y depending on condition

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    condition: Tensor
        When True (nonzero), yield x, else y

    x: Tensor
        value(s) if true

    y: Tensor
        value(s) if false

    Returns
    -------
    Tensor
        Buda tensor
    """

    result: Tensor = op("where", name, condition, x, y).get_tensor()
    return result


def IndexCopy(
        name: str,
        operandA: Tensor,
        index: Tensor,
        value: Tensor,
        dim: int) -> Tensor:
    """
    Copies the elements of value into operandA at index along dim

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    index: Tensor
        Index at which to write into operandA
    
    value: Tensor
        Value to write out

    dim: int
        Dimension to broadcast

    Returns
    -------
    Tensor
        Buda tensor
    """
    if dim < 0:
        dim += len(operandA.shape)
    return op("index_copy", name, operandA, index, value, attrs=(dim, )).get_tensor()


def Stack(
        name: str, 
        *operands: Tensor,
        axis: int) -> Tensor:

    """
    Stack tensors along new axis

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operands: Tuple[Tensor, ...]
        tensors to be stacked

    axis: int
        new stack axis


    Returns
    -------
    Tensor
        Buda tensor
    """

    result: Tensor = op("stack", name, *operands, attrs=(axis,)).get_tensor()
    return result


def Interleave(
        name: str, 
        *operands: Tensor,
        axis: int,
        stride: int,) -> Tensor:

    """
    Interleave tensors along an axis with stride
        - each operand must have the same stride

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operands: Tuple[Tensor, ...]
        tensors to be interleaved

    axis: int
        interleave axis

    stride: int
        stride to interleave each operand


    Returns
    -------
    Tensor
        Buda tensor
    """
    assert axis == -3, "Only support Z dim interleave"
    assert stride == 1, "Only support interleave with stride 1 for now"
    result: Tensor = op("interleave", name, *operands, attrs=(axis, stride)).get_tensor()
    return result