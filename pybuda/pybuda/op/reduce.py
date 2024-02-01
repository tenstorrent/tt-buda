# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..tensor import Tensor
from .common import PyBudaOp as op

def ReduceSum(
        name: str,
        operandA: Tensor,
        dim: int) -> Tensor:
    """
    Reduce by summing along the given dimension

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

    Returns
    -------
    Tensor
        Buda tensor
    """

    assert (dim >= -4) and (dim <= 3)
    # if dim < 0:
    #     dim += 4

    return op("reduce_sum", name, operandA, attrs=(dim,)).get_tensor()

def ReduceAvg(
        name: str,
        operandA: Tensor,
        dim: int) -> Tensor:
    """
    Reduce by averaging along the given dimension

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

    Returns
    -------
    Tensor
        Buda tensor
    """

    assert (dim >= -4) and (dim <= 3)
    # if dim < 0:
    #     dim += 4

    return op("reduce_avg", name, operandA, attrs=(dim,)).get_tensor()

def GroupedReduceAvg(
        name: str,
        operandA: Tensor,
        dim: int,
        groups: int,
        keep_dims: bool = False) -> Tensor:
    """
    Reduce by averaging along the given dimension

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.
    
    groups: int
        Number of groups to reduce along dim. Must be a factor of the dimension size.
        i.e: GroupReduce(-2, 32) on a tensor of shape (1, 1, 320, 1024) will reduce to (1, 1, 32, 1024)
             such that nth row on the new tensor is the average of the nth section of 10 rows (320 // 32) on the old tensor.
    
    keep_dims: bool
        Whether or not to keep the dimension size averaged groups such that each element in a group is replaced with the average of the group.
        i.e GroupReduce(-2, 2, keep_dims=True) on a tensor of shape (4, 1) will reduce to (4, 1)
            Say we have [[1], [2], [3], [4]] as the original tensor. The new tensor will be [[1.5], [1.5], [3.5], [3.5]]
            If keep_dims=False, the new tensor will be [[1.5], [3.5]]
    Returns
    -------
    Tensor
        Buda tensor
    """

    assert (dim >= -4) and (dim <= 3)
    return op("grouped_reduce_avg", name, operandA, attrs=(dim, groups, keep_dims)).get_tensor()

def ReduceMax(
        name: str,
        operandA: Tensor,
        dim: int,
        stride: int = -1) -> Tensor:
    """
    Reduce by averaging along the given dimension

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        Dimension along which to reduce. A positive number 0 - 3 or negative from -1 to -4.

    Returns
    -------
    Tensor
        Buda tensor
    """
    if stride == -1:
        stride = int(operandA.shape[dim])
    assert (dim >= -4) and (dim <= 3)
    # if dim < 0:
    #     dim += 4

    return op("reduce_max", name, operandA, attrs=(dim,stride)).get_tensor()
