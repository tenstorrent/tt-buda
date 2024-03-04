# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union, Tuple
from ..tensor import Tensor
from .common import PyBudaOp as op

def Abs(
        name: str,
        operandA: Tensor) -> Tensor:
    """
    Sigmoid

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("abs", name, operandA).get_tensor()

def Exp(
        name: str, 
        operandA: Tensor) -> Tensor:

    """
    Exponent operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("exp", name, operandA).get_tensor()

def Log(
        name: str,
        operandA: Tensor) -> Tensor:

    """
    Log operation: natural logarithm of the elements of `operandA`
        yi = log_e(xi) for all xi in operandA tensor

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("log", name, operandA).get_tensor()


def Pow(
        name: str,
        operandA: Tensor,
        exponent: Union[int, float]) -> Tensor:

    """
    Pow operation: `operandA` to the power of `exponent`
        yi = pow(xi, exponent) for all xi in operandA tensor

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("pow", name, operandA, attrs=(exponent,)).get_tensor()


def Identity(
        name: str, 
        operandA: Tensor,
        unsqueeze : str =  None,
        unsqueeze_dim : int = None) -> Tensor:

    """
    Identity operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    unsqueeze: str
        If set, the operation returns a new tensor with a dimension of size one inserted at the specified position.

    unsqueeze_dim: int
        The index at where singleton dimenion can be inserted

    Returns
    -------
    Tensor
        Buda tensor
    """

    if unsqueeze==None and unsqueeze_dim==None:
        return op("nop", name, operandA).get_tensor()
    else:
        return op("nop", name, operandA, unsqueeze=unsqueeze, unsqueeze_dim=unsqueeze_dim).get_tensor()


def Buffer(
        name: str, 
        operandA: Tensor) -> Tensor:

    """
    Identity operation. One key difference is a Buffer op will not get
    lowered into a NOP and avoid being removed by the time it gets to lowering.


    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("buffer", name, operandA).get_tensor()


def Reciprocal(
        name: str, 
        operandA: Tensor) -> Tensor:

    """
    Reciprocal operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("reciprocal", name, operandA).get_tensor()

def Sqrt(
        name: str, 
        operandA: Tensor) -> Tensor:

    """
    Square root.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("sqrt", name, operandA).get_tensor()

def Relu(
        name: str, 
        operandA: Tensor,
        threshold=0.0,
        mode="min") -> Tensor:

    """
    ReLU

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """
    if threshold == 0.0 and mode == "min":
        return op("relu", name, operandA).get_tensor() # avoid threshold < 0.0 error due to FP arithmetics
    else:
        return op("relu", name, operandA, attrs=(threshold, mode)).get_tensor()

def LeakyRelu(
        name: str, 
        operandA: Tensor,
        alpha: int) -> Tensor:

    """
    Leaky ReLU

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    alpha: int
        Controls the angle of the negative slope

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("leaky_relu", name, operandA, attrs=(alpha,)).get_tensor()

def Gelu(
        name: str, 
        operandA: Tensor,
        approximate="none") -> Tensor:

    """
    GeLU

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    approximate: str
        The gelu approximation algorithm to use: 'none' | 'tanh'.
        Default: 'none'

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("gelu", name, operandA, attrs=(approximate,)).get_tensor()

def Sigmoid(
        name: str,
        operandA: Tensor) -> Tensor:
    """
    Sigmoid

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("sigmoid", name, operandA).get_tensor()

def Argmax(
        name: str,
        operandA: Tensor,
        dim: int = None) -> Tensor:
    """
    Argmax

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("argmax", name, operandA, attrs=((dim,) if dim is not None else ())).get_tensor()

def Clip(
        name: str,
        operandA: Tensor,
        min: float,
        max: float) -> Tensor:
    """
    Clips tensor values between min and max

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    min: float
        Minimum value

    max: float
        Maximum value

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("clip", name, operandA, min=min, max=max).get_tensor()

def Sine(
        name: str,
        operandA: Tensor) -> Tensor:
    """
    Elementwise sine

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    min: float
        Minimum value

    max: float
        Maximum value

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("sine", name, operandA).get_tensor()

def Cosine(
        name: str,
        operandA: Tensor) -> Tensor:
    """
    Elementwise cosine

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    min: float
        Minimum value

    max: float
        Maximum value

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("cosine", name, operandA).get_tensor()

def Tanh(
        name: str,
        operandA: Tensor) -> Tensor:

    """
    Tanh operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("tanh", name, operandA).get_tensor()


def CumSum(
        name: str,
        operandA: Tensor,
        axis: int,
        exclusive: bool = False) -> Tensor:

    """
    Cumulative sum operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand
    
    exclusive: bool
        Perform exclusive cumulative sum which includes (or not) the 
        first operand. For example:
        x: [2, 4, 6, 8]
        
        cumsum(x, exclusive=False) 
        [2, 6, 12, 20]
        
        cumsum(x, exclusive=True) 
        [0,  2,  6, 12]

    Returns
    -------
    Tensor
        Buda tensor
    """
    
    assert not exclusive, "Currently not supported"        

    return op("cumsum", name, operandA, axis=axis, exclusive=exclusive).get_tensor()


def LogicalNot(
        name: str,
        operandA: Tensor) -> Tensor:

    """
    Logical not operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("logical_not", name, operandA).get_tensor()


def Dropout(
        name: str,
        operandA: Tensor,
        p: float = 0.5,
        training: bool = True,
        seed: int = 0) -> Tensor:
    """
    Dropout

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    p: float
        Probability of an element to be zeroed.

    training: bool
        Apply dropout if true

    seed: int
        RNG seed

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("dropout", name, operandA, attrs=(p, training, seed)).get_tensor()

def Tilize(
        name: str, 
        operandA: Tensor) -> Tensor:

    """
    Tilize operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("tilizer", name, operandA).get_tensor()