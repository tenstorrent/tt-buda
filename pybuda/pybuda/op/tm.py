# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union, Tuple, List, Dict
from ..pybudaglobal import TILE_DIM
from .common import PyBudaOp as op
from ..tensor import Tensor, pytorch_dtype_to_buda_dataformat

import torch

def HSlice(
        name: str, 
        operandA: Tensor,
        slices: int) -> Tensor:

    """
    Slice along horizontal axis into given number of pieces.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    slices: int
        The number of slices to create

    Returns
    -------
    Tensor
        Buda tensor
    """

    hdim = operandA.shape.c
    assert hdim % slices == 0, f"Tensor must be evenly divisible by the number of slices."
    return op("hslice", name, operandA, attrs=(slices,)).get_tensor()

def HStack(
        name: str, 
        operandA: Tensor,
        slices: int = -1) -> Tensor:

    """
    Stack Z dimension along horizontal dimension.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    slices: int, optional
        The number of slices to create. If not provided, it will be equal to current Z dimension.

    Returns
    -------
    Tensor
        Buda tensor
    """
    if slices == -1:
        slices = operandA.shape.z
    return op("hstack", name, operandA, attrs=(slices,)).get_tensor()

def VSlice(
        name: str, 
        operandA: Tensor,
        slices: int) -> Tensor:

    """
    Slice along vertical axis into given number of pieces.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    slices: int
        The number of slices to create

    Returns
    -------
    Tensor
        Buda tensor
    """

    vdim = operandA.shape.r
    assert vdim % slices == 0, f"Tensor must be evenly divisible by the number of slices."
    return op("vslice", name, operandA, attrs=(slices,)).get_tensor()

def VStack(
        name: str, 
        operandA: Tensor,
        slices: int = -1) -> Tensor:

    """
    Stack Z dimension along vertical dimension.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    slices: int, optional
        The number of slices to create. If not provided, it will be equal to current Z dimension.

    Returns
    -------
    Tensor
        Buda tensor
    """
    if slices == -1:
        slices = operandA.shape.z
    return op("vstack", name, operandA, attrs=(slices,)).get_tensor()

def Transpose(
        name: str, 
        operandA: Tensor,
        dim0: int,
        dim1: int,
        z_dim_slice: int = -1,
        out_dtype: torch.dtype = torch.float32) -> Tensor:

    """
    Tranpose X and Y (i.e. rows and columns) dimensions.

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
    assert dim0 != dim1

    dims = len(operandA.shape.dims)
    if dim0 >= 0:
        dim0 -= dims

    if dim1 >= 0:
        dim1 -= dims

    assert dim0 < 0
    assert dim1 < 0

    if dim0 > dim1:
        dim0, dim1 = dim1, dim0

    return op("transpose", name, operandA, dim0=dim0, dim1=dim1, z_dim_slice=z_dim_slice).get_tensor(out_df=pytorch_dtype_to_buda_dataformat(out_dtype))

def Reshape(
        name: str, 
        operandA: Tensor,
        shape: Tuple[int, ...]) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    Returns
    -------
    Tensor
        Buda tensor
    """
    tensor_volume = 1
    for dim in operandA.shape.dims:
        tensor_volume *= dim

    blank_idx = -1
    volume = 1
    for idx, d in enumerate(shape):
        if d == -1:
            assert blank_idx == -1, "Cannot have multiple -1 dims"
            blank_idx = idx
        else:
            volume *= d

    if blank_idx != -1:
        assert (tensor_volume % volume) == 0, "-1 dim does not divide evenly"
        shape[blank_idx] = tensor_volume // volume
        volume *= shape[blank_idx]

    assert tensor_volume == volume

    return op("reshape", name, operandA, attrs=shape).get_tensor()

def Index(
        name: str,
        operandA: Tensor,
        dim: int,
        start: int,
        stop: int = None,
        stride: int = 1) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to slice

    start: int
        Starting slice index (inclusive)

    stop: int
        Stopping slice index (exclusive)

    stride: int
        Stride amount along that dimension

    Returns
    -------
    Tensor
        Buda tensor
    """
    if dim >= 0:
        dim -= len(operandA.shape)

    if stop is None:
        stop = start + 1

    if stop < 0:
        stop += operandA.shape[dim]

    assert stride > 0

    assert start < operandA.shape[dim]
    assert stop <= operandA.shape[dim]
    assert stride <= operandA.shape[dim]

    return op("index", name, operandA, attrs=(dim, start, stop, stride)).get_tensor()


def AdvIndex(
        name: str,
        operandA: Tensor,
        operandB: Tensor,
        dim: int = 0,
    ) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A - reference tensor

    operandA: Tensor
        Input operand B - indices

    dim: int
        Dimension to fetch indices over

    Returns
    -------
    Tensor
        Buda tensor
    """
    return op("adv_index", name, operandA, operandB, attrs=(dim,)).get_tensor()


def Select(
        name: str,
        operandA: Tensor,
        dim: int,
        index: Union[int, Tuple[int, int]],
        stride: int = 0) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to slice

    index: int
        int: Index to select from that dimension
        [start: int, length: int]: Index range to select from that dimension

    stride: int
        Stride amount along that dimension

    Returns
    -------
    Tensor
        Buda tensor
    """
    dims = len(operandA.shape.dims)
    if dim < 0:
        dim += dims

    assert dim < 4

    if type(index) is int:
        index = (index, 1)

    if stride == 0:
        stride = operandA.shape.get_pytorch_shape()[dim]

    start, length = index
    assert start < operandA.shape.get_pytorch_shape()[dim], f"start = {start} should be < operandA.shape.get_pytorch_shape()[{dim}] = {operandA.shape.get_pytorch_shape()[dim]}"
    assert (start + length) <= operandA.shape.get_pytorch_shape()[dim], f"(start = {start} + length = {length}) should be <= operandA.shape.get_pytorch_shape()[{dim}] = {operandA.shape.get_pytorch_shape()[dim]}"
    assert stride <= operandA.shape.get_pytorch_shape()[dim], f"stride = {stride} should be <= operandA.shape.get_pytorch_shape()[{dim}] = {operandA.shape.get_pytorch_shape()[dim]}"
    assert (start + length) <= stride, f"(start = {start} + length = {length}) should be <= stride = {stride}"
    assert (start + length) > 0, f"(start = {start} + length = {length}) should be > 0"

    return op("select", name, operandA, attrs=(dim, index[0], index[1], stride)).get_tensor()


def Pad(
        name: str,
        operandA: Tensor,
        pad: Union[Tuple[int, int, int, int], Tuple[int, int]],
        mode: str = "constant",
        channel_last: bool = False) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    pad: tuple
        Either (padding_left, padding_right) or (padding_left, padding_right, padding_top, padding_bottom))

    Returns
    -------
    Tensor
        Buda tensor
    """
    assert len(pad) == 2 or len(pad) == 4, "Expect (padding_left, padding_right) or (padding_left, padding_right, padding_top, padding_bottom)"
    assert mode in ["constant", "replicate", "reflect"], "Currently pad op only supports constant/replicate/reflect mode"

    mode_index = {
        "constant"  : 0,
        "replicate" : 1,
        "reflect"   : 2,
    }

    attrs = list(pad) + [mode_index[mode], channel_last]
    return op("pad", name, operandA, attrs=attrs,).get_tensor()


def PadTile(
        name: str,
        operandA: Tensor,
        dim: int,
        original_length: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension which to pad to tile dim

    original_length: int
        Original length of the dimension before calling this function

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("pad_tile", name, operandA, attrs=(dim, original_length)).get_tensor()


def Broadcast(
        name: str,
        operandA: Tensor,
        dim: int,
        shape: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to broadcast 

    shape: int
        Output length of dim 

    Returns
    -------
    Tensor
        Buda tensor
    """
  
    return op("broadcast", name, operandA, attrs=(dim, shape, True)).get_tensor()


def Repeat(
        name: str,
        operandA: Tensor,
        factors: List[int]) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    factors: List[int]
        Multiplier on respective dim

    Returns
    -------
    Tensor
        Buda tensor
    """
    assert len(operandA.shape) == len(factors)
    return op("repeat", name, operandA, attrs=factors).get_tensor()

def Unsqueeze(
        name: str,
        operandA: Tensor,
        dim: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to broadcast 

    Returns
    -------
    Tensor
        Buda tensor
    """
  
    return op("unsqueeze", name, operandA, attrs=(dim, len(operandA.shape))).get_tensor()

def Squeeze(
        name: str,
        operandA: Tensor,
        dim: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension to broadcast 

    Returns
    -------
    Tensor
        Buda tensor
    """
  
    return op("squeeze", name, operandA, attrs=(dim,)).get_tensor()


def Narrow(
        name: str,
        operandA: Tensor,
        dim: int,
        start: int,
        length: int,
        original_length: int) -> Tensor:
    """
    TM

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    dim: int
        Dimension which to pad to tile dim

    start: int
        Start index in the dimension to be narrowed

    length: int
        Number of items to take from start

    original_length: int
        Original length of the dimension before calling this function

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("narrow", name, operandA, attrs=(dim, start, length, original_length)).get_tensor()


def PixelShuffle(
        name: str,
        operandA: Tensor,
        upscale_factor: int) -> Tensor:
    """
    Pixel shuffle operation.
    
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
    return op("pixel_shuffle", name, operandA, attrs=(upscale_factor,)).get_tensor()


def BudaPad(
        name: str,
        operandA: Tensor,
        paddings: Tuple[int, int],
        value: float) -> Tensor:
    """
    Pad operation that expands a given tensor with arbitrary number of tiles by any dimension.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    paddings: Tuple[int, int]
        Tuple of paddings for R and C dimensions

    value: float
        Value to pad with
    """
    return op("buda_pad", name, operandA, attrs=(paddings[0], paddings[1], value)).get_tensor()


def BudaUnpad(
        name: str,
        operandA: Tensor,
        original_length: Tuple[int, ...],
        paddings: Tuple[int, int]) -> Tensor:
    """
    Unpad operation that removes arbitrary number of tiles by any dimension.
    
    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset
        
    operandA: Tensor
        First operand

    original_length: Tuple[int, ...]
        Original length of R and C dimensions before padding

    paddings: Tuple[int, int]
        Tuple of paddings for R and C dimensions
    """
    return op("buda_unpad", name, operandA, attrs=(paddings[0], paddings[1], original_length[0], original_length[1])).get_tensor()
