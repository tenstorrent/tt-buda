# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List

from ..tensor import Tensor
from ..parameter import Parameter
from .common import PyBudaOp as op

RESIZE2d_METHOD_TO_INT = {
    "nearest_neighbor" : 0,
    "linear"           : 1,
    "bilinear"           : 1,
}

INT_TO_RESIZE2d_METHOD = {
    0    : "nearest",
    1    : "bilinear",
}

def Resize2d(
    name: str,
    operandA: Tensor,
    sizes: List[int],
    method: str = "nearest_neighbor",
    align_corners=False,
    extrapolation_value: int = 0,
    channel_last: bool = False,
) -> Tensor:
    """
    Resize input activations, with default method 'nearest_neighbor'

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    sizes: List[int]
        The target 2D sizes to extrapolate to

    method: str
        Extrapolation method

    extrapolation_value: int

    """
    assert len(sizes) == 2
    assert method == "nearest_neighbor" or method == "linear" or method == "bilinear", "Only support nearest_neighbor and linear interpolation for now"
    result : Tensor = op("resize2d", name, operandA, attrs=(*sizes, RESIZE2d_METHOD_TO_INT[method], int(align_corners), int(channel_last))).get_tensor()

    return result


def Resize3d(
    name: str,
    operandA: Tensor,
    sizes: List[int],
    method: str = "nearest_neighbor",
    align_corners=False,
    extrapolation_value: int = 0,
    channel_last: bool = False,
) -> Tensor:
    """
    Resize input activations, with default method 'nearest_neighbor'

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    sizes: List[int]
        The target 2D sizes to extrapolate to

    method: str
        Extrapolation method

    extrapolation_value: int

    """
    assert len(sizes) == 3
    assert method == "nearest_neighbor", "Only support nearest_neighbor for now"
    assert not channel_last, "Decomposition for channel-last Resize3d is not added yet"
    result : Tensor = op("resize3d", name, operandA, attrs=(*sizes, RESIZE2d_METHOD_TO_INT[method], int(align_corners), int(channel_last))).get_tensor()

    return result
