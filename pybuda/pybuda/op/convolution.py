# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Tuple, List

from ..tensor import Tensor
from ..parameter import Parameter
from .common import PyBudaOp as op

from pybuda.op.eval.sparse_utils import conv2d_padding_to_canonical, conv3d_padding_to_canonical


def Conv2d(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]] = None,
    stride: int = 1,
    padding: Union[int, str, List] = "same",
    dilation: int = 1,
    groups: int = 1,
    channel_last: bool = False,
) -> Tensor:
    """
    Conv2d transformation on input activations, with optional bias.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iH, iW)

    weights:
        Tensor
            Input weights of shape (Cout, Cin / groups, kH, kW)
        [Tensor]
            Internal Use pre-split
            Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)]
            of length: (K*K // weight_grouping)

    bias: Tenor, optional
        Optional bias tensor of shape (Cout)
    """
    if isinstance(stride, int):
        stride = [stride] * 2

    padding = conv2d_padding_to_canonical(padding, (weights.shape[2], weights.shape[3]))

    inputs = [activations, weights]
    if bias is not None:
        inputs.append(bias)

    # Attrs are:
    # [
    #     stride_height,
    #     stride_width,
    #     dilation,
    #     groups,
    #     padding_left,
    #     padding_right,
    #     padding_top,
    #     padding_bottom,
    #     is_convtranspose2d,
    #     output_height_transpose,
    #     output_width_transpose,
    #     stride_transpose,
    #     channel_last,
    # ]
    attrs = stride + [dilation, groups] + padding + [False, 0, 0, 0, channel_last]

    return op(
        "conv2d",
        name,
        *inputs,
        attrs=attrs,
    ).get_tensor()


def Conv2dTranspose(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]] = None,
    stride: int = 1,
    padding: Union[int, str] = "same",
    dilation: int = 1,
    groups: int = 1,
    channel_last: bool = False,
) -> Tensor:
    """
    Conv2dTranspose transformation on input activations, with optional bias.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, iH, iW)

    weights:
        Tensor
            Input weights of shape (Cout, Cin / groups, kH, kW)
        [Tensor]
            Internal Use pre-split
            Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)]
            of length: (K*K // weight_grouping)

    bias: Tenor, optional
        Optional bias tensor of shape (Cout)
    """
    if isinstance(stride, int):
        stride = [stride] * 2

    if padding == "same":
        padding = [weights.shape[3] // 2] * 2 + [weights.shape[2] // 2] * 2

    if isinstance(padding, int):
        padding = [padding] * 4  # [left, right, top, bottom]

    inputs = [activations, weights]
    if bias is not None:
        inputs.append(bias)

    # Attrs are:
    # [
    #     stride_height,
    #     stride_width,
    #     dilation,
    #     groups,
    #     padding_left,
    #     padding_right,
    #     padding_top,
    #     padding_bottom,
    #     channel_last,
    # ]
    attrs = stride + [dilation, groups] + padding + [channel_last]
    return op(
        "conv2d_transpose",
        name,
        *inputs,
        attrs=attrs,
    ).get_tensor()


def Conv3d(
    name: str,
    activations: Tensor,
    weights: Union[Tensor, Parameter],
    bias: Optional[Union[Tensor, Parameter]] = None,
    stride: int = 1,
    padding: Union[int, str, List] = "same",
    dilation: int = 1,
    groups: int = 1,
    channel_last: bool = False,
) -> Tensor:
    """
    Conv3d on input activations, with optional bias.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    activations: Tensor
        Input activations of shape (N, Cin, Din, iH, iW)

    weights:
        Tensor
            Input weights of shape (Cout, Cin / groups, kD, kH, kW)
        [Tensor]
            Internal Use pre-split
            Optional Input weights list of shape [(weight_grouping, Cin / groups, Cout)]
            of length: (K*K // weight_grouping)

    bias: Tenor, optional
        Optional bias tensor of shape (Cout)
    """
    assert not channel_last, "Decomposition for channel-last Conv3d is not added yet"

    if isinstance(stride, int):
        stride = [stride] * 3

    padding = conv3d_padding_to_canonical(padding, (weights.shape[2],weights.shape[3],weights.shape[4]))

    inputs = [activations, weights]
    if bias is not None:
        inputs.append(bias)

    attrs = stride + [dilation, groups] + padding + [channel_last]
    return op(
        "conv3d",
        name,
        *inputs,
        attrs=attrs,
    ).get_tensor()


