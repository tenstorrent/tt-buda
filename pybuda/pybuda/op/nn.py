# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.op.matmul import SparseMatmul
import math
import torch
from typing import Union

from ..tensor import Tensor, TensorShape
from ..parameter import Parameter
from ..module import PyBudaModule
from .common import PyBudaOp as op

from .eltwise_unary import Exp, Reciprocal, Sqrt
from .eltwise_binary import Multiply, Subtract, Add
from .reduce import ReduceSum, ReduceAvg
from .constant import Constant
from .convolution import Conv2d, Conv2dTranspose
from .pooling import MaxPool1d, MaxPool2d, AvgPool2d
from .matmul import Matmul
from .tm import Reshape, Transpose, Unsqueeze
from pybuda.pybudaglobal import get_unique_node_id

import os

def Softmax(
    name: str, 
    operandA: Tensor, *, 
    dim: int, 
    stable: bool = True) -> Tensor:

    """
    Softmax operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

    stable: bool
        Use stable softmax or not.

    Returns
    -------
    Tensor
        Buda tensor
    """
    return op("softmax", name, operandA, attrs=(dim, stable)).get_tensor()


def LogSoftmax(
    name: str, 
    operandA: Tensor, *, 
    dim: int, 
    stable: bool = True) -> Tensor:

    """
    LogSoftmax operation.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    dim: int
        – A dimension along which Softmax will be computed

    stable: bool
        Use stable softmax or not.

    Returns
    -------
    Tensor
        Buda tensor
    """
    return op("log_softmax", name, operandA, attrs=(dim, stable)).get_tensor()

def Layernorm(
        name: str,
        operandA: Tensor,
        weights: Union[Tensor, Parameter],
        bias: Union[Tensor, Parameter],
        dim: int = -1,
        epsilon: float = 1e-5) -> Tensor:
    """
    Layer normalization.

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

    if name == "":
        name = f"layernorm_{get_unique_node_id()}"

    operand_shape = operandA.shape
    layernorm_flag = True
    for item in operand_shape[:-2]:
        if item != 1:
            layernorm_flag &= False
            break

    if (dim != -1 and dim != len(operandA.shape) - 1) or not layernorm_flag:

        epsilon_constant = Constant(name + "_eps", constant=epsilon)
        mean = ReduceAvg(name + "_mean", operandA, dim)
        x_minus_mean = Subtract(name + "_sub", operandA, mean)
        squared = Multiply(name + "_sq", x_minus_mean, x_minus_mean)
        var = ReduceAvg(name + "_var", squared, dim)
        var_plus_eps = Add(name + "_var_plus_eps", var, epsilon_constant)
        recip = Reciprocal(name + "_recip", Sqrt(name + "_sqrt", var_plus_eps))
        out = Multiply(name + "_output", x_minus_mean, recip)
        return Add(name + "_bias", Multiply(name + "_weights", out, weights), bias)

    else:
        return op("layernorm", name, operandA, weights, bias, attrs=(dim, epsilon)).get_tensor()

def Batchnorm(
        name: str,
        operandA: Tensor,
        weights: Union[Tensor, Parameter],
        bias: Union[Tensor, Parameter],
        running_mean: Union[Tensor, Parameter],
        running_var: Union[Tensor, Parameter],
        epsilon: float = 1e-5) -> Tensor:
    """
    Batch normalization.

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

    # NOTE: the decomposition below does not assume training context (running_mean/var update is not included)
    batchnorm_flag = True
    if name == "":
        name = f"batchnorm_{get_unique_node_id()}"

    if batchnorm_flag:
        return op("batchnorm", name, operandA, weights, bias, running_mean, running_var, attrs=(epsilon,)).get_tensor()
    else:
        running_mean = Unsqueeze(name + "_mean_unsqueeze_1", running_mean, 1)
        running_mean = Unsqueeze(name + "_mean_unsqueeze_2", running_mean, 1)
        running_var = Unsqueeze(name + "_var_unsqueeze_1", running_var, 1)
        running_var = Unsqueeze(name + "_var_unsqueeze_2", running_var, 1)

        weights = Unsqueeze(name + "_weights_unsqueeze_1", weights, 1)
        weights = Unsqueeze(name + "_weights_unsqueeze_2", weights, 1)
        bias = Unsqueeze(name + "_bias_unsqueeze_1", bias, 1)
        bias = Unsqueeze(name + "_bias_unsqueeze_2", bias, 1)
        epsilon_constant = Constant(name + "_eps", constant=epsilon)
        x_minus_mean = Subtract(name + "_sub", operandA, running_mean)
        var_plus_eps = Add(name + "_var_plus_eps", running_var, epsilon_constant)
        recip = Reciprocal(name + "_recip", Sqrt(name + "_sqrt", var_plus_eps))
        out = Multiply(name + "_output", x_minus_mean, recip)
        return Add(name + "_bias", Multiply(name + "_weights", out, weights), bias)


class Linear(PyBudaModule):
    """
    Linear transformation module.

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    in_features: int
        Number of input features

    out_features: int
        Number of output features

    bias: bool
        Include a bias

    Returns
    -------
    Tensor
        Buda tensor
    """

    def __init__(
        self,
        name,
        in_features,
        out_features,
        bias=True
    ):
        super().__init__(name)

        self.in_features = in_features
        self.out_features = out_features
        self.bias_ = bias

        self.weights = Parameter(self.in_features, self.out_features)
        weights_tensor = torch.empty(self.weights.shape.get_pytorch_shape())
        torch.nn.init.kaiming_uniform_(weights_tensor, a=math.sqrt(5))
        self.set_parameter("weights", weights_tensor)

        if self.bias_:
            bias_tensor = torch.empty(self.bias.shape.get_pytorch_shape())
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights.value())
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias_tensor, -bound, bound)
            self.set_parameter("bias", bias_tensor)

    def forward(self, activations):
        return Matmul(
            name=self.name,
            operandA=activations,
            operandB=self.weights,
            bias=self.bias if self.bias_ else None
        )


class Conv2dModule(PyBudaModule):
    """
    Conv2dModule
    """

    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=True,
        t_stream_workaround=False, # add reshape/transpose to allow isolated conv2d to t-stream
    ):
        super().__init__(name)
        assert (in_channels % groups) == 0, f"{in_channels} {groups}"
        self.kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        self.t_stream_workaround = t_stream_workaround

        # Using pytorch style initialization:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L146
        # https://github.com/pytorch/pytorch/issues/15314

        self.weights = Parameter(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        weights_tensor = torch.empty(self.weights.shape.get_pytorch_shape())
        torch.nn.init.kaiming_uniform_(weights_tensor, a=math.sqrt(5))
        self.set_parameter("weights", weights_tensor)

        self.bias = (
            Parameter(
                out_channels,
                requires_grad=True,
            )
            if bias
            else None
        )
        if bias:
            bias_tensor = torch.empty(self.bias.shape.get_pytorch_shape())
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights.value())
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(bias_tensor, -bound, bound)
            self.set_parameter("bias", bias_tensor)

    def forward(self, activations):
        m1 = Conv2d(
            self.name, activations, self.weights, bias=self.bias, **self.kwargs
        )
        if self.t_stream_workaround:
            m1 = Reshape("", m1, (1, 1, m1.shape[-3], m1.shape[-2] * m1.shape[-1]))
            m1 = Transpose("", m1, -2, -1)

        return m1


class ConvTranspose2dModule(PyBudaModule):
    """
    ConvTranspose2dModule
    """

    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        # t_stream_workaround=False, # add reshape/transpose to allow isolated conv2d to t-stream
    ):
        super().__init__(name)
        assert out_channels % groups == 0, f"{in_channels} {groups}"
        assert padding_mode == "zeros", "Only \"zeros\" supported for padding_mode"
        assert isinstance(output_padding, int) and output_padding == 0, "output_padding not supported yet, can only be 0"

        self.kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        # self.t_stream_workaround = t_stream_workaround

        # Using pytorch style initialization:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L146
        # https://github.com/pytorch/pytorch/issues/15314

        self.weights = Parameter(in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        weights_tensor = torch.empty(self.weights.shape.get_pytorch_shape())
        torch.nn.init.kaiming_uniform_(weights_tensor, a=math.sqrt(5))
        self.set_parameter("weights", weights_tensor)

        self.bias = (
            Parameter(
                out_channels,
                requires_grad=True,
            )
            if bias
            else None
        )
        if bias:
            bias_tensor = torch.empty(self.bias.shape.get_pytorch_shape())
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights.value())
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(bias_tensor, -bound, bound)
            self.set_parameter("bias", bias_tensor)

    def forward(self, activations):
        m1 = Conv2dTranspose(
            self.name, activations, self.weights, bias=self.bias, **self.kwargs
        )
        # if self.t_stream_workaround:
        #     m1 = Reshape("", m1, (1, 1, m1.shape[-3], m1.shape[-2] * m1.shape[-1]))
        #     m1 = Transpose("", m1, -2, -1)

        return m1


class MaxPool1dModule(PyBudaModule):
    """
    MaxPool1dModule
    """

    def __init__(
        self,
        name,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
    ):
        super().__init__(name)
        self.kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
            "return_indices": return_indices,
        }

    def forward(self, activations):
        return MaxPool1d(self.name, activations, **self.kwargs)


class MaxPool2dModule(PyBudaModule):
    """
    MaxPool2dModule
    """

    def __init__(
        self,
        name,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        ceil_mode=False,
        return_indices=False,
    ):
        super().__init__(name)
        self.kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
            "return_indices": return_indices,
        }

    def forward(self, activations):
        return MaxPool2d(self.name, activations, **self.kwargs)


class AvgPool2dModule(PyBudaModule):
    """
    AvgPool2dModule
    """

    def __init__(
        self,
        name,
        kernel_size,
        stride=1,
        padding="same",
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        super().__init__(name)
        self.kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
            "count_include_pad": count_include_pad,
            "divisor_override": divisor_override,
        }

    def forward(self, activations):
        return AvgPool2d(self.name, activations, **self.kwargs)


class SparseMatmulModule(PyBudaModule):
    """
    SparseMatmulModule
    """

    def __init__(
        self,
        name: str,
        sparseA: Tensor,
    ):
        super().__init__(name)

        self.sparseA = Parameter(*sparseA.value().shape, requires_grad=False, name="sparseA")
        self.set_parameter("sparseA", sparseA.value())

    def forward(self, denseB):
        m1 = SparseMatmul(
            self.name, self.sparseA, denseB
        )
        return m1
