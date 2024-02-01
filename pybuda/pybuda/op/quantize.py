# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..tensor import Tensor, pytorch_dtype_to_buda_dataformat
from .common import PyBudaOp as op
import torch

def Quantize(
        name: str,
        operandA: Tensor,
        operandB: Tensor,
        zero_point: int, 
        axis: int,
        out_dtype: torch.dtype) -> Tensor:
    """
    Quantize input tensor to INT using provided zero point and scale

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Operand to be quantized

    operandB: Tensor
        scale for quantization

    zero_point: int
        zero point for quantization

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    out_dtype : torch.dtype
        The data type of the input tensor. Can be [int8, uint8, int32]

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("quantize", name, operandA, operandB, attrs=(zero_point, axis, str(out_dtype))).get_tensor(out_df=pytorch_dtype_to_buda_dataformat(out_dtype))


def Dequantize(
        name: str,
        operandA: Tensor,
        operandB: Tensor,
        zero_point: int,
        axis: int = -1,
        out_dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Dequantize input tensor to Float using provided zero point and scale

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Operand to be dequantized

    operandB: Tensor
        scale for dequantization

    zero_point: int
        zero point for quantization

    Returns
    -------
    Tensor
        Buda tensor
    """


    return op("dequantize", name, operandA, operandB, attrs=(zero_point, axis,)).get_tensor(out_df=pytorch_dtype_to_buda_dataformat(out_dtype))

def Requantize(
        name: str,
        operandA: Tensor,
        operand_inp_scale: Tensor,
        operand_out_scale: Tensor,
        input_zero_point: float,
        output_zero_point: float,
        out_dtype: torch.dtype,
        axis: int =-1,
        rounding: str = "None",) -> Tensor:
    """
    ReQuantize input tensor using provided zero point and scale

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        First operand

    operand_inp_scale: Tensor

    operand_inp_zp: Tensor

    operand_out_scale: Tensor

    output_zero_point: float

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    rounding : string, optional
        Defines the rounding direction when the value is midway between two
        representable values.

    out_dtype : torch.dtype
        Specifies the output data type.

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op(
        "requantize", name, operandA, operand_inp_scale,operand_out_scale,attrs=(output_zero_point, input_zero_point, axis,rounding,str(out_dtype))).get_tensor(out_df=pytorch_dtype_to_buda_dataformat(out_dtype))


def BudaRequantize(
        name: str,
        operandA: Tensor,
        scale: Tensor,
        zero_point: float,
        out_dtype: torch.dtype,
        axis: int =-1,
        rounding: str = "None") -> Tensor:



    return op(
        "buda_requantize", name, operandA, scale,attrs=(zero_point,axis,rounding,str(out_dtype))).get_tensor(out_df=pytorch_dtype_to_buda_dataformat(out_dtype))