# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union

from ..tensor import Tensor
from ..parameter import Parameter
from .common import PyBudaOp as op
from pybuda.pybudaglobal import get_unique_node_id
from pybuda import DataFormat

def Matmul(
        name: str, 
        operandA: Tensor, 
        operandB: Union[Tensor, Parameter], 
        bias: Optional[Union[Tensor, Parameter]] = None) -> Tensor:
    """
    Matrix multiplication transformation on input activations, with optional bias. y = ab + bias

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    operandA: Tensor
        Input operand A

    operandB: Tensor
        Input operand B

    bias: Tenor, optional
        Optional bias tensor
    """

    if name == "":
        name = f"matmul_{get_unique_node_id()}"
    result : Tensor = op("matmul", name, operandA, operandB).get_tensor()

    if bias is not None:
        result = op("add", name + ".bias", result, bias).get_tensor()

    return result

def SparseMatmul(
        name: str, 
        sparseA: Tensor, 
        denseB: Tensor) -> Tensor:
    """
    Sparse matrix multiplication transformation on input activations. y = ab

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    sparseA: Tensor
        Input operand A sparse tensor

    denseB: Tensor
        Input operand B dense tensor
    """
    assert sparseA.has_value()
    assert sparseA.value().is_sparse

    return op("sparse_matmul", name, sparseA, denseB).get_tensor()

