# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union

from ..tensor import Tensor
from ..parameter import Parameter
from .common import PyBudaOp as op

def Embedding(
        name: str, 
        embedding_table: Union[Tensor, Parameter],
        indices: Tensor) -> Tensor:
    """
    Embedding lookup

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    indices: Tensor
        Integer tensor, the elements of which are used to index into the embedding table

    embedding_table: Tensor
        Dictionary of embeddings
    """

    return op("embedding", name, embedding_table, indices).get_tensor()
