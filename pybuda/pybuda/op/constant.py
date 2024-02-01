# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..tensor import Tensor
from .common import PyBudaOp as op
from typing import Union, Tuple, List

def Constant(name: str, *, constant: float) -> Tensor:
    """
    Op representing user-defined constant

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    constant: float
        Constant value

    Returns
    -------
    Tensor
        Buda tensor
    """
    return op("constant", name, attrs=(constant,)).get_tensor()
