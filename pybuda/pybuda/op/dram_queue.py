# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from ..tensor import Tensor
from .common import PyBudaOp as op


def DRAMQueue(name: str, operandA: Tensor, *, num_entries: int) -> Tensor:
    """
    Explicit operation in the graph to buffer the input operand data through
    DRAM to its consumer(s).

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    num_entries: int
        configuration for the number of entries that can be stored in the queue.
        num_entries shouldn't have default value because if queue turns out to be static it should
        have num_entries equal to microbatch_size. 
        Only in special cases, when we are sure we will need less space than microbatch size, we can 
        set num_entries to something less than microbatch_size.

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("dram_queue", name, operandA, attrs=(num_entries,)).get_tensor()

