# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from ..tensor import Tensor
from .common import PyBudaOp as op

DEFAULT_NUM_ENTRIES = 4 # configured as a heuristic to hide DRAM latency

def DRAMQueue(name: str, operandA: Tensor, *, num_entries: int = DEFAULT_NUM_ENTRIES) -> Tensor:
    """
    Explicit operation in the graph to buffer the input operand data through
    DRAM to its consumer(s).

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    num_entries: int
        configuration for the number of entries that can be stored in the queue

    Returns
    -------
    Tensor
        Buda tensor
    """

    return op("dram_queue", name, operandA, attrs=(num_entries,)).get_tensor()

