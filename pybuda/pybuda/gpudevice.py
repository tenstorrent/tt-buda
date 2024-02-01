# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from .cpudevice import CPUDevice

class GPUDevice(CPUDevice):
    """
    GPUDevice represents a GPU processor. It will spawn a process and run local operations on the assigned processor.
    """

    def __init__(self,
        *args,
        **kwargs
    ):
        """
        Create a GPU device with a given name.
        """
        super().__init__(*args, **kwargs)
        self.device = "cuda"

    def __repr__(self):
        return f"GPUDevice '{self.name}'"
