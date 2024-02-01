# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable
from typing import Dict, Callable

import torch

from .optimizers import Optimizer
from pybuda.torch_schedulers import TorchLearningRateScheduler


class LearningRateScheduler:
    """
    Learning rate scheduler base class
    """

    def __init__(self, optimizer: Optimizer):
        """
        Constructs a baseline learning rate scheduler which updates the learning rate of an optimizer
        based on the next value of some step function written by the user
        """

        assert isinstance(optimizer, Optimizer), "'optimizer' must be a TT optimizer"
        self.optimizer = optimizer

        # Verification attributes
        self.torch_scheduler = None

    def step(self):
        """
        Calls the 'get_lr' method which fetches the next learning rate and then
        updates the optimizer learning rate with this value
        """
        new_lr = self.get_lr()
        self.optimizer.set_optimizer_parameters(learning_rate=new_lr)

    def get_lr(self):
        """
        Returns or yields the optimizer's next learning rate.
        """
        raise RuntimeError("Needs to be implemented for child class")

    def get_scheduler_params(self):
        """
        Returns optimizer params that the scheduler accesses
        """
        raise RuntimeError("Needs to be implemented for child class")


    def get_pytorch_scheduler(self, optimizer: torch.optim.Optimizer):
        """
        Returns an equivalent pytorch scheduler, used for verification.
        """
        if self.torch_scheduler is None:
            self.torch_scheduler = TorchLearningRateScheduler(
                optimizer=optimizer
            )

        return self.torch_scheduler
