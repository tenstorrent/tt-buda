# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable

from torch.optim.lr_scheduler import _LRScheduler
import torch


class TorchLearningRateScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Updates a given optimizer's learning rate based on the values yielded
    by an iterable
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lrs = [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]

    def step(self, epoch=None):
        param_group = self.optimizer.param_groups[0]
        if 'step' in param_group:
            self.last_epoch = param_group['step'] + 1
            param_group['step'] += 1
        else:
            self.last_epoch = 1
            param_group['step'] = 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        raise RuntimeError("'get_lr' needs to be implemented for the child class")