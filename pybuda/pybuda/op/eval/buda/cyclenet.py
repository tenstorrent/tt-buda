# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Might be an overkill for predicting simple functions, consider parameterizing # of layers too
class CycleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CycleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out) # makes no sense to have negative cycles
        return out

class CycleNetConfig:
    config = {
        'matmul': [13, 64, 1],
        'default': [7, 32, 1],
    }
    def __init__(self, op_type):
        if op_type not in self.config:
            print(f"Warning: op_type {op_type} not found in config. Using default config.")
            op_type = 'default'

        self._input_dim = self.config[op_type][0]
        self._hidden_dim = self.config[op_type][1]
        self._output_dim =  self.config[op_type][2]

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def output_dim(self):
        return self._output_dim

# self._data_formats = ['none', 'float16', 'float16_b', 'bfp8', 'bfp8_b', 'bfp4', 'bfp4_b', 'bfp2', 'bfp2_b', 'float32', 'tf32', 'lf8']
# self._fidelity_phases = ['none', 'lofi', 'hifi2', 'hifi3', 'hifi4']
