# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Simple Graph CNN basic bring-up tests of tracing functionality
#
import pytest

import torch

from pybuda import (
    PyTorchModule,
    VerifyConfig,
)
# from torch_geometric.nn import GCNConv

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind


def test_tvm_graph_cnn(test_kind, test_device):
    # Scatter Addition op is not supported in PyBuda. Can be revised
    # once embeddings (over take op) are supported on HW side
    pytest.skip()

    if test_kind == TestKind.TRAINING:  
        # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    class GCNWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.num_features = 1433
            self.num_classes = 7
            self.conv = GCN(self.num_features, self.num_classes)
            self.edge_index = torch.randint(0, 2708, (1, 2, 10556), dtype=torch.int64)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x, self.edge_index)

    class GCN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = GCNConv(in_channels, out_channels)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            if len(x.shape) != 2:
                x = x.squeeze(0)

            if len(edge_index.shape) != 2:
                edge_index = edge_index.squeeze(0)

            x = self.conv(x, edge_index)

            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.START_COMPILE
    else:
        compiler_cfg.compile_depth = CompileDepth.START_COMPILE

    pytorch_model = GCNWrapper()
    module = PyTorchModule("graph_cnn", pytorch_model)

    x = torch.rand((1, 2708, 1433))

    verify_module(
        module,
        (),
        inputs=[(x,),],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
