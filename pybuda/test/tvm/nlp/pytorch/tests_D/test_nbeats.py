# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from pybuda.config import CompileDepth
from pytorch_forecasting.models.nbeats.sub_modules import NBEATSBlock, NBEATSGenericBlock, NBEATSTrendBlock, NBEATSSeasonalBlock

import torch
import pytest

from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from test.tvm.utils import evaluate_framework_vs_pybuda

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind


def test_tvm_nbeats_block(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    input_shape = (1, 64, 64, 64)
    model = NBEATSBlock(100, 100, backcast_length=input_shape[-1])

    mod = PyTorchModule("nbeats_block", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )



def test_tvm_nbeats_generic_block(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER


    input_shape = (1, 64, 64, 64)

    model = NBEATSGenericBlock(100, 100, backcast_length=input_shape[-1])

    mod = PyTorchModule("nbeats_generic_block", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

def test_tvm_nbeats_seasonal_block(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    
    input_shape = (1, 64, 64, 64)

    class NBeatsSeasonal(NBEATSSeasonalBlock):
        def __init__(
            self,
            units,
            thetas_dim,
            backcast_length
        ):
            super(). __init__(
                units,
                thetas_dim=thetas_dim,
                backcast_length=backcast_length,
            )

        def forward(self, x):
            x = super(NBEATSSeasonalBlock, self).forward(x)
            amplitudes_backward = self.theta_b_fc(x)
            backcast = amplitudes_backward.matmul(self.S_backcast)
            amplitudes_forward = self.theta_f_fc(x)
            forecast = amplitudes_forward.matmul(self.S_forecast)

            return backcast, forecast

    model = NBeatsSeasonal(100, 100, backcast_length=input_shape[-1])

    mod = PyTorchModule("nbeats_seasonal_block", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_nbeats_trend_block(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    input_shape = (1, 64, 64, 64)

    class NBeatsTrend(NBEATSTrendBlock):
        def __init__(
            self,
            units,
            thetas_dim,
            backcast_length
        ):
            super(). __init__(
                units,
                thetas_dim=thetas_dim,
                backcast_length=backcast_length,
            )

        def forward(self, x):
            x = super(NBEATSTrendBlock, self).forward(x)
            backcast = self.theta_b_fc(x).matmul(self.T_backcast)
            forecast = self.theta_f_fc(x).matmul(self.T_forecast)
            return backcast, forecast

    model = NBeatsTrend(100, 100, backcast_length=input_shape[-1])

    mod = PyTorchModule("nbeats_trend_block", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )