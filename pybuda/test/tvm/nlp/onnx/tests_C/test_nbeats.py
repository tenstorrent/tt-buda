# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import onnx
import torch
import pytest
from pytorch_forecasting.models.nbeats.sub_modules import (
    NBEATSBlock,
    NBEATSGenericBlock,
    NBEATSTrendBlock,
    NBEATSSeasonalBlock,
)

from pybuda import (
    OnnxModule,
    BackendType,
    VerifyConfig,
)
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import CompileDepth


def test_tvm_nbeats_block(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # Unsupported HW ops
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    if not test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    # Configure PyTorch module
    input_shape = (1, 64, 64, 64)
    pytorch_module = NBEATSBlock(100, 100, backcast_length=input_shape[-1])

    # Export to ONNX
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/nbeats_block.onnx"
    torch.onnx.export(
        pytorch_module,
        torch.rand(input_shape),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "nbeats_block_onnx",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

    # Cleanup
    os.remove(save_path)


def test_tvm_nbeats_generic_block(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # Unsupported HW ops
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    if not test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    # Configure PyTorch module
    input_shape = (1, 64, 64, 64)
    pytorch_module = NBEATSGenericBlock(100, 100, backcast_length=input_shape[-1])

    # Export to ONNX
    save_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/nbeats_generic_block.onnx"
    )
    torch.onnx.export(
        pytorch_module,
        torch.rand(input_shape),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "nbeats_generic_block_onnx",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

    # Cleanup
    os.remove(save_path)


def test_tvm_nbeats_seasonal_block(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class NBeatsSeasonal(NBEATSSeasonalBlock):
        def __init__(self, units, thetas_dim, backcast_length):
            super().__init__(
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

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # Unsupported HW ops
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    if not test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    # Configure PyTorch module
    input_shape = (1, 64, 64, 64)
    pytorch_module = NBeatsSeasonal(100, 100, backcast_length=input_shape[-1])

    # Export to ONNX
    save_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/nbeats_seasonal_block.onnx"
    )
    torch.onnx.export(
        pytorch_module,
        torch.rand(input_shape),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "nbeats_seasonal_block_onnx",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

    # Cleanup
    os.remove(save_path)


def test_tvm_nbeats_trend_block(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class NBeatsTrend(NBEATSTrendBlock):
        def __init__(self, units, thetas_dim, backcast_length):
            super().__init__(
                units,
                thetas_dim=thetas_dim,
                backcast_length=backcast_length,
            )

        def forward(self, x):
            x = super(NBEATSTrendBlock, self).forward(x)
            backcast = self.theta_b_fc(x).matmul(self.T_backcast)
            forecast = self.theta_f_fc(x).matmul(self.T_forecast)
            return backcast, forecast

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # Unsupported HW ops
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    if not test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    # Configure PyTorch module
    input_shape = (1, 64, 64, 64)
    pytorch_module = NBeatsTrend(100, 100, backcast_length=input_shape[-1])

    # Export to ONNX
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/nbeats_trend_block.onnx"
    torch.onnx.export(
        pytorch_module,
        torch.rand(input_shape),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "nbeats_trend_block_onnx",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

    # Cleanup
    os.remove(save_path)
