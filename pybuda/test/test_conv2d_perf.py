# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import pybuda
from pybuda.op.nn import Conv2dModule, MaxPool2dModule
from pybuda.verify import VerifyConfig

def get_relaxed_atol_pcc(test_kind, test_device, microbatch_size = 1):
    """
    Figure out reasonable pcc/atol for training on silicon
    """
    training_atol = 0.3
    training_pcc = 0.95
    if test_device.is_silicon():
        training_pcc = 0.8
        training_atol = 0.55
    inference_atol = 0.1
    inference_pcc = 0.95
    relative_atol = training_atol if test_kind.is_training() else inference_atol
    if test_device.is_silicon() and test_kind.is_training():
        relative_atol *= 3.5
    pcc = training_pcc if test_kind.is_training() else inference_pcc

    return relative_atol, pcc

@pytest.mark.parametrize("kernel", [1, 3, 7])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("input_dim", [224, 64, 14])
@pytest.mark.parametrize("input_c", [3, 32, 64, 256])
@pytest.mark.parametrize("output_c", [32, 64, 256])
def test_single_conv(test_kind, test_device, kernel, stride, input_dim, input_c, output_c):

    mod = Conv2dModule(
        "conv2d_perf",
        input_c,
        output_c,
        (kernel, kernel),
        stride=stride,
        padding=kernel//2,
        t_stream_workaround=True,
    )
        
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    pybuda.config._get_global_compiler_config().performance_trace = pybuda.config.PerfTraceLevel.VERBOSE
    pybuda.config.override_op_size("conv2d_perf.dc.buffer.12", (1, 9))
    #pybuda.config.override_op_size("conv2d_perf.dc.matmul.9", (8, 1))
    #pybuda.config.override_t_stream_shape("conv2d_perf.dc.matmul.9", (1, 1))
    pybuda.config.override_op_size("conv2d_perf.dc.sparse_matmul.18.lc2", (8, 1))
    pybuda.config.override_t_stream_dir("conv2d_perf.dc.sparse_matmul.18.lc2", "r")
    pybuda.config.override_t_stream_shape("conv2d_perf.dc.sparse_matmul.18.lc2", (16, 1))
    #pybuda.config.override_t_stream_shape("conv2d_perf.dc.sparse_matmul.18.lc2", (1, 1))
    pybuda.verify.verify_module(mod, [(16, input_c, input_dim, input_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc, fp32_fallback=pybuda.DataFormat.Bfp8_b))

@pytest.mark.parametrize("kernel", [1, 3, 7])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("input_dim", [224, 64, 14])
@pytest.mark.parametrize("input_c", [3, 32, 64, 256])
def test_single_maxpool(test_kind, test_device, kernel, stride, input_dim, input_c):

    mod = MaxPool2dModule(
        "maxpool_perf",
        kernel,
        stride=stride,
    )
        
    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)
    pybuda.config._get_global_compiler_config().performance_trace = pybuda.config.PerfTraceLevel.VERBOSE
    pybuda.config.override_op_size("maxpool_perf.dc.sparse_matmul.5.lc2", (6, 1))
    #pybuda.config.override_t_stream_shape("conv2d_perf.dc.matmul.9", (1, 1))
    #pybuda.config.override_op_size("conv2d_perf.dc.sparse_matmul.18.lc2", (8, 1))
    #pybuda.config.override_t_stream_dir("conv2d_perf.dc.sparse_matmul.18.lc2", "r")
    #pybuda.config.override_t_stream_shape("conv2d_perf.dc.sparse_matmul.18.lc2", (16, 1))
    #pybuda.config.override_t_stream_shape("conv2d_perf.dc.sparse_matmul.18.lc2", (1, 1))
    pybuda.verify.verify_module(mod, [(16, input_c, input_dim, input_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc, fp32_fallback=pybuda.DataFormat.Bfp8_b))

