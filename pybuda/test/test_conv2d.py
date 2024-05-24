# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from pybuda._C.balancer import OpOverride
from pybuda.verify.config import TestKind
from pybuda._C import DataFormat, MathFidelity
import pytest

import torch

import pybuda
import os
import random
from pybuda import (
    PyBudaModule,
    TTDevice,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    CompileDepth,
    VerifyConfig,
)
from pybuda._C.backend_api import BackendType
from pybuda._C.graph import RuntimeTensorTransformType
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.op.eval import compare_tensor_to_golden, does_prestriding_improve_perf
from pybuda.op.nn import Conv2dModule, ConvTranspose2dModule, MaxPool2dModule, AvgPool2dModule
from pybuda.utils import align_up_tile, round_up_div
from .common import run
from .module_utils import Conv2dTModule
from pybuda.op.eval.sparse_utils import calculate_conv2d_output_dimensions, calculate_conv2d_transpose_output_dimensions, conv2d_padding_to_canonical, calculate_conv2d_prestride_weights_and_padding, can_conv2d_prestride


# TODO: test grouped convs (not depthwise)
@pytest.mark.parametrize("in_channels", [35])
@pytest.mark.parametrize("out_channels", [65])
@pytest.mark.parametrize("kernel_size", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (5, 5), (6, 6), (2,4)])
@pytest.mark.parametrize("original_shape", [(7, 7), (25, 25)], ids=["orig7x7", "orig25x25"])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", ["same", 0, 1, 3, 8])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("depthwise", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("padding_mode", ["zeros"])
# @pytest.mark.parametrize("recompute", [True, False])
def test_conv2d(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    kernel_size,
    original_shape,
    stride,
    padding,
    dilation,
    depthwise,
    bias,
    padding_mode,
):
    training = test_kind.is_training()

    padding = conv2d_padding_to_canonical(padding, kernel_size)
    outy, outx = calculate_conv2d_output_dimensions(original_shape[0], original_shape[1], kernel_size, stride, padding, dilation)
    if bias and (outy * outx) < 32:
        # TODO: Re-enable when bcast on < tile dim isn't broken
        pytest.skip()

    total_kernel_size_x = (kernel_size[0] - 1) * dilation + 1
    total_kernel_size_y = (kernel_size[1] - 1) * dilation + 1
    if total_kernel_size_x > (original_shape[0] + padding[2] + padding[3]) or total_kernel_size_y > (original_shape[1] + padding[0] + padding[1]):
        # Can't have the kernel be larger than the input itself
        pytest.skip()

    groups = 1

    if depthwise:
        out_channels = in_channels
        groups = in_channels

    mod = Conv2dModule(
        "conv2d",
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99

    pybuda.config.set_configuration_options(enable_conv_prestride=False)
    try:
        pybuda.verify.verify_module(mod, [(1, in_channels, original_shape[0], original_shape[1])],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc))
    except RuntimeError as e:
        if (
            "Compile failed for TTDevice" in str(e) or
            "Could not satisfy all constraints for edge" in str(e)
            ):
            pytest.xfail("tenstorrent/pybuda#185")
        raise


@pytest.mark.parametrize("original_shape", [(7, 7), (25, 25), (31, 33)])
@pytest.mark.parametrize("in_channels", [35])
@pytest.mark.parametrize("out_channels", [65])
@pytest.mark.parametrize("kernel_size", [(3, 3), (2, 5)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1, 2, 3])
@pytest.mark.parametrize("output_padding", [0])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("padding_mode", ["zeros"])
# @pytest.mark.parametrize("recompute", [True, False])
def test_convtranspose2d(
    test_kind,
    test_device,
    original_shape,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    output_padding,
    groups,
    bias,
    dilation,
    padding_mode,
):
    compiler_cfg = _get_global_compiler_config()

    training = test_kind.is_training()
    if training:
        pytest.skip()

    padding = conv2d_padding_to_canonical(padding, kernel_size)
    outy, outx = calculate_conv2d_transpose_output_dimensions(original_shape[0], original_shape[1], kernel_size, stride, padding, dilation, output_padding)
    if bias and (outy * outx) < 32:
        # TODO: Re-enable when bcast on < tile dim isn't broken
        pytest.skip()

    total_kernel_size_x = (kernel_size[0] - 1) * dilation + 1
    total_kernel_size_y = (kernel_size[1] - 1) * dilation + 1
    if total_kernel_size_x > (original_shape[0] + padding[2] + padding[3]) or total_kernel_size_y > (original_shape[1] + padding[0] + padding[1]):
        # Can't have the kernel be larger than the input itself
        pytest.skip()

    mod = ConvTranspose2dModule(
        "convtranspose2d",
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        padding_mode=padding_mode
    )

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99

    pybuda.config.set_configuration_options(enable_conv_prestride=False)
    try:
        pybuda.verify.verify_module(mod, [(1, in_channels, original_shape[0], original_shape[1])],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc))
    except RuntimeError as e:
        if "Compile failed for TTDevice" in str(e):
            pytest.xfail("tenstorrent/pybuda#185")
        raise


def test_convtranspose2d_data_mismatch_repro(test_device):
    compiler_cfg = _get_global_compiler_config()

    # Fracturing the conv causes the data mismatch
    # Forcing the fracturing here, so the mismatch repros with small input
    os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    return test_convtranspose2d(
        test_kind=TestKind.INFERENCE,
        test_device=test_device,
        original_shape=(2, 2),
        in_channels=1,
        out_channels=1,
        kernel_size=(2, 2),
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=False,
        dilation=1,
        padding_mode='zeros'
    )


@pytest.mark.parametrize("in_channels", [3])
@pytest.mark.parametrize("out_channels", [64])
@pytest.mark.parametrize("kernel_size", [(3, 3)])
@pytest.mark.parametrize("original_shape", [(16, 16)], ids=["orig16x16"])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("depthwise", [False])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("padding_mode", ["zeros"])
def test_conv2d_t_streaming(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    kernel_size,
    original_shape,
    stride,
    padding,
    dilation,
    depthwise,
    bias,
    padding_mode,
):
    if test_kind != TestKind.INFERENCE:
        pytest.skip()

    padding = conv2d_padding_to_canonical(padding, kernel_size)
    outy, outx = calculate_conv2d_output_dimensions(original_shape[0], original_shape[1], kernel_size, stride, padding, dilation)
    if bias and (outy * outx) < 32:
        # TODO: Re-enable when bcast on < tile dim isn't broken
        pytest.skip()

    total_kernel_size_x = (kernel_size[0] - 1) * dilation + 1
    total_kernel_size_y = (kernel_size[1] - 1) * dilation + 1
    if total_kernel_size_x > (original_shape[0] + padding[2] + padding[3]) or total_kernel_size_y > (original_shape[1] + padding[0] + padding[1]):
        # Can't have the kernel be larger than the input itself
        pytest.skip()

    mod = Conv2dTModule(
        "conv2d_t_streaming",
        in_channels,
        out_channels if not depthwise else in_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=in_channels if depthwise else 1,
        bias=bias,
    )

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"
    compiler_cfg.enable_conv_prestride = False

    pybuda.verify.verify_module(mod, [(1, in_channels, original_shape[0], original_shape[1])],
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc))


@pytest.mark.parametrize("in_channels", [3])
@pytest.mark.parametrize("out_channels", [61])
@pytest.mark.parametrize("kernel_size", [(2, 2), (3, 3), (4, 4)])
@pytest.mark.parametrize("original_shape", [(8, 8), (17, 15)], ids=["orig8x8", "orig17x15"])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("depthwise", [False])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("padding_mode", ["zeros"])
def test_conv2d_multi_op_fractured(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    kernel_size,
    original_shape,
    stride,
    padding,
    dilation,
    depthwise,
    bias,
    padding_mode,
):
    if test_kind != TestKind.INFERENCE:
        pytest.skip()

    padding = conv2d_padding_to_canonical(padding, kernel_size)
    outy, outx = calculate_conv2d_output_dimensions(original_shape[0], original_shape[1], kernel_size, stride, padding, dilation)
    if bias and (outy * outx) < 32:
        # TODO: Re-enable when bcast on < tile dim isn't broken
        pytest.skip()

    total_kernel_size_x = (kernel_size[0] - 1) * dilation + 1
    total_kernel_size_y = (kernel_size[1] - 1) * dilation + 1
    if total_kernel_size_x > (original_shape[0] + padding[2] + padding[3]) or total_kernel_size_y > (original_shape[1] + padding[0] + padding[1]):
        # Can't have the kernel be larger than the input itself
        pytest.skip()

    mod = Conv2dTModule(
        "conv2d_multi_op_fractured",
        in_channels,
        out_channels if not depthwise else in_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=in_channels if depthwise else 1,
        bias=bias,
    )

    # This makes the conv fracture into multiple ops
    pybuda.config.override_multi_op_fracture_factor("conv2d_multi_op_fractured.conv", kernel_size[0])

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"
    compiler_cfg.enable_conv_prestride = False

    os.environ["PYBUDA_FORCE_DISALLOW_FRACTURING"] = "1"  # Disables "within-op" fracturing

    devices = pybuda.verify.verify_module(mod, [(1, in_channels, original_shape[0], original_shape[1])],
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc))

    # Confirm conv has fractured into multiple ops
    assert len(devices[0]._compiled_graph_state.ordered_constant_node_names) == kernel_size[0] * 2, f"Expected {kernel_size[0] * 2} constant nodes (2 per each sparse matmul), got {len(devices[0]._compiled_graph_state.ordered_constant_node_names)}"
    assert len(devices[0]._compiled_graph_state.ordered_parameter_node_names), f"Expected {kernel_size[0]} parameter nodes (1 per each sparse matmul), got {len(devices[0]._compiled_graph_state.ordered_parameter_node_names)}"
    del os.environ["PYBUDA_FORCE_DISALLOW_FRACTURING"]


@pytest.mark.parametrize("in_channels", [3, 4, 6, 7])
@pytest.mark.parametrize("out_channels", [31])
@pytest.mark.parametrize("kernel_size", [(3, 3), (7, 7)])
@pytest.mark.parametrize("original_shape", [(8, 8), (18, 15), (33, 33), (10, 10), (42, 42)], ids=["orig8x8", "orig18x15", "orig33x33", "orig10x10", "orig42x42"])
@pytest.mark.parametrize("stride", [2, 3])
@pytest.mark.parametrize("padding", ["same"])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("depthwise", [False])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("padding_mode", ["zeros"])
def test_conv2d_prestrided(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    kernel_size,
    original_shape,
    stride,
    padding,
    dilation,
    depthwise,
    bias,
    padding_mode,
):
    if test_kind != TestKind.INFERENCE:
        pytest.skip()

    padding = conv2d_padding_to_canonical(padding, kernel_size)
    if not can_conv2d_prestride(
            (1, in_channels, *original_shape),
            (out_channels, in_channels, *kernel_size),
            (stride, stride),
            dilation,
            1, # dilation
            padding,
            False, # channel_last
            True, # graph_input
        ):
        pytest.skip()

    mod = Conv2dTModule(
        "conv2d_prestrided",
        in_channels,
        out_channels if not depthwise else in_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=in_channels if depthwise else 1,
        bias=bias,
        add_reshape_transpose_to_end=False,
    )

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"

    os.environ["PYBUDA_FORCE_DISALLOW_FRACTURING"] = "1"

    devices = pybuda.verify.verify_module(mod, [(1, in_channels, original_shape[0], original_shape[1])],
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc))

    # Confirm the conv was prestrided
    assert len(devices) == 1
    transforms = devices[0]._compiled_graph_state.ordered_input_runtime_tensor_transforms
    assert len(transforms) == 1
    assert transforms[0].type == RuntimeTensorTransformType.Prestride
    del os.environ["PYBUDA_FORCE_DISALLOW_FRACTURING"]


@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("out_channels", [64])
@pytest.mark.parametrize("kernel_size", [(3, 3)])
@pytest.mark.parametrize("original_shape", [(16, 16)], ids=["orig16x16"])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("depthwise", [False])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("padding_mode", ["zeros"])
# @pytest.mark.parametrize("recompute", [True, False])
def test_simple_convnet(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    kernel_size,
    original_shape,
    stride,
    padding,
    dilation,
    depthwise,
    bias,
    padding_mode,
):
    training = test_kind.is_training()

    padding = conv2d_padding_to_canonical(padding, kernel_size)
    outy, outx = calculate_conv2d_output_dimensions(original_shape[0], original_shape[1], kernel_size, stride, padding, dilation)
    if bias and (outy * outx) < 32:
        # TODO: Re-enable when bcast on < tile dim isn't broken
        pytest.skip()

    total_kernel_size_x = (kernel_size[0] - 1) * dilation + 1
    total_kernel_size_y = (kernel_size[1] - 1) * dilation + 1
    if total_kernel_size_x > (original_shape[0] + padding[2] + padding[3]) or total_kernel_size_y > (original_shape[1] + padding[0] + padding[1]):
        # Can't have the kernel be larger than the input itself
        pytest.skip()

    class ConvNet(PyBudaModule):
        def __init__(self, name, **kwargs):
            super().__init__(name)

            self.mod0 = Conv2dModule(
                name + "_layer0",
                **kwargs,
            )

            self.mod1 = Conv2dModule(
                name=name + "_layer1",
                in_channels=kwargs["out_channels"],
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=bias,
            )

        def forward(self, activations):
            return self.mod1(self.mod0(activations))

    mod = ConvNet(
        "convnet",
        in_channels=in_channels,
        out_channels=out_channels if not depthwise else in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=in_channels if depthwise else 1,
        bias=bias,
    )

    pybuda.config.set_configuration_options(enable_conv_prestride=False)
    relative_atol = 0.4 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.15
    pcc = 0.94 if test_device.devtype == BackendType.Silicon else 0.99
    pybuda.verify.verify_module(mod, [(1, in_channels, original_shape[0], original_shape[1])],
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc))


@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("kernel_size", [3, 5, (2, 5)])
@pytest.mark.parametrize("original_shape", [(16, 16), (5, 6), (33, 31)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("ceil_mode", [True, False])
def test_max_pool2d(
    test_kind,
    test_device,
    in_channels,
    kernel_size,
    original_shape,
    stride,
    dilation,
    ceil_mode,
    padding="same",
):
    training = test_kind.is_training()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_broadcast_splitting = True # tenstorrent/budabackend#694
    compiler_cfg.enable_conv_prestride = False

    if training:
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    df = DataFormat.Float16
    compiler_cfg.default_df_override = df
    compiler_cfg.default_accumulate_df = df

    mod = MaxPool2dModule(
        "max_pool2d",
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    s = (1, in_channels, original_shape[0], original_shape[1])
    t = torch.arange(s[2]*s[3], dtype=torch.float32)
    t = t.remainder(33)
    t = t.reshape((s[-2], s[-1]))
    t = t.repeat(1, s[-3], 1, 1)
    t[:, 1:, :, :] = 0.0
    t.requires_grad_(requires_grad=training)
    activations = Tensor.create_from_torch(t)
    padding = (kernel_size // 2 if isinstance(kernel_size,int) else (kernel_size[0] // 2, kernel_size[1] // 2 )) if padding == "same" else padding
    padding = conv2d_padding_to_canonical(padding, kernel_size)
    outy, outx = calculate_conv2d_output_dimensions(original_shape[0], original_shape[1], kernel_size, stride, padding, dilation, ceil_mode)
    def f(a, b):
        a = a.transpose(2, 3).reshape(in_channels, outy, outx)
        b = b.transpose(2, 3).reshape(in_channels, outy, outx)
        passed = True
        for z in range(a.shape[-3]):
            for i in range(a.shape[-2]):
                if not torch.allclose(a[z][i], b[z][i]):
                    passed = False
                    print(f"[{z}][{i}/{a.shape[-2]}]", a[z][i])
                    print(f"[{z}][{i}/{a.shape[-2]}]", b[z][i])
        return passed

    #
    # Note: PCC is relatively low for silicon because lower precision can actually cause different max values
    #       to win.  As an example, consider the values [0.997, 0.998]:
    #         - When doing this as fp32, both of those values are representable so index 1 wins max
    #         - When doing this as bfloat16, both values quantize to 0.9961 and first winner wins, i.e. index 0
    #
    #       Ideally we could set `input_params=[{"data_format": torch.bfloat16}]`, but pytorch doesn't implement
    #       maxpool for any of the half precision floating point types.
    #
    relative_atol = 0.9 if test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.90 if test_device.devtype == BackendType.Silicon else 0.99
    pybuda.verify.verify_module(mod, [(1, in_channels, original_shape[0], original_shape[1])],
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc, golden_compare_callback=f, fp32_fallback=df),
        inputs=[activations])


@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("kernel_size", [3, 5, (3,3)])
@pytest.mark.parametrize("original_shape", [(160, 160)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1])
def test_max_pool2d_dataflow(
    test_kind,
    test_device,
    in_channels,
    kernel_size,
    original_shape,
    stride,
    dilation,
):
    test_max_pool2d(test_kind, test_device, in_channels, kernel_size, original_shape, stride, dilation, ceil_mode=False)


def test_max_pool2d_stream_through_queue(test_device):
    in_channels = 32
    kernel_size = 3
    original_shape = (28, 28)
    stride = 1
    dilation = 1
    ceil_mode = False
    pybuda.config.override_t_stream_shape("max_pool2d.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", (5, 1))
    pybuda.config.override_t_stream_shape("max_pool2d.dc.reduce_max.6", (1, 1))
    test_max_pool2d(TestKind.INFERENCE, test_device, in_channels, kernel_size, original_shape, stride, dilation, ceil_mode)


@pytest.mark.parametrize("producer_stream_factor", [1, 2, 4])
def test_max_pool2d_stream_through_queue_1x1(test_device, producer_stream_factor):
    if not test_device.is_wormhole_b0():
        pytest.skip()
    in_channels = 64
    kernel_size = 2
    original_shape = (32, 32)
    stride = 2
    dilation = 1
    ceil_mode = False
    padding = [0, 0, 0, 0]
    os.environ["PYBUDA_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    pybuda.config.override_t_stream_shape("max_pool2d.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", (producer_stream_factor, 1))
    pybuda.config.override_t_stream_shape("max_pool2d.dc.reduce_max.6", (2, 1))
    test_max_pool2d(TestKind.INFERENCE, test_device, in_channels, kernel_size, original_shape, stride, dilation, ceil_mode, padding=padding)


@pytest.mark.parametrize("in_channels", [64])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("original_shape", [(31, 31), (33, 33), (6, 5)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("ceil_mode", [True, False])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("count_include_pad", [True, False])
@pytest.mark.parametrize("divisor_override", [None])  # TODO: add this
def test_avg_pool2d(
    test_kind,
    test_device,
    in_channels,
    kernel_size,
    original_shape,
    stride,
    ceil_mode,
    padding,
    count_include_pad,
    divisor_override
):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    compiler_cfg.enable_conv_prestride = False

    mod = AvgPool2dModule(
        "avg_pool2d",
        kernel_size,
        stride=stride,
        padding="same" if padding else 0,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.96 if test_device.devtype == BackendType.Silicon else 0.99
    pybuda.verify.verify_module(mod, [(1, in_channels, original_shape[0], original_shape[1])],
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc))


def test_conv2d_stream_data_mismatch(test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"
    return test_conv2d(
        TestKind.INFERENCE,
        test_device,
        32,
        32,
        (7, 7),
        (224, 224),
        2,
        "same",
        1,
        False,
        False,
        "zeros",
    )


def test_conv2d_stream_through_queue(test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"
    compiler_cfg.place_on_new_epoch("conv2d.dc.matmul.11")
    return test_conv2d(
        TestKind.INFERENCE,
        test_device,
        1920,
        640,
        (3, 3),
        (32, 32),
        1,
        "same",
        1,
        False,
        True,
        "zeros",
    )


def test_conv2d_vgg_head(test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "MaximizeTMinimizeGrid"
    pybuda.config.override_t_stream_shape("conv2d.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (28, 1))
    return test_conv2d(
        TestKind.INFERENCE,
        test_device,
        3,
        64,
        (3, 3),
        (224, 224),
        1,
        "same",
        1,
        False,
        False,
        "zeros",
    )


def test_conv2d_prestride_front_end():
    def round_up_mod(n, d):
        m = n % d
        return 0 if m == 0 else d - m

    pad = torch.torch.nn.functional.pad

    def test(original_shape, kernel_size, stride, padding):
        if (
            kernel_size[0] > padding[2] + padding[3] + original_shape[0] or
            kernel_size[1] > padding[0] + padding[1] + original_shape[1]
        ):
            # Kernel bigger than image, skipping test...
            print(f"Skipping test({original_shape}, kernel_size={kernel_size}, stride={stride}, padding={padding})")
            return True

        torch_activations = torch.randn(1, 1, original_shape[0], original_shape[1])
        torch_weights = torch.randn(1, 1, kernel_size[0], kernel_size[1])
        padded_torch_activations = pad(torch_activations, padding)
        torch_output = torch.nn.functional.conv2d(padded_torch_activations, torch_weights, stride=stride, padding=0)

        act_view = pad(torch_activations, (0, round_up_mod(torch_activations.shape[-1], stride[1]), 0, round_up_mod(torch_activations.shape[-2], stride[0])))
        ps_activations = []
        for y in range(stride[0]):
            for x in range(stride[1]):
                ps_activations.append(act_view[:, :, y::stride[0], x::stride[1]])

        ps_weights, ps_padding = calculate_conv2d_prestride_weights_and_padding(torch_weights, original_shape[0], original_shape[1], stride, padding)
        ps_activations = torch.cat(ps_activations, dim=-3)
        padded_ps_activations = pad(ps_activations, ps_padding)
        ps_output = torch.nn.functional.conv2d(padded_ps_activations, ps_weights, stride=1, padding=0)
        try:
            assert torch.allclose(torch_output, ps_output, atol=1e-03)
        except:
            raise RuntimeError(f"test({original_shape}, kernel_size={kernel_size}, stride={stride}, padding={padding})\n{torch_output}\n{ps_output}\nFAILURE")

    # Local seed for rng
    rng = random.Random(4)

    # 20k random tests
    for _ in range(20000):
        y   = rng.randint(8, 17)
        x   = rng.randint(8, 17)
        k_h = rng.randint(2, 8)
        k_w = rng.randint(2, 8)
        s_h = rng.randint(2, 6)
        s_w = rng.randint(2, 6)
        p_l = rng.randint(0, 6)
        p_r = rng.randint(0, 6)
        p_t = rng.randint(0, 6)
        p_b = rng.randint(0, 6)

        test(
            original_shape=(y, x),
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_l, p_r, p_t, p_b)
        )

