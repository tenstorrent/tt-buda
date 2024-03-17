# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import os
from typing import OrderedDict
from pybuda.tvm_to_python import compile_tvm_to_python
from pybuda.verify.config import TestKind
import pytest
from sqlalchemy import true
from test.tvm.python.test_sanity import test_linear
import pybuda

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pybuda
from pybuda import (
    PyBudaModule,
    Tensor,
    PyTorchModule,
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    CompileDepth,
)
from pybuda.config import _get_global_compiler_config
from pybuda.verify.backend import verify_module
from pybuda.op.eval import compare_tensor_to_golden
from test.tvm.utils import evaluate_framework_vs_pybuda

input_shapes = [(1, 1, 8, 64)]
linear_features_in = [64]
linear_features_out = [64]


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "lin_in", linear_features_in, ids=[f"lin_in({str(l)})" for l in linear_features_in]
)
@pytest.mark.parametrize(
    "lin_out",
    linear_features_out,
    ids=[f"lin_out({str(l)})" for l in linear_features_out],
)
def test_tvm_linear(test_kind, test_device, input_shape, lin_in, lin_out):
    if test_kind.is_training():
        pytest.skip()
    import os
    os.environ["PYBUDA_ENABLE_TINY_TILE"] = "1"
    _get_global_compiler_config().compile_depth = CompileDepth.GENERATE_NETLIST
    class DoubleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(lin_in, lin_out, bias=False)

        def forward(self, x1,):
            m1 = self.l1(x1)

            return m1

    model = DoubleLinear()
    mod = PyTorchModule("tiny_tile_linear", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


input_shapes = [(1, 64, 64)]



@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "lin_in", linear_features_in, ids=[f"lin_in({str(l)})" for l in linear_features_in]
)
@pytest.mark.parametrize(
    "lin_out",
    linear_features_out,
    ids=[f"lin_out({str(l)})" for l in linear_features_out],
)
def test_tvm_softmax(test_kind, test_device, input_shape, lin_in, lin_out):
    class LinearSoftmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(lin_in, lin_out, bias=False)

        def forward(self, x1):
            m1 = self.l1(x1)
            out = nn.Softmax(dim=-1)(m1)

            return out + out

    model = LinearSoftmax()
    mod = PyTorchModule("softmax", model)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


input_shapes = [(1, 32, 64, 32)]
tensor_permutations = [(0, 3, 1, 2)]


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "tensor_permutation",
    tensor_permutations,
    ids=[f"tp{str(t)}" for t in tensor_permutations],
)
def test_tvm_transpose(test_kind, test_device, input_shape, tensor_permutation):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    class MultiAxisTranspose(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            x2 = torch.permute(x1, tensor_permutation)

            return x2

    model = MultiAxisTranspose()
    mod = PyTorchModule("MultiAxisTranspose", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
   

input_shapes_first = [(1, 32, 17, 32)]
input_shapes_second = [(1, 32, 12, 32)]


@pytest.mark.parametrize(
    "input_shape_first",
    input_shapes_first,
    ids=[f"input{str(s)}" for s in input_shapes_first],
)
@pytest.mark.parametrize(
    "input_shape_second",
    input_shapes_second,
    ids=[f"input{str(s)}" for s in input_shapes_second],
)
def test_tvm_concat(test_kind, test_device, input_shape_first, input_shape_second):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    class Concat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            x1 = torch.add(x1, x1)
            x2 = torch.add(x2, x2)
            return torch.cat([x1, x2], axis=2)

    model = Concat()
    mod = PyTorchModule("concat", model)
    verify_module(
        mod,
        (input_shape_first, input_shape_second),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

# TODO: Make all einsum tests use the verify_module flow once the microbatch issue is fixed
einsum_inputs = [
    ((16, 16, 16, 16), (1, 16, 16, 64), "bnij,jbnd->ibnd"),
    ((1, 16, 32,), (1, 16, 64), "bct,bcs->bts"),
    ((1, 16, 32,), (1, 128, 32), "bts,bcs->bct"),
    ((1, 16, 32,64), (1, 16, 128,64), "bnqd,bnkd->bnqk"),
    ((1, 16, 32), (32, 128, 64), "ibh,hnd->ibnd"),
    ((1, 16, 32, 64), (2, 16, 32, 64), "ibnd,jbnd->bnij"),
    ((1, 16, 32, 64), (2, 32, 64), "ibnd,snd->ibns"),
    ((1, 16, 32, 64), (1, 32, 128, 64), "ijbs,ibns->bnij"),
    ((1, 16, 32, 64), (64, 1, 16, 128), "bnij,jbnd->ibnd"),
    ((1, 16, 32, 64), (128, 32, 64), "ibnd,hnd->ibh"),
    ((1, 1, 32, 64), (2, 16, 32, 64), "ibnd,jbnd->bnij"),
    ((2, 32, 64), (2, 128, 64), "bhd,bmd->bhmd"),
    ((784, 128), (32, 16), "ij,qr->i"),
    ((16, 32, 128), (128, 64), "ijk,kr->ijr"),
    ((128, 32, 64), (16, 9), "ijk,qr->ijr"),
    ((16, 1, 16, 64), (32, 1, 16, 64), "ibnd,jbnd->ijbn"),
    ((16, 16, 1, 16), (16, 1, 16, 64), "ijbn,jbnd->ibnd"),
    ((1, 128, 2, 64), (1, 128, 2, 64), "jbki,jfki->jkbf"),
    ((2,), (1, 2, 128, 128), "f,bfde->fbde"),
    ((1, 128, 2, 64), (1, 2, 128, 128), "jikd,jkgi->jkdg"),
    ((1, 32, 128), (1, 64, 128), "b i d, b j d -> b i j"),
    ((1, 32, 64), (1, 64, 128), "b i j, b j d -> b i d"),
]

@pytest.mark.parametrize(
    "einsum_inputs",
    einsum_inputs,
    ids=[f"input{str(s)}" for s in einsum_inputs],
)
def test_tvm_einsum(test_kind, test_device, einsum_inputs):

    # Set of equations whose decomps result in unsupported hw ops
    makes_unsupported_hw_ops = {
        "ijbs,ibns->bnij",
        "jbki,jfki->jkbf",
        "jikd,jkgi->jkdg",
        "ibnd,hnd->ibh",
        "bnij,jbnd->ibnd",
        "ibnd,jbnd->bnij",
        "ibnd,snd->ibns",
        "ibnd,jbnd->bnij",
        "bhd,bmd->bhmd",
        "ibnd,jbnd->ijbn",
        "ijbn,jbnd->ibnd",
        "f,bfde->fbde",
    }

    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    input_shape_first, input_shape_second, equation = einsum_inputs
    class Einsum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            return torch.einsum(equation, [x1, x2])

    model = Einsum()
    mod = PyTorchModule("Einsum", model)

    use_verify_module = input_shape_first[0] == input_shape_second[0] == 1 # Ensure microbatch of 1
    if use_verify_module:
        if einsum_inputs[2] in makes_unsupported_hw_ops:
            _get_global_compiler_config().compile_depth = CompileDepth.PRE_LOWERING_PASS

        verify_module(
            mod,
            (input_shape_first, input_shape_second),
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                test_kind=test_kind,
            )
        )
    else:
        if einsum_inputs[2] in makes_unsupported_hw_ops:
            compile_depth = CompileDepth.PRE_LOWERING_PASS
        else:
            compile_depth = CompileDepth.FULL
        sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
        tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
        tt0.place_module(mod)

        #Fusing disabled due to tenstorrent/pybuda#789
        acts = [torch.rand(*input_shape_first), torch.rand(*input_shape_second)]
        ret = pybuda_compile(
            tt0,
            "Einsum",
            *acts,
            compiler_cfg=CompilerConfig(
                enable_training=test_kind.is_training(),
                enable_recompute=test_kind.is_recompute(),
                compile_depth=compile_depth,
                enable_auto_fusing=False
            ),
            verify_cfg=VerifyConfig(
                intermediates=True,
            ),
        )

        pytorch_out = model(*acts)
        assert compare_tensor_to_golden("output", ret.golden_outputs[0], pytorch_out, is_buda=True, verify_cfg=VerifyConfig())


input_shapes = [(1, 32, 32, 32)]
conv_in_chs = [32]
conv_out_chs = [32]
conv_kernel_sizes = [5]
conv_paddings = [2]
groups = [1, 2, 4, 32]
pool_kernel_sizes = [3]
pool_paddings = [1]


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])

@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "conv_in_ch", conv_in_chs, ids=[f"cinch({str(c)})" for c in conv_in_chs]
)
@pytest.mark.parametrize(
    "conv_out_ch", conv_out_chs, ids=[f"coutch({str(c)})" for c in conv_out_chs]
)
@pytest.mark.parametrize(
    "conv_kernel_size",
    conv_kernel_sizes,
    ids=[f"cks{str(c)}" for c in conv_kernel_sizes],
)
@pytest.mark.parametrize(
    "conv_padding", conv_paddings, ids=[f"cpad({str(c)})" for c in conv_paddings]
)
@pytest.mark.parametrize(
    "conv_groups", groups, ids=[f"cgroup({str(c)})" for c in groups]
)
@pytest.mark.parametrize(
    "pool_kernel_size",
    pool_kernel_sizes,
    ids=[f"pks{str(p)}" for p in pool_kernel_sizes],
)
@pytest.mark.parametrize(
    "pool_padding", pool_paddings, ids=[f"ppad({str(p)})" for p in pool_paddings]
)
def test_tvm_conv(
    training,
    recompute,
    input_shape,
    conv_in_ch,
    conv_out_ch,
    conv_kernel_size,
    conv_padding,
    conv_groups,
    pool_kernel_size,
    pool_padding,
):
    if training:
        pytest.xfail()  # Backward is currently unsupported

    class CONV(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
                conv_in_ch,
                conv_out_ch,
                kernel_size=conv_kernel_size,
                padding=conv_padding,
                groups=conv_groups,
            )

        def forward(self, a):
            b = self.conv1(a)
            c = F.max_pool2d(
                b,
                pool_kernel_size,
                padding=pool_padding,
            )
            d = F.relu(c)

            return d

    model = CONV()
    mod = PyTorchModule("conv", model)
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = torch.rand(*input_shape)

    ret = pybuda_compile(
        tt0,
        "conv",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    pytorch_out = model(act1)
    assert compare_tensor_to_golden("output", ret.golden_outputs[0], pytorch_out, is_buda=True, verify_cfg=VerifyConfig())


input_shapes = [(1, 512, 10, 10), (1, 512, 20, 10), (1, 512, 10, 30)]
pool_out_sizes = [(1, 1)]


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "pool_out_size", pool_out_sizes, ids=[f"pouts({str(p)})" for p in pool_out_sizes]
)
def test_tvm_avg_pool(test_kind, test_device, input_shape, pool_out_size):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    class AdaptiveAvgPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(pool_out_size)

        def forward(self, a):
            b = self.avgpool(a)
            c = torch.flatten(b, 1)

            return c

    model = AdaptiveAvgPool()
    mod = PyTorchModule("AdaptiveAvgPool", model)

    verify_module(
            mod,
            (input_shape,),
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                test_kind=test_kind,
            )
    )
    

input_shapes = [(1, 512, 10, 10)]


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
def test_tvm_sigmoid(test_kind, test_device, input_shape):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, a):
            b = self.linear(a) + 7
            b = torch.sigmoid(b)

            return b

    model = Sigmoid()
    mod = PyTorchModule("Sigmoid", model)
    input_tensor = torch.rand(input_shape) * 10
    verify_module(
            mod,
            (input_shape,),
            inputs=[(input_tensor,),],
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                test_kind=test_kind,
            )
    )


input_shapes = [(1, 128, 10, 10), (1, 16, 34, 60)]
scale_factors = [2, 3]
upsample_modes = ["nearest", "bilinear"]


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "scale_factors", scale_factors, ids=[f"sfactor({str(s)})" for s in scale_factors]
)
@pytest.mark.parametrize(
    "upsample_mode", upsample_modes, ids=[f"umode({str(u)})" for u in upsample_modes]
)
@pytest.mark.parametrize("align_corners", (True, False), ids=["align", "no_align"])
def test_tvm_upsample2d(test_kind, test_device, input_shape, scale_factors, upsample_mode, align_corners):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    if align_corners and upsample_mode != "bilinear":
        pytest.skip()

    class Upsample2d(nn.Module):
        def __init__(self, scale_factors, upsample_mode, align_corners):
            super().__init__()
            if upsample_mode == "nearest":
                self.resize = torch.nn.Upsample(
                    scale_factor=scale_factors,
                    mode=upsample_mode,
                )
            else:
                self.resize = torch.nn.Upsample(
                    scale_factor=scale_factors,
                    mode=upsample_mode,
                    align_corners=align_corners,
                )
        def forward(self, a):
            b = self.resize(a)

            return b

    model = Upsample2d(scale_factors, upsample_mode, align_corners)
    mod = PyTorchModule("Upsample2d", model)
    verify_module(
            mod,
            (input_shape,),
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                test_kind=test_kind,
            )
    )


input_shapes = [(1, 64, 64)]
linear_features_out = [192]


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "lin_in", linear_features_in, ids=[f"lin_in({str(l)})" for l in linear_features_in]
)
@pytest.mark.parametrize(
    "lin_out",
    linear_features_out,
    ids=[f"lin_out({str(l)})" for l in linear_features_out],
)
def test_tvm_lift_linear_split(test_kind, test_device, input_shape, lin_in, lin_out):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    class LinearLift(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(lin_in, lin_out, bias=True)

        def forward(self, a):
            b = self.l1(a)
            c = b.split(lin_in, 2)
            d = c[0] + c[1]
            e = d - c[2]

            return e

    model = LinearLift()
    mod = PyTorchModule("linear", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
   

input_shapes = [(1, 64, 32, 32)]
conv_in_chs = [32 * 2]
conv_out_chs = [32 * 8]
conv_kernel_sizes = [2]
conv_strides = [2]


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "conv_in_ch", conv_in_chs, ids=[f"cinch({str(c)})" for c in conv_in_chs]
)
@pytest.mark.parametrize(
    "conv_out_ch", conv_out_chs, ids=[f"coutch({str(c)})" for c in conv_out_chs]
)
@pytest.mark.parametrize(
    "conv_kernel_size",
    conv_kernel_sizes,
    ids=[f"cks{str(c)}" for c in conv_kernel_sizes],
)
@pytest.mark.parametrize(
    "conv_stride", conv_strides, ids=[f"cstride({str(c)})" for c in conv_strides]
)
def test_tvm_transpose_conv(
    training,
    recompute,
    input_shape,
    conv_in_ch,
    conv_out_ch,
    conv_kernel_size,
    conv_stride,
):
    if training:
        pytest.xfail()  # Backward is currently unsupported

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class TransposeConv(nn.Module):
        def __init__(self):
            super().__init__()

            features = 32
            self.upconv4 = nn.ConvTranspose2d(
                conv_in_ch,
                conv_out_ch,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
            )

        def forward(self, x):
            dec4 = self.upconv4(x)

            return dec4

    model = TransposeConv()
    mod = PyTorchModule("unet", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act1 = torch.rand(*input_shape)

    ret = pybuda_compile(
        tt0,
        "conv2d_transpose",
        act1,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.PRE_LOWERING_PASS
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    pytorch_out = model(act1)
    assert compare_tensor_to_golden("output", ret.golden_outputs[0], pytorch_out, is_buda=True, verify_cfg=VerifyConfig())

channels = [9]
input_shapes = [9, 32]
output_shapes = [1, 3, 9]


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
@pytest.mark.parametrize(
    "input_channel", channels, ids=[f"inputch{c}" for c in channels]
)
@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"inputsh{s}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "output_shape", output_shapes, ids=[f"outputch{s}" for s in output_shapes]
)
def test_tvm_adaptive_avg_pool(
    training, recompute, input_channel, input_shape, output_shape
):
    if training:
        pytest.skip()  # Backward is currently unsupported

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    if input_shape % output_shape != 0:
        pytest.xfail()  # Potential bug, tensor mismatch issue on placer

    class AdaptiveAvgPool(nn.Module):
        def __init__(self, out_size):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(out_size)

        def forward(self, a):
            b = self.avgpool(a)
            c = torch.flatten(b, 1)
            return c

    pytorch_model = AdaptiveAvgPool(out_size=output_shape)
    module = PyTorchModule("adaptive_average_pool", pytorch_model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    hidden_states = torch.rand((1, input_channel, input_shape, input_shape))

    res = pybuda_compile(
        tt0,
        "adaptive_average_pool",
        hidden_states,
        compiler_cfg=CompilerConfig(
            enable_training=training, 
            enable_recompute=recompute,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    pytorch_golden_out = pytorch_model(hidden_states)
    assert compare_tensor_to_golden("output", res.golden_outputs[0], pytorch_golden_out, is_buda=True, verify_cfg=VerifyConfig())

@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
@pytest.mark.parametrize(
    "input_channel", channels, ids=[f"inputch{c}" for c in channels]
)
@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"inputsh{s}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "output_shape", output_shapes, ids=[f"outputch{s}" for s in output_shapes]
)
@pytest.mark.skip("Skip for golden wormhole_b0")
def test_tvm_adaptive_max_pool(
    training, recompute, input_channel, input_shape, output_shape
):
    if training:
        pytest.xfail()  # Backward is currently unsupported

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    if input_shape % output_shape != 0:
        pytest.xfail()  # Potential bug, tensor mismatch issue on placer

    class AdaptiveMaxPool(nn.Module):
        def __init__(self, out_size):
            super().__init__()
            self.maxpool = nn.AdaptiveMaxPool2d(out_size)

        def forward(self, a):
            b = self.maxpool(a)
            c = torch.flatten(b, 1)
            return c

    pytorch_model = AdaptiveMaxPool(out_size=output_shape)
    module = PyTorchModule("adaptive_max_pool", pytorch_model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    hidden_states = torch.rand((1, input_channel, input_shape, input_shape))

    res = pybuda_compile(
        tt0,
        "adaptive_max_pool",
        hidden_states,
        compiler_cfg=CompilerConfig(
            enable_training=training, enable_recompute=recompute
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    pytorch_golden_out = pytorch_model(hidden_states)
    assert compare_tensor_to_golden("output", res.golden_outputs[0], pytorch_golden_out[0], is_buda=True, verify_cfg=VerifyConfig())


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
@pytest.mark.parametrize("in_channel", [1, 3])
@pytest.mark.parametrize("out_channel", [3, 9])
@pytest.mark.parametrize("feature_size", [128, 512])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_tvm_conv1d(training, recompute, in_channel, out_channel, feature_size, bias, kernel_size, padding):
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference


    class Conv1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(
                in_channel, 
                out_channel, 
                kernel_size=kernel_size, 
                bias=bias,
                padding=padding,
            )
    
        def forward(self, x):
            return self.conv(x)


    framework_model = Conv1D()
    mod = PyTorchModule("Conv1D", framework_model)
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    input_shape = (1, in_channel, feature_size)
    inputs = [
        torch.rand(*input_shape)
    ]

    pybuda_model_results = pybuda_compile(
        tt0,
        "Conv1D",
        *inputs,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    evaluate_framework_vs_pybuda(framework_model, pybuda_model_results, *inputs)


def test_tvm_clip(test_kind, test_device):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    #Clip should have been decomposed to something else error in Generate Netlist. Attributes still present
    _get_global_compiler_config().compile_depth = CompileDepth.BALANCER_PASS
    class clip(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            b = torch.clip(a, 0, 0.1)
            return b

    pytorch_model = clip()
    module = PyTorchModule("clip", pytorch_model)
    verify_module(
            module,
            ((1, 32, 32, 32),),
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                test_kind=test_kind,
            )
    )

@pytest.mark.parametrize("ceil_mode", (True, False), ids=["ceil_mode_true", "ceil_mode_false"])
@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_tvm_avgpool2d(training, ceil_mode, recompute):
    if training:
        pytest.xfail()  # Backward is currently unsupported

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class avgpool2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.avgpool = nn.AvgPool2d(3, stride=2, padding=1, ceil_mode=ceil_mode)

        def forward(self, a):
            b = self.avgpool(a)
            return b

    pytorch_model = avgpool2d()
    module = PyTorchModule("avgpool2d", pytorch_model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    hidden_states = torch.rand((1, 32, 12, 12))

    res = pybuda_compile(
        tt0,
        "avgpool2d",
        hidden_states,
        compiler_cfg=CompilerConfig(
            enable_training=training, enable_recompute=recompute, compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    pytorch_golden_out = pytorch_model(hidden_states)
    assert compare_tensor_to_golden("output", pytorch_golden_out[0], res.golden_outputs[0], is_buda=True, verify_cfg=VerifyConfig())


@pytest.mark.parametrize("ceil_mode", (True, False), ids=["ceil_mode_true", "ceil_mode_false"])
@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_tvm_maxpool2d(training, ceil_mode, recompute):
    if training:
        pytest.xfail()  # Backward is currently unsupported

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class avgpool2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.MaxPool2d = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=ceil_mode)

        def forward(self, a):
            b = self.MaxPool2d(a)
            return b

    pytorch_model = avgpool2d()
    module = PyTorchModule("MaxPool2d", pytorch_model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    hidden_states = torch.rand((1, 32, 12, 12))

    res = pybuda_compile(
        tt0,
        "MaxPool2d",
        hidden_states,
        compiler_cfg=CompilerConfig(
            enable_training=training, enable_recompute=recompute, compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    pytorch_golden_out = pytorch_model(hidden_states)
    assert compare_tensor_to_golden("output", pytorch_golden_out[0], res.golden_outputs[0], is_buda=True, verify_cfg=VerifyConfig())

def test_tvm_abs(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    class Abs(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            return torch.abs(x1)
    
    model = Abs()
    mod = PyTorchModule("abs", model)

    verify_module(
        mod,
        ((1, 10, 10),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


def test_tvm_cos(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    class Cos(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x1):
            return torch.cos(x1)

    model = Cos()
    mod = PyTorchModule("cos", model)

    verify_module(
        mod,
        ((1, 10, 10),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


def test_tvm_sin(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    class Sin(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x1):
            return torch.sin(x1)

    model = Sin()
    mod = PyTorchModule("sin", model)

    verify_module(
        mod,
        ((1, 10, 10),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


input_shape_to_pad = ([3, 6], [32, 32], [1, 32, 32, 32], [1, 5, 10, 10], [1, 32, 3, 32],)
paddings = ((1, 2, 3, 4), (4, 3, 2, 1), (7, 8))
@pytest.mark.parametrize(
    "input_shape_to_pad", input_shape_to_pad, ids=[f"inputsh{s}" for s in input_shape_to_pad]
)
@pytest.mark.parametrize(
    "padding", paddings, ids=[f"padding{s}" for s in paddings]
)
def test_tvm_pad(test_kind, test_device, input_shape_to_pad, padding):
    if test_kind.is_training():
        pytest.xfail()  # Backward is currently unsupported

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    class PAD(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            b = torch.nn.functional.pad(a, padding, mode='constant', value=0.0)
            return b

    pytorch_model = PAD()
    module = PyTorchModule("PAD", pytorch_model)

    verify_module(
        module,
        (input_shape_to_pad,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
    

# shape, index 
args = ([[1 ,4, 16], [0, 2, 4]], [[1, 3, 32], [0, 0, 3]], [[1, 32, 32], [0, 3, 31]], )
@pytest.mark.parametrize(
    "args", args, ids=[f"shape: {s[0]}, index: {s[1]}" for s in args])
def test_tvm_take(test_kind, test_device, args):
    input_shape, indices = args
    if test_kind.is_training():
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    class Indexing(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            x1 = torch.sin(x1)
            return x1[:, :, indices[2]]

    model = Indexing()
    mod = PyTorchModule("indexing", model)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


#dim = None does not work, bug in TVM
@pytest.mark.parametrize("dim", [1, 2, -1],) 
def test_tvm_argmax(test_kind, test_device, dim):

    if test_kind.is_training():
        pytest.skip()

    if dim == 0 or dim == 1:
        _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    class Argmax(nn.Module):
        def __init__(self):
            super().__init__()
            

        def forward(self, x1, ):
            if dim is not None:
                return x1.argmax(dim=dim, keepdims=True)
            return torch.argmax(x1)

    model = Argmax()
    mod = PyTorchModule("argmax", model)

    verify_module(
        mod,
        ((1, 16, 60, 30),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


@pytest.mark.parametrize("dim", [1, 2, 3, -1],)
def test_tvm_sum(test_kind, test_device, dim):
    # Sum across batch dim does not work
    # dim = 1 Creates unsupported HW op: matmul(accumulate: 1)
    if dim == 1:
        _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    class Sum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sum(x, dim=dim, keepdim=True)

    model = Sum()
    mod = PyTorchModule("sum", model)
    
    verify_module(
        mod,
        ((16, 4, 32, 9),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_multiple_outputs(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip("Verify Module not equipped to handle multiple outputs backwards")

    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(32, 64, bias=True)
            self.l2 = nn.Linear(64, 32, bias=True)
            self.l3 = nn.Linear(32, 128, bias=True)

        def forward(self, a):
            b = self.l1(a)
            c = self.l2(b)
            d = self.l3(c)

            return b, c, d

    model = SimpleLinear()
    mod = PyTorchModule("multiple_outputs", model)
    input_shape = (1, 1, 32, 32)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_broadcast(test_kind, test_device):

    if test_kind == TestKind.TRAINING:
        pytest.skip()

    # if test_kind.is_training():
    #     _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS

    class Broadcast(nn.Module):
        def __init__(self):
            super().__init__()
            

        def forward(self, x1, x2):
            tmp = x1.expand_as(x2)
            return tmp + x2

    
    model = Broadcast()
    act1 = torch.rand((1, 1, 1, 1))
    act2 = torch.rand((1, 16, 32, 64))

    mod = PyTorchModule("Broadcast", model)
    verify_module(
        mod,
        ((1, 1, 1, 1), (1, 16, 32, 64)),
        input_params=[{"requires_grad": False}, {}],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            graph_name="broadcast_vm",
        )
    )

 

def test_multiout(test_kind, test_device):
    # if not training and recompute:
    #     pytest.skip()

    class MultiOut(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            x1 = x1 + 7
            x2 = x2 * 9
            return x1, x2

    
    model = MultiOut()

    mod = PyTorchModule("MultiOut", model)
    verify_module(
        mod,
        ((1,64,), (1,64,)),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_embedding(training, recompute):

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class EmbModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_bag = nn.Embedding(3, 5)

        def forward(self, input):
            return self.emb_bag(input)


    model = EmbModel()
    mod = PyTorchModule("embedding", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    input = torch.tensor([1, 2], dtype=torch.int32)

    inps = [input]
    compiler_cfg = CompilerConfig(enable_training=False, enable_tvm_unsupported_ops=True, enable_tvm_cpu_fallback=False)

    test_name = "embedding"
    writers, _ = compile_tvm_to_python(mod, test_name, inps, compiler_cfg=compiler_cfg)
    
    

@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_embedding_bag(mode):
    _get_global_compiler_config().enable_tvm_cpu_fallback = False
    model = nn.EmbeddingBag(10, 3, mode=mode)
    mod = PyTorchModule("embedding_bag", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    input = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8 ,9], dtype=torch.int32)
    offsets = torch.tensor([0], dtype=torch.long)

    inps = [input, offsets]

    test_name = "embedding_bag"
    writers, _ = compile_tvm_to_python(mod, test_name, inps)
    
def test_passthrough(test_kind, test_device):
    class PassThrough(nn.Module):
        def forward(self, x1, x2):
            return x1 + 7, x2

    model = PassThrough()

    mod = PyTorchModule("PassThrough", model)
    verify_module(
        mod,
        ((1,64,), (1,64,)),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
def test_tvm_new_empty(training, recompute):
    pytest.skip() # Due to the randomness of empty tensors (and corresponding memory locations), 
                  # this test will be skipped as it can fail from time to time. 

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class NewEmpty(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(3, 3, bias=True)

        def forward(self, reference_tensor):
            linear_transformation = self.l1(reference_tensor)

            shape = tuple([int(x) for x in reference_tensor.shape])
            zero_tensor = reference_tensor.new_empty(shape)

            return linear_transformation + zero_tensor

    framework_model = NewEmpty()
    mod = PyTorchModule("NewEmpty", framework_model)
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    reference_input = (1, 1, 3, 3)
    reference_input = torch.rand(reference_input)

    pybuda_model_results = pybuda_compile(
        tt0,
        "NewEmpty",
        reference_input,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )


def test_tvm_new_ones(test_kind, test_device):

    class NewOnes(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(3, 3, bias=True)

        def forward(self, reference_tensor):
            linear_transformation = self.l1(reference_tensor)

            shape = tuple([int(x) for x in reference_tensor.shape])
            ones_tensor = reference_tensor.new_ones(shape)

            return linear_transformation + ones_tensor

    framework_model = NewOnes()
    mod = PyTorchModule("NewOnes", framework_model)
    verify_module(
        mod,
        ((1, 1, 3, 3),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_new_zeros(test_kind, test_device):
    class NewZeros(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(3, 3, bias=True)

        def forward(self, reference_tensor):
            linear_transformation = self.l1(reference_tensor)

            shape = tuple([int(x) for x in reference_tensor.shape])
            zero_tensor = reference_tensor.new_zeros(shape)

            return linear_transformation + zero_tensor

    framework_model = NewZeros()
    mod = PyTorchModule("NewZeros", framework_model)
    verify_module(
        mod,
        ((1, 1, 3, 3),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize("input_shape", [(1, 1, 5, 5), (1, 2, 5, 5), (2, 2, 5, 5)])
def test_tvm_tril(test_kind, test_device, input_shape):
    class Tril(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_tensor):
            return torch.tril(input_tensor)

    framework_model = Tril()
    mod = PyTorchModule("Tril", framework_model)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize("input_shape", [(1, 1, 5, 5), (1, 2, 5, 5), (1, 3, 5, 5)])
@pytest.mark.parametrize("size", [(1, 1, 2, 2), (1, 2, 3, 3), (2, 2, 2, 2), (2, 2, 3, 3)])
@pytest.mark.parametrize("stride", [(1, 1, 2, 2), (2, 2, 2, 2)])
def test_tvm_as_strided(test_kind, test_device, input_shape, size, stride):
    
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.PRE_LOWERING_PASS # Unsupported HW ops (reshape, sparse_matmul, concatenate)

    class AsStrided(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_tensor):
            return torch.as_strided(input_tensor, size, stride)

    framework_model = AsStrided()
    mod = PyTorchModule("AsStrided", framework_model)
    verify_module(
        mod,
        (input_shape,),
        input_params=[{"requires_grad": False}],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_nonzero():
    pytest.skip() # Pausing with nonzero support which is required by Longformer.
    # More details in Longformer test (test_longformer.py)

    training = False
    recompute = False
    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class Nonzero(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, reference_tensor):
            return reference_tensor.nonzero(as_tuple=True)

    framework_model = Nonzero()
    mod = PyTorchModule("Nonzero", framework_model)
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    reference_input_shape = (1, 1, 3, 3)
    reference_input = torch.rand(reference_input_shape)

    pybuda_model_results = pybuda_compile(
        tt0,
        "Nonzero",
        reference_input,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    evaluate_framework_vs_pybuda(framework_model, pybuda_model_results, reference_input)


@pytest.mark.parametrize("dim", [0, 1])
def test_tvm_weight_norm_linear(test_kind, test_device, dim):
    if test_kind == TestKind.TRAINING:
        pytest.skip()
    
    if test_kind.is_training():
        # Backend error: Unsupported tt_op_info::gradient_op set for op bw_in0_linear.weight_v_combine_add_1, type add
        test_device.devtype = BackendType.NoBackend
    class WeightNormModuleLinear(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear = torch.nn.utils.weight_norm(nn.Linear(3, 9, bias=True), name="weight", dim=dim)

        def forward(self, input_tensor):
            return self.linear(input_tensor)

    framework_model = WeightNormModuleLinear()
    mod = PyTorchModule("WeightNormModuleLinear", framework_model)
    input_shape = (1, 1, 1, 3)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_model_param_dtype():
    pytest.skip() # Skipping this one for now. Will require custom solution for handling
    # bfloat16 explicit cast. For example specific TVM op, customized PyTorch TVM frontend
    # solution, etc.
    training = False
    recompute = False

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class ModelDtype(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear = nn.Linear(3, 9, bias=True).bfloat16()

        def forward(self, x):
            x = x.type(torch.bfloat16)

            return self.linear(x)

    framework_model = ModelDtype()
    mod = PyTorchModule("ModelDtype", framework_model)
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    input_shape = (1, 1, 3, 3)

    input =  torch.randint(0, 10, input_shape, dtype=torch.bfloat16)
    inputs = [
        input
    ]

    pybuda_model_results = pybuda_compile(
        tt0,
        "ModelDtype",
        *inputs,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )

    evaluate_framework_vs_pybuda(framework_model, pybuda_model_results, *inputs)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_tvm_torch_np_concat(test_kind, test_device, axis):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    if axis == 0 or test_kind.is_training():
        _get_global_compiler_config().compile_depth = CompileDepth.PRE_LOWERING_PASS

    class TorchNumpyConcat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            a = torch.cat((x, x), dim=axis)
            b = torch.from_numpy(np.concatenate((x, x), axis=axis))
            return a + b

    framework_model = TorchNumpyConcat()
    mod = PyTorchModule("TorchNumpyConcat", framework_model)
    input_shape = (1, 1, 1, 3)
    verify_module(
        mod,
        (input_shape,),
        input_params=[{"requires_grad": False}],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


shapes = [((1, 144, 32, 32), (1, 12, 384, 32)), ((1, 32, 1, 32), (1, 1, 32, 32))]
@pytest.mark.parametrize("shapes", shapes)
def test_tvm_vstack(test_kind, test_device, shapes):
    input_shape, output_shape = shapes
    
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    class Vstack(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = torch.nn.Softmax()
            
        def forward(self, x):
            x = x.reshape(output_shape)
            return self.g(x)

    framework_model = Vstack()
    mod = PyTorchModule("Vstack", framework_model)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


shapes = [((1, 144, 32, 32), (1, 12, 384, 32)), ((1, 2, 1, 1), (1, 1, 2, 1))]

@pytest.mark.parametrize("shapes", shapes)
def test_tvm_vslice(test_kind, test_device, shapes):
    input_shape, output_shape = shapes
    
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    class Vslice(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = torch.nn.Softmax()
            
        def forward(self, x):
            x = x.reshape(output_shape)
            return self.g(x)

    framework_model = Vslice()
    mod = PyTorchModule("Vslice", framework_model)
    
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_leaky_relu(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER  # Unsupported HW ops
        
    class LeakyRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(9, 9)
            self.leaky_relu = nn.LeakyReLU(0.2)
            
        def forward(self, x):
            x = self.linear(x)

            return self.leaky_relu(x)

    framework_model = LeakyRelu()
    mod = PyTorchModule("LeakyRelu", framework_model)
    input_shape = (1, 3, 9, 9)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize("padding", (0, 1, 2))
def test_tvm_max_pool_2d_to_1d(test_kind, test_device, padding):
    class MaxPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = nn.MaxPool2d(
                kernel_size=(1, 40),
                padding=(0, padding),
            )

        def forward(self, a):
            b = self.maxpool(a)
            
            return b

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS
    else:
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    framework_model = MaxPool()
    module = PyTorchModule("MaxPool", framework_model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    input_shape = (10, 100, 50)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

def test_eval_mode(test_kind):
    """
    Verify that module is put into eval/trainig mode appropriately
    """

    mod = torch.nn.Linear(64, 64)
    mod.training = True
    print(f"mod.training = {mod.training}")
    verify_module(PyTorchModule("linear", mod), [(1, 64, 64)],
            verify_cfg=VerifyConfig(test_kind=test_kind))

    print(f"mod.training = {mod.training}")
    assert mod.training == test_kind.is_training()

class PTLinear1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 128, bias=True)

    def forward(self, x1):
        return self.l1(x1)

class PTLinear2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.Linear(128, 64, bias=True)

    def forward(self, x1):
        return self.l2(x1)

class TTLinear1(PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(64, 128, requires_grad=True)

    def forward(self, act1):
        return pybuda.op.Matmul("", act1, self.weights1)

class TTLinear2(PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
        self.weights2 = pybuda.Parameter(128, 64, requires_grad=True)

    def forward(self, act1):
        return pybuda.op.Matmul("", act1, self.weights2)

@pytest.mark.parametrize("first_module_pt", (True, False), )
@pytest.mark.parametrize("second_module_pt", (True, False), )
def test_multiple_modules_on_device(test_kind, test_device, first_module_pt, second_module_pt):
    if test_kind.is_training(): 
        pytest.skip()

    if first_module_pt:
        mod_1 = PyTorchModule("linear_1", PTLinear1())
        inputs = [torch.rand((1, 64, 64))]
    else:
        mod_1 = TTLinear1("linear_1")
        mod_1.set_parameter("weights1", torch.rand(64, 128, requires_grad=True))
        inputs = [Tensor.create_from_torch(torch.rand((1, 64, 64)))]

    if second_module_pt:
        mod_2 = PyTorchModule("linear_2", PTLinear2())
    else:
        mod_2 = TTLinear2("linear_2")
        mod_2.set_parameter("weights2", torch.rand(128, 64, requires_grad=True))

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=test_device.devtype)
    tt0.place_module(mod_1)
    tt0.place_module(mod_2)

    tt0.push_to_inputs(inputs)
    output_q = pybuda.run_inference()
    data = output_q.get(timeout = 0.5)
    assert data[0].shape.get_pytorch_shape() == (1, 64, 64)


def test_tvm_lstm(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                1024,
                1024,
                num_layers=1,
                bias=True,
                batch_first=False,
                bidirectional=True,
            )

        def forward(self, a):
            b, (c, d) = self.lstm(a)
            
            return b, c, d

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS

    framework_model = LSTM()
    module = PyTorchModule("LSTM", framework_model)

    input_shape = (64, 32, 1024)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

tensordot_inputs = [
    ((3, 4, 5), (4, 5, 6), 2),
    ((3, 5, 4, 6), (6, 4, 5, 3), ([2, 1, 3], [1, 2, 0])),
    ((36, 6, 64), (1, 4), ([-1], [0])),
    ((64, 16, 32), (16, 4, 2), ([1], [0])),
    ((32, 64), (1, 1), ([-1], [0]))
]

# TODO: Convert to verify_module flow when batch dim > 1 is supported
@pytest.mark.parametrize("recompute", (True, False), ids=["recompute", "no_recompute"])
@pytest.mark.parametrize(
    "tensordot_inputs",
    tensordot_inputs,
    ids=[f"input{str(s)}" for s in tensordot_inputs],
)
def test_tvm_tensordot(training, recompute, tensordot_inputs):
    shape_x, shape_y, dims = tensordot_inputs
    if training:
        pytest.xfail()  # Backward is currently unsupported 

    if not training and recompute:
        pytest.skip()  # inference + recompute is the same as just inference

    class TensordotMod(nn.Module):
        
        def forward(self, x, y):
            return torch.tensordot(x, y, dims)

    module = PyTorchModule("tensordot", TensordotMod())

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    x, y = torch.randn(shape_x), torch.randn(shape_y)

    try:
        pybuda_compile(
            tt0,
            "tensordot",
            x,y,
            compiler_cfg=CompilerConfig(
                enable_training=training,
                enable_recompute=recompute,
            ),
            verify_cfg=VerifyConfig(
                intermediates=True,
            ),
        )
    except pybuda._C.UnsupportedHWOpsError:
        pass


def test_tvm_bfloat_custom_param(test_kind, test_device):
    class BFloatCustomParamModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.rand((1, 1, 9, 3), dtype=torch.bfloat16))

        def forward(self, x, b):
            m = torch.matmul(x, self.w)

            return m + b

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        compiler_cfg.compile_depth = CompileDepth.FULL

    framework_model = BFloatCustomParamModule()
    module = PyTorchModule("BFloatCustomParamModule", framework_model)

    verify_module(
        module,
        [(1, 1, 3, 9), (1, 1, 1, 3)],
        input_params=[{"data_format": torch.bfloat16}, {"data_format": torch.bfloat16}],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_arbitrary_attribute_set(test_kind, test_device):
    class ArbitraryAttributeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.rand((1, 1, 9, 3), dtype=torch.bfloat16))

        def forward(self, x, b):
            self.m = torch.matmul(x, self.w)

            return self.m + b

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        compiler_cfg.compile_depth = CompileDepth.FULL

    framework_model = ArbitraryAttributeModule()
    module = PyTorchModule("ArbitraryAttributeModule", framework_model)
    
    original_model_param_dtype = {}
    for key, val in framework_model.state_dict().items():
        original_model_param_dtype[key] = val.dtype

    verify_module(
        module,
        [(1, 1, 3, 9), (1, 1, 1, 3)],
        input_params=[{"data_format": torch.bfloat16}, {"data_format": torch.bfloat16}],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
    
    for key, val in module.module.state_dict().items():
        if original_model_param_dtype[key] != val.dtype:
            msg = "Original PyTorch model params have been modified (not allowed during compilation)."
            msg += f"Parameter '{key}', has dtype '{val.dtype}', while original is '{original_model_param_dtype[key]}'"
            pytest.fail(msg)


@pytest.mark.parametrize("layer_norm_shape", ((9), (9, 9), (3, 9, 9)))
@pytest.mark.parametrize("input_shape", ((1, 3, 9, 9),))
def test_tvm_layernorm(test_kind, test_device, layer_norm_shape, input_shape):
    if type(layer_norm_shape) != int:
        pytest.skip()  # Supports only normalization over the single last dimension
                
    class LayerNormModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(layer_norm_shape)

        def forward(self, x):
            return self.ln(x)

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER  # Unsupported HW ops
    else:
        compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER  # Unsupported HW ops

    framework_model = LayerNormModule()
    module = PyTorchModule("LayerNormModule", framework_model)

    verify_module(
        module,
        [input_shape,],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_list_input(test_kind, test_device):
    
    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    class ListInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 3)
            self.linear2 = nn.Linear(3, 3)
            self.smx = nn.Softmax(2)


        def forward(self, inputs):
            x = inputs[0]
            y = inputs[1]

            x = self.linear1(x)
            x = self.linear2(x)
            y = self.linear1(y)
            y = self.linear2(y)
            return x + y

    model = ListInputModel()
    module = PyTorchModule("ListInputModel", model)
       
    input_shape = (1, 1, 3)
    inputs = [(torch.randn(input_shape), torch.randn(input_shape))]

    verify_module(
        module,
        (),
        inputs=[inputs], # Each input is a list for one iteration of forward/backward
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_list_input_mixture(test_kind, test_device):

    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    class ListInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 3)
            self.linear2 = nn.Linear(3, 3)
            self.smx = nn.Softmax(2)


        def forward(self, inputs1, z, inputs2, w):
            x = inputs1[0]
            y = inputs1[1]

            a = inputs2[0]
            b = inputs2[1]
            c = inputs2[2]

            x = self.linear1(x)
            x = self.linear2(x)
            y = self.linear1(y)
            y = self.linear2(y)

            w = self.linear2(self.linear1(w))
            z = self.linear2(self.linear1(z))
            return x + y + w + z + a + b + c

    model = ListInputModel()
    module = PyTorchModule("ListInputModel", model)
   
    input_shape = (1, 1, 3)
    # Inputs are: tensor, list[tensor], list[tensor], tensor
    inputs = [[2*torch.ones(input_shape), 3*torch.ones(input_shape)], torch.ones(input_shape), [4*torch.ones(input_shape), 5*torch.ones(input_shape), 6*torch.ones(input_shape)], 7*torch.ones(input_shape)]

    verify_module(
        module,
        (),
        inputs=[inputs], # Each input is a list for one iteration of forward/backward
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_dict_input(test_kind, test_device):
    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend
    class DictInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, 3)
            self.linear2 = nn.Linear(3, 3)
            self.smx = nn.Softmax(2)


        def forward(self, z, inputs, x):
            q = inputs['x']
            y = inputs['y']

            q = self.linear1(q)
            q = self.linear2(q)
            y = self.linear1(y)
            y = self.linear2(y)
            return q + y + x[0] + x[1] + z

    model = DictInputModel()
    module = PyTorchModule("DictInputModel", model)
       
    input_shape = (1, 1, 3)
    inputs = [torch.randn(input_shape), {'x':torch.randn(input_shape), 'y':torch.randn(input_shape)}, [torch.randn(input_shape), torch.randn(input_shape)]]

    verify_module(
        module,
        (),
        inputs=[inputs], # Each input is a list for one iteration of forward/backward
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_dropout(test_kind, test_device):
    if test_kind != TestKind.TRAINING_RECOMPUTE:
        pytest.skip()
    class Dropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, a):
            b = self.dropout(a)
            
            return b

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    compiler_cfg.enable_tvm_dropout = True


    framework_model = Dropout()
    module = PyTorchModule("Dropout", framework_model)

    input_shape = (10, 100, 50)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                test_kind=test_kind,
            ),
        )


def test_tvm_indexing(test_kind, test_device):
    class IndexModuleWrapper(nn.Module):
        def __init__(self):
            super().__init__()

            self.index_module = IndexModule()
            self.mask = torch.zeros((5,), dtype=torch.bool)
            self.mask[-1] = True
            self.mask[-3] = True
            self.linear = nn.Linear(2, 2, bias=True)

        def forward(self, x):
            a = self.index_module(x, self.mask)
            return self.linear(a)

    class IndexModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, mask):
            x = x[0]
            x = x[:, mask]

            return x

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        # Unsupported AdvIndex decomposition. Atm, needed HW support for for
        # op similar to take (lookup) with indices.
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH
    else:
        # Unsupported AdvIndex decomposition. Atm, needed HW support for for
        # op similar to take (lookup) with indices.
        compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    framework_model = IndexModuleWrapper()
    module = PyTorchModule("IndexModule", framework_model)

    x = torch.rand((1, 2, 5))
    verify_module(
        module,
        [],
        inputs=[(x,),],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

@pytest.mark.parametrize("dim", (0, 1, 2, 3, -1, -2, -3, -4))
@pytest.mark.parametrize("input_shape", ((1, 4, 4), (1, 3, 7), (1, 7, 4), (1, 4, 7), (1, 8, 7, 9), (1, 8, 7, 9, 5)))
def test_tvm_torch_flip(test_kind, test_device, input_shape, dim):
    if dim >= len(input_shape[1:]) or (dim < 0 and abs(dim) > len(input_shape[1:])):
        pytest.skip()
    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.compile_depth = CompileDepth.CONSTEVAL_GRAPH
    class Flip(torch.nn.Module):
        def __init__(self,dim,feature_size):
            super().__init__()
            self.dim = dim
            self.l1 = torch.nn.Linear(feature_size, feature_size)
        def forward(self,input):
            input = self.l1(input)
            input = input[0]
            output = torch.flip(input, [self.dim])
            return output
    model = Flip(dim=dim,feature_size=input_shape[-1])
    model.eval()
    tt_model = pybuda.PyTorchModule("flip_tvm_decompose_adv_index", model)
    verify_module(
        tt_model,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_tvm_compile = True,
        ),
    )

def test_tvm_adv_indexing_batch1(test_kind, test_device):
    # reproduce the decomposition of adv_index op at the end of gpt_neo model
    class AdvIndexBatch1Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.mask = torch.randint(0,2,size=(1,))

        def forward(self, x):
            x = x[self.mask]
            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH
    framework_model = AdvIndexBatch1Module()
    module = PyTorchModule("AdvIndexBatch1Module", framework_model)

    x = torch.rand((1, 2))
    verify_module(
        module,
        [],
        inputs=[(x,),],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_torch_tanh_gelu(test_kind, test_device):

    if test_kind == TestKind.TRAINING:
        pytest.skip()

    from transformers.activations import NewGELUActivation

    framework_model = NewGELUActivation()
    module = PyTorchModule("TorchTanhGeluModule", framework_model)

    verify_module(
        module,
        ((1, 32, 64, 64),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            enable_input_gradient_checking=False,
        ),
    )

def test_tvm_torch_erf_gelu(test_kind, test_device):

    if test_kind == TestKind.TRAINING:
        pytest.skip()

    from transformers.activations import GELUActivation

    framework_model = GELUActivation()
    module = PyTorchModule("TorchErfGeluModule", framework_model)

    verify_module(
        module,
        ((1, 32, 64, 64),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            enable_input_gradient_checking=False,
        ),
    )

@pytest.mark.parametrize("dim", (0, 1, 2))
@pytest.mark.parametrize("exclusive", (True, False, None))
def test_tvm_cumulative_sum(test_kind, test_device, dim, exclusive):
    if dim != 0:
        pytest.skip()  # Unsupported
        
    if exclusive:
        pytest.skip()  # Unsupported
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.PRE_LOWERING_PASS 
    compiler_cfg.retain_tvm_python_files = True
    # compiler_cfg.cpu_fallback_ops.add("cumsum")
    class CumulativeSum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cumsum(x, dim=dim)
        
    framework_module = CumulativeSum()
    module = PyTorchModule("pt_cumsum", framework_module)

    input_shape = (1, 2, 3, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )

def test_tvm_repeat(test_kind, test_device,):

    if test_kind.is_training():
        pytest.skip()


    class Repeat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.repeat(1,3,4)

    framework_module = Repeat()
    module = PyTorchModule("Repeat", framework_module)

    input_shape = (1, 31, 63)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


def test_tvm_nd_reshape(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.PRE_LOWERING_PASS  # Unsupported HW ops
        
    class NdReshape(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64, bias=True)
            
        def forward(self, x):
            x = self.linear(x)
            x = torch.reshape(x, (1, 1, 3, 32, 64))
            x = self.linear(x)

            return x + 1

    framework_model = NdReshape()
    mod = PyTorchModule("NdReshape", framework_model)
    input_shape = (1, 3, 32, 64)
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
    
def test_tvm_layernorm_cpu(test_device):

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("layernorm")

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.l1 = torch.nn.Linear(9, 9)
            self.layer_norm = torch.nn.LayerNorm(9)

        def forward(self, x):
            x = self.l1(x)
            x = self.layer_norm(x)
            return x

    framework_module = Module()
    framework_module.eval()
    pybuda_module = PyTorchModule("pt_layermorm_cpu", framework_module)

    input_shape = (1, 9, 9)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


def test_tvm_dropout_cpu(test_device):

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("nn.dropout")

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout =  torch.nn.Dropout()
        def forward(self, x):
            x = self.dropout(x)
            x = torch.add(x, x)
            return x

    framework_module = Module()
    framework_module.eval()
    pybuda_module = PyTorchModule("pt_dropout_cpu", framework_module)

    input_shape = (1, 9, 9)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )


def test_tvm_adv_index_bool_cpu_0(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("adv_index")
    compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.l1 = torch.nn.Linear(9, 21)
            self.mask = torch.randint(0, 2, (21,), dtype=torch.bool)

        def forward(self, x):
            x = self.l1(x)
            
            x = x[0]
            x = x[:, self.mask]

            return x

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_adv_index_bool_cpu_0", framework_module)

    input_shape = (1, 9, 9)

    # Sanity run
    # input_x = torch.rand(input_shape)
    # out = framework_module(input_x)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_tvm_adv_index_bool_cpu_1(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("adv_index")
    compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.l1 = torch.nn.Linear(9, 21)
            self.mask = torch.randint(0, 2, (21,), dtype=torch.bool)

        def forward(self, x):
            x = self.l1(x)
            
            x = x[0][3]
            x = x[self.mask]

            return x

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_adv_index_bool_cpu_1", framework_module)

    input_shape = (1, 9, 9)

    # Sanity run
    # input_x = torch.rand(input_shape)
    # out = framework_module(input_x)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_tvm_adv_index_bool_cpu_2(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tm_cpu_fallback = True
    compiler_cfg.tm_cpu_fallback_max_depth = 3
    compiler_cfg.cpu_fallback_ops.add("adv_index")
    compiler_cfg.cpu_fallback_ops.add("argwhere")
    compiler_cfg.cpu_fallback_ops.add("less")
    compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.l1 = torch.nn.Linear(9, 21)

        def forward(self, x):
            x = self.l1(x)
            
            x = x[0][3]
            mask = x < 0
            x[mask] = x[mask] + 1

            return x

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_adv_index_bool_cpu_2", framework_module)

    input_shape = (1, 9, 9)

    # Sanity run
    # input_x = torch.rand(input_shape)
    # out = framework_module(input_x)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_tvm_hslice_a(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind.is_training():
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 2048
            self.num_heads = 16
            self.head_dim = 128
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
            self.k_attn = nn.Linear(self.embed_dim, self.head_dim)

        def forward(self, hidden_states, past_key):
            hidden_states = hidden_states.squeeze(0)
            past_key = past_key.squeeze(0)

            query = self.q_attn(hidden_states)
            key = self.k_attn(hidden_states)

            seq_length = query.size(0)

            query = query.reshape(seq_length, self.num_heads, self.head_dim)
            query = query.permute(1,0,2)
            query = query.reshape(self.num_heads * seq_length, self.head_dim)

            key = key.permute(1, 0)
            full_key = torch.cat((past_key, key), dim=-1)

            return query.unsqueeze(0), full_key.unsqueeze(0)


    framework_module = Module()
    pybuda_module = PyTorchModule("pt_hslice_a", framework_module)

    hidden_states_shape = (1, 1, 2048)
    past_key_shape = (1, 128, 2047)

    # Sanity check
    # hidden_states = torch.randn(hidden_states_shape)
    # past_key = torch.randn(past_key_shape)
    # out = framework_module(hidden_states, past_key)

    verify_module(
        pybuda_module,
        (hidden_states_shape, past_key_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_tvm_hslice_c(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind.is_training():
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 2048
            self.num_heads = 16
            self.head_dim = 128
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
            self.k_attn = nn.Linear(self.embed_dim, self.head_dim)

        def forward(self, hidden_states, past_key):
            hidden_states = hidden_states.squeeze(0)
            past_key = past_key.squeeze(0)

            query = self.q_attn(hidden_states)
            key = self.k_attn(hidden_states)

            seq_length = query.size(0)

            query = query.reshape(seq_length, self.num_heads, self.head_dim)
            query = query.permute(1,0,2)
            query = query.reshape(self.num_heads * seq_length, self.head_dim)

            key = key.permute(1, 0)
            full_key = torch.cat((past_key, key), dim=-1)

            query = query.unsqueeze(0)
            full_key = full_key.unsqueeze(0)

            query = query.squeeze(0)
            full_key = full_key.squeeze(0)

            attn_weights = torch.mm(query, full_key)

            return attn_weights.unsqueeze(0)

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_hslice_c", framework_module)

    hidden_states_shape = (1, 1, 2048)
    past_key_shape = (1, 128, 2047)

    # Sanity check
    # hidden_states = torch.randn(hidden_states_shape)
    # past_key = torch.randn(past_key_shape)
    # out = framework_module(hidden_states, past_key)

    verify_module(
        pybuda_module,
        (hidden_states_shape, past_key_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_tvm_non_hslice(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind.is_training():
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY
    compiler_cfg.enable_t_streaming = False

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 2048
            self.num_heads = 16
            self.head_dim = 128
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
            self.k_attn = nn.Linear(self.embed_dim, self.head_dim)

        def forward(self, hidden_states, past_key):
            query = self.q_attn(hidden_states) # (1, 2048)
            key = self.k_attn(hidden_states)   # (1,  128)

            assert 1 == query.size(0), "single batch decode only"
            assert 1 == query.size(1), "single token decode only"

            query = query.view(1, self.num_heads, self.head_dim) # (1, 16, 128)
            key = key.permute(0, 2, 1)                           # (1, 128, 1)
            full_key = torch.cat((past_key, key), dim=-1)        # (1, 128, 2048)

            attn_weights = torch.bmm(query, full_key)            # (1, 16, 2048)

            return attn_weights

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_hslice_d", framework_module)

    hidden_states_shape = (1, 1, 2048)
    past_key_shape = (1, 128, 2047)

    # Sanity check
    # hidden_states = torch.randn(hidden_states_shape)
    # past_key = torch.randn(past_key_shape)
    # out = framework_module(hidden_states, past_key)

    verify_module(
        pybuda_module,
        (hidden_states_shape, past_key_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_tvm_splice_with_scalar_concat(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind.is_training():
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.const1 = torch.tensor([1.0]).reshape(1, 1)

        def forward(self, act):
            out = torch.sum(act, dim=1)
            out = torch.sum(out, dim=1)
            out = out.reshape(-1, 1)
            out = torch.cat([out, self.const1], dim=1)

            return out

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_splice_with_scalar_concat", framework_module)

    act_shape = (1, 3, 9)

    # Sanity check
    act = torch.randn(act_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )
    

def test_tvm_concat_decomp_inti_smm(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind.is_training():
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.const1 = torch.tensor([1.0]).reshape(1, 1)

        def forward(self, act):
            out = torch.matmul(act, act.transpose(-1, -2))
            out = torch.sum(act, dim=-1)
            
            act = torch.sum(act, dim=-1)
            out = torch.cat([act, out], dim=-1)

            return out

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_concat_decomp_smm", framework_module)

    act_shape = (1, 3, 9, 9)

    # Sanity check
    act = torch.randn(act_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


@pytest.mark.parametrize("tile_aligned", (True, False))
def test_yz_transpose(test_kind, test_device, tile_aligned):
    if test_kind.is_training():
        pytest.skip()
    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear0 = nn.Linear(32, 32, bias=False)
            self.linear1 = nn.Linear(32, 32, bias=False)

        def forward(self, act):
            out = self.linear0(act)
            out = out.permute(0, 2, 1, 3)
            out = self.linear1(out)

            return out
        

    if not tile_aligned:
        act_shape = (1, 3, 9, 32)
    else:
        act_shape = (1, 3, 32, 32)

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_yz_transpose", framework_module)

    act = torch.randn(act_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_tvm_embedding(test_kind, test_device):
    pytest.xfail(reason="We currently can't do embedding on TT device if there are some ops between embeddings and input activations.")
    
    if test_kind.is_training():
        pytest.skip()

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.enable_tm_cpu_fallback = False

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(50265, 1024, 1)

        def forward(self, input_ids):
            input_ids = torch.reshape(input_ids, (-1,))
            out = self.embedding(input_ids)
            
            return out

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_embedding", framework_module)

    input_ids_shape = (1, 1, 1, 3)

    # Sanity check
    input_ids = torch.randint(high=25000, size=input_ids_shape)
    out = framework_module(input_ids)

    verify_module(
        pybuda_module,
        (input_ids_shape,),
        inputs=[(input_ids,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
    
    
def test_tvm_splice_with_tm_vslice_0(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, act):
            act = torch.reshape(act, (1, 64, 64, 1))
            act = act[:, :, :32, :]

            return act

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_splice_with_tm_vslice_0", framework_module)

    act_shape = (1, 1, 4096, 1)

    # Sanity check
    # act = torch.randn(act_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_splice_with_tm_vslice_1(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = 4

        def forward(self, act):
            bs, width, length = act.shape
            ch = width // (3 * self.n_heads)
            reshaped_qkv = act.reshape(bs * self.n_heads, ch * 3, length)
            q, k, v = reshaped_qkv.split(ch, dim=1)

            return q, k, v

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_splice_with_tm_vslice_1", framework_module)

    act_shape = (1, 1536, 1024)

    # Sanity check
    # act = torch.randn(act_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_splice_with_tm_hslice(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, act):
            act = torch.reshape(act, (1, 1, 64, 64))
            act = act.transpose(1, 2)
            act = act[:, :, :, :32]
            
            return act

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_splice_with_tm_hslice", framework_module)

    act_shape = (1, 1, 1, 4096)

    # Sanity check
    # act = torch.randn(act_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_splice_with_tm_vstack(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, act):
            act = torch.reshape(act, (1, 1, 1, 4096))
            act = act[:, :, :, :128]
            
            return act

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_splice_with_tm_vstack", framework_module)

    act_shape = (1, 1, 64, 64)

    # Sanity check
    # act = torch.randn(act_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_splice_with_tm_hstack(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, act):
            act = act.transpose(1, 2)
            act = torch.reshape(act, (1, 1, 1, 4096))
            act = act[:, :, :, :128]

            return act

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_splice_with_tm_hstack", framework_module)

    act_shape = (1, 64, 1, 64)

    # Sanity check
    # act = torch.randn(act_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_kernel_fracturing_with_grouped_conv(test_kind, test_device):
    pytest.skip("tenstorrent/pybuda#455")
    
    if test_kind.is_training():
        pytest.skip()

    import os
    if test_device.is_wormhole():
        os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "60000"

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    compiler_cfg.conv_multi_op_fracture_factor_override["conv2d_1"] = 2

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                3,
                3,
                kernel_size=2,
                padding=0,
                groups=3,
            )

        def forward(self, act):
            act = self.conv(act)

            return act

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_kernel_fracturing_with_grouped_conv", framework_module)

    act_shape = (1, 3, 9, 9)

    # Sanity check
    act = torch.randn(act_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_BN_no_stats(test_kind, test_device):
    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        pytest.skip("Skip this test for golden Wormhole B0")
    
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip("Wait until #1006 is resolved")

    if test_kind.is_training():
        pytest.skip()
        
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"
    #Fusing disabled due to tenstorrent/pybuda#789
    compiler_cfg.enable_auto_fusing=False
    class ModelBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64, track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128, track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256, track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512, track_running_stats=False),
                nn.ReLU()
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(512, 10)
            

        def forward(self, x):
            x = self.cnn(x)
            x = self.pool(x)
            x = x.view(x.shape[0], -1)
            x = self.linear(x)
            return x

    framework_module = ModelBN()
    pybuda_module = PyTorchModule("pt_BN_no_stats", framework_module)

    act_shape = (1, 3, 32, 32)

    # Sanity check
    act = torch.randn(act_shape)
    out = framework_module(act)

    try:
        verify_module(
            pybuda_module,
            (act_shape,),
            verify_cfg=VerifyConfig(
                arch=test_device.arch,
                devtype=test_device.devtype,
                test_kind=test_kind,
                verify_all=True,
            ),
        )
    finally:
        if test_device.is_wormhole():
            del os.environ["PYBUDA_EXTRA_L1_MARGIN"]


def test_prelu_pytorch(test_kind, test_device):

    if test_kind.is_training():
        pytest.skip()
    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear0 = nn.Linear(32, 32, bias=False)
            self.prelu = nn.PReLU(num_parameters=1024)

        def forward(self, act):
            out = self.linear0(act)
            out = out.reshape((1, 1024))
            out = self.prelu(out)

            return out

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_prelu", framework_module)

    act_shape = (1, 1, 32, 32)

    # Sanity check
    act = torch.randn(act_shape)

    verify_module(
        pybuda_module,
        (act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )

def test_tvm_float16_custom_param_pytorch(test_kind, test_device):
    class Float16CustomParamModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 32, dtype=torch.float32)
            self.w = torch.nn.Parameter(torch.rand((1, 1, 32, 32), dtype=torch.float32))

        def forward(self, tokens):
            x = self.emb(tokens)
            x = torch.matmul(x, self.w)
            x = torch.matmul(x, self.emb.weight.T)
            return x


    module = PyTorchModule("BFloatCustomParamModule", Float16CustomParamModule())
    inputs = [torch.tensor([[1, 2]])]
    verify_module(
        module,
        input_shapes=([1,2],),
        inputs=inputs,
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

import tensorflow as tf
def test_tvm_float16_custom_param_tensorflow(test_kind, test_device):
    class Float16CustomParamModule(tf.keras.Model):
        def __init__(self):
            super().__init__(autocast=False)
            self.dense1 = tf.keras.layers.Dense(32, use_bias=True, dtype=tf.bfloat16)

        def call(self, x):
            return self.dense1(x)


    module = TFModule("BFloatCustomParamModule", Float16CustomParamModule())
    verify_module(
        module,
        [(1, 1, 32, 32)],
        input_params=[{"data_format": tf.bfloat16}],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_disjoint_graph():

    class Module_A(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear0 = nn.Linear(32, 32, bias=False)
            self.prelu = nn.PReLU(num_parameters=1024)

        def forward(self, act):
            out = self.linear0(act)
            out = out.reshape((1, 1024))
            out = nn.Softmax(dim=-1)(out)

            return out


    class Module_B(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(9, 9)
            self.leaky_relu = nn.LeakyReLU(0.2)
            
        def forward(self, x):
            x = self.linear(x)

            return self.leaky_relu(x)

    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    _get_global_compiler_config().compile_subgraphs = True
    _get_global_compiler_config().enable_auto_fusing = False
    mod_A = PyTorchModule("mod_A", Module_A())
    tt0.place_module(mod_A)
    
    mod_B = PyTorchModule("mod_B", Module_B())
    tt0.place_module(mod_B)


    act_shape_A = (1, 1, 32, 32)
    act_shape_B = (1, 3, 9, 9)

    output_q = pybuda.initialize_pipeline(
        training=False,
        sample_inputs=((torch.rand(act_shape_A),), (torch.rand(act_shape_B),)))
    
    tt0.set_active_subgraph(0)
    tt0.push_to_inputs((torch.rand(act_shape_A), ))
    pybuda.run_forward()

    tt0.set_active_subgraph(1)
    tt0.push_to_inputs((torch.rand(act_shape_B), ))
    pybuda.run_forward()


def test_override_removal_flag(test_kind, test_device):
    from pybuda.config import _set_pybuda_override_veto
    
    # Setup override veto
    os.environ["PYBUDA_OVERRIDES_VETO"] = "1"
    _set_pybuda_override_veto({
        # Level 0 overrides
        "backend_device_descriptor_path": "",
        
        # Level 1 overrides
        "balancer_policy": "",
        "enable_t_streaming": "",
    },
    {
        # Level 2 overrides
        "PYBUDA_RIBBON2": "",
        "PYBUDA_DISABLE_STREAM_OUTPUT": "",
        "PYBUDA_PAD_OUTPUT_BUFFER": "",
    })

    # Only single run run is needed
    if test_kind.is_training():
        pytest.skip()
        
    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32, bias=False)

        def forward(self, act):
            out = self.linear(act)

            return out

    # General compiler configuration overrides
    start_compiler_cfg = _get_global_compiler_config()
    start_compiler_cfg.balancer_policy = "CNN"
    start_compiler_cfg.amp_level = 1
    
    # Environement variable compiler configuration overrides
    os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "60000"
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Load PyBuda module
    pybuda_module = PyTorchModule("pt_override_removal", Module())
    
    env_vars_before_compile = os.environ
    verify_module(
        pybuda_module,
        ((1, 1, 32, 32),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
    env_vars_after_compile = os.environ
    
    # Difference between general compiler configurations
    del os.environ["PYBUDA_OVERRIDES_VETO"]
    end_compiler_cfg = _get_global_compiler_config()
    compiler_cfg_attrs = [a for a in dir(end_compiler_cfg) if not a.startswith('__') and not callable(getattr(end_compiler_cfg, a))]
    diff = {}
    for k in compiler_cfg_attrs:
        if getattr(start_compiler_cfg, k) != getattr(end_compiler_cfg, k):
            diff[k] = str(getattr(start_compiler_cfg, k)) + " vs " + str(getattr(end_compiler_cfg, k))
    print("General compile difference : ", diff)
    
    # TODO: Thise are runtime set, shouldn't be used as part of compiler config
    diff.pop('backend_output_dir', None) # different hash generated because envs change between the two runs
    diff.pop('backend_device_descriptor_path', None)
    assert diff == {'amp_level': '1 vs None'}

    # Difference between environment variable compiler configurations
    diff = {k: env_vars_after_compile[k] for k in set(env_vars_after_compile) - set(env_vars_before_compile)}
    print("Env var compile difference: ", diff)  
    
    assert diff == {}, "There should be no difference in env vars before and after compile"


def test_torch_conv3d(test_device):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    inC, inD, inH, inW = (2, 5, 5, 5)
    outC, kD, kH, kW = (4, 3, 3, 3)

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(inC, inD, (kD, kH, kW))

        def forward(self, act):
            act = self.conv(act)
            return act

    pybuda_module = PyTorchModule("pt_conv3d", Module())

    input_shape = (1, inC, inD, inH, inW)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
            verify_post_autograd_passes=False,
            verify_post_placer=False,
        ),
    )


def test_torch_maxpool3d(test_device):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    inC, inD, inH, inW = (3, 8, 8, 8)
    outC, kD, kH, kW = (3, 3, 3, 3)
    stride = 1

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = nn.MaxPool3d((kD, kH, kW), stride=stride)

        def forward(self, act):
            act = self.maxpool(act)
            return act

    pybuda_module = PyTorchModule("pt_maxpool3d", Module())

    input_shape = (1, inC, inD, inH, inW)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
            verify_post_autograd_passes=False,
            verify_post_placer=False,
        ),
    )

def test_reflection_pad(test_device):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.reflection_pad = nn.ReflectionPad2d(2)

        def forward(self, act):
            act = self.reflection_pad(act)
            return act

    pybuda_module = PyTorchModule("reflection_pad", Module())


    verify_module(
        pybuda_module,
        ((1, 1, 3, 3),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
            verify_all=True,
        ),
    )

@pytest.mark.parametrize("mha", (True, False), )
@pytest.mark.parametrize("prefill", (True, False), )
def test_tvm_scaled_dot_product_attention(test_device, mha, prefill):
    nqh = 16
    s = 2048
    b = 1
    dh = 64
    sq = s if prefill else 1
    nkvh = nqh if mha else 1

    q_shape = (b, nqh, sq, dh)
    k_shape = (b, nkvh, s, dh)
    v_shape = (b, nkvh, s, dh)
    attn_mask_shape = (b, 1, sq, s)


    pybuda.set_configuration_options(enable_auto_fusing=False)

    class SDPA(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, Q, K, V, attn_mask):
            return F.scaled_dot_product_attention(Q, K, V, attn_mask)

    model = SDPA()
    mod = PyTorchModule("scaled_dot_product_attention", model)

    verify_module(
        mod,
        (q_shape, k_shape, v_shape, attn_mask_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
        )
    )

