# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from typing import Sequence

import jax
from jax import numpy as jnp
# from flax import linen as nn

from pybuda import (
    JaxModule,
    VerifyConfig,
)
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module
from pybuda.config import CompileDepth, _get_global_compiler_config

def test_tvm_linear(test_kind, test_device):
    pytest.skip()

    class Linear(nn.Module):
        features: Sequence[int]

        def setup(self):
            self.dense = nn.Dense(features=self.features[0], use_bias=True)

        def __call__(self, x):
            x = self.dense(x)

            return x

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.FULL
    else:
        compiler_config.compile_depth = CompileDepth.FULL

    # Initialize module
    input_shape = (1, 64)
    framework_module = Linear([64])

    # Bind params to module
    key = jax.random.PRNGKey(0)
    act = jax.random.uniform(key, input_shape)
    vars = framework_module.init(key, act)
    framework_module = framework_module.bind(vars)

    pybuda_module = JaxModule("linear", framework_module)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_linear_relu(test_kind, test_device):
    pytest.skip()

    class Linear(nn.Module):
        features: Sequence[int]

        def setup(self):
            self.dense = nn.Dense(features=self.features[0], use_bias=True)

        def __call__(self, x):
            x = self.dense(x)
            x = nn.relu(x)

            return x

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        # Op type parsed is not supported in backend maximum
        compiler_config.compile_depth = CompileDepth.GENERATE_NETLIST
    else:
        # Data mismatch on maximum on:
        # Tensor mismatch on bw_in0_maximum_14_nop_0 from add_12
        compiler_config.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    # Initialize module
    input_shape = (1, 64)
    framework_module = Linear([64])

    # Bind params to module
    key = jax.random.PRNGKey(0)
    act = jax.random.uniform(key, input_shape)
    vars = framework_module.init(key, act)
    framework_module = framework_module.bind(vars)

    pybuda_module = JaxModule("linear", framework_module)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_multiple_outputs(test_kind, test_device):
    pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    class Linear(nn.Module):
        features: Sequence[int]

        def setup(self):
            self.dense1 = nn.Dense(features=self.features[0], use_bias=True)
            self.dense2 = nn.Dense(features=self.features[1], use_bias=True)
            self.dense3 = nn.Dense(features=self.features[2], use_bias=True)

        def __call__(self, x):
            a = self.dense1(x)
            b = self.dense2(a)
            c = self.dense3(b)

            return a, b, c

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.FULL
    else:
        compiler_config.compile_depth = CompileDepth.FULL
    compiler_config.retain_tvm_python_files = True

    # Initialize module
    input_shape = (1, 1, 32, 32)
    framework_module = Linear([64, 32, 128])

    # Bind params to module
    key = jax.random.PRNGKey(0)
    act = jax.random.uniform(key, input_shape)
    vars = framework_module.init(key, act)
    framework_module = framework_module.bind(vars)

    pybuda_module = JaxModule("jax_multiple_output", framework_module)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            # verify_each_buda_pass=True,
        )
    )


def test_tvm_scaled_dot_product_attention(test_kind, test_device):
    pytest.skip()

    class ScaledDotProductAttention(nn.Module):
        def __call__(self, q, k, v):
            kt = jnp.transpose(k, axes=[0, 2, 1])
            qk = jnp.matmul(q, kt)

            qk_den = jnp.sqrt(qk.shape[-1])
            qk_scaled = qk / qk_den

            qk_norm = nn.softmax(qk_scaled)

            qkv = jnp.matmul(qk_norm, v)

            return qkv

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.FULL
    else:
        compiler_config.compile_depth = CompileDepth.FULL
    compiler_config.retain_tvm_python_files = True

    # Initialize module
    input_shape = (1, 32, 64)
    framework_module = ScaledDotProductAttention()

    # Bind params to module
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, num=3)
    q = jax.random.uniform(subkeys[0], input_shape)
    k = jax.random.uniform(subkeys[1], input_shape)
    v = jax.random.uniform(subkeys[2], input_shape)
    vars = framework_module.init(key, q, k, v)
    framework_module = framework_module.bind(vars)

    # Run module
    # res = framework_module(q, k, v)

    pybuda_module = JaxModule("jax_scaled_dot_product_attention", framework_module)
    verify_module(
        pybuda_module,
        (input_shape, input_shape, input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_layer_norm(test_kind, test_device):
    pytest.skip()

    # TODO: Checkout why recompute fails.
    if test_kind == TestKind.TRAINING_RECOMPUTE:
        pytest.skip()

    class Wrapper(nn.Module):
        def setup(self):
            self.dense = nn.Dense(features=128, use_bias=True)
            self.norm = nn.LayerNorm()

        def __call__(self, x):
            act = self.dense(x)
            act = self.norm(act)

            return act

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.FULL
    else:
        compiler_config.compile_depth = CompileDepth.FULL
    compiler_config.retain_tvm_python_files = True

    # Initialize module
    input_shape = (1, 64, 128)
    framework_module = Wrapper()

    # Bind params to module
    key = jax.random.PRNGKey(0)
    act = jax.random.uniform(key, input_shape)
    vars = framework_module.init(key, act)
    framework_module = framework_module.bind(vars)
    
    # Run module
    # res = framework_module(act)

    pybuda_module = JaxModule("jax_layer_norm", framework_module)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_conv2d(test_kind, test_device):
    pytest.skip()

    class Conv2d(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=4, feature_group_count=2, kernel_size=(3, 3))(x)
            x = x[:, :, 0:3, 0]
            # x = nn.avg_pool(x, window_shape=(2,2), strides=(1,1,), padding="VALID",)
            # x = nn.relu(x)
            # x = nn.Conv(features=64, kernel_size=(3, 3))(x)
            # x = nn.relu(x)
            # x = x.reshape((x.shape[0], -1))
            # x = nn.Dense(features=256)(x)
            # x = nn.relu(x)
            # x = nn.Dense(features=10)(x)
            # x = nn.log_softmax(x)

            return x

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    compiler_config.enable_xla_jax_convert = True

    # Initialize module
    input_shape = (1, 28, 28, 4)
    framework_module = Conv2d()

    # Bind params to module
    key = jax.random.PRNGKey(0)
    act = jax.random.uniform(key, input_shape)
    vars = framework_module.init(key, act)
    framework_module = framework_module.bind(vars)
    
    # Run module
    # res = framework_module(act)

    pybuda_module = JaxModule("jax_conv2d_test", framework_module)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

# class XLAGatherModule1(nn.Module):
#     def __call__(self, x):
#         return x[:, :, :, 0]

# class XLAGatherModule2(nn.Module):
#     def __call__(self, x):
#         return x[:, :, 0:3, 0]

# class XLAGatherModule3(nn.Module):
#     def __call__(self, x):
#         return x[:, :, 3:9, 0]

# class XLAGatherModule4(nn.Module):
#     def __call__(self, x):
#         return x[:, 3:9, :, 0]
    
# class XLAGatherModule5(nn.Module):
#     def __call__(self, x):
#         return x[:, 3:9, :, 2:4]
    
# class XLAGatherModule6(nn.Module):
#     def __call__(self, x):
#         return x[:, 3:9, 1:6, 2:4]

# @pytest.mark.parametrize("slice_module", (XLAGatherModule1, XLAGatherModule2, XLAGatherModule3, XLAGatherModule4, XLAGatherModule5, XLAGatherModule6,))
# def test_tvm_xla_gather(test_kind, test_device, slice_module):
#     if test_kind.is_training():
#         pytest.skip()
    
#     if slice_module in [XLAGatherModule3, XLAGatherModule4, XLAGatherModule5, XLAGatherModule6]:
#         # tenstorrent/pybuda#1608
#         pytest.skip()

#     compiler_config = _get_global_compiler_config()
#     if not test_kind.is_training():
#         compiler_config.compile_depth = CompileDepth.FULL
#     else:
#         compiler_config.compile_depth = CompileDepth.FULL
#     compiler_config.retain_tvm_python_files = True
#     compiler_config.enable_xla_jax_convert = True

#     # Initialize module
#     input_shape = (1, 28, 28, 4)
#     framework_module = slice_module()

#     # Bind params to module
#     key = jax.random.PRNGKey(0)
#     act = jax.random.uniform(key, input_shape)
#     vars = framework_module.init(key, act)
#     framework_module = framework_module.bind(vars)

#     # Run module
#     # res = framework_module(act)

#     pybuda_module = JaxModule("jax_xla_gather", framework_module)
#     verify_module(
#         pybuda_module,
#         (input_shape,),
#         verify_cfg=VerifyConfig(
#             arch=test_device.arch,
#             devtype=test_device.devtype,
#             test_kind=test_kind,
#         )
#     )


def test_tvm_dense(test_kind, test_device):
    pytest.skip()

    class JAX_dense(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=256)(x)

            return x

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    compiler_config.enable_xla_jax_convert = True
    # Initialize module
    input_shape = (1, 28, 256)
    framework_module = JAX_dense()

    # Bind params to module
    key = jax.random.PRNGKey(0)
    act = jax.random.uniform(key, input_shape)
    vars = framework_module.init(key, act)
    framework_module = framework_module.bind(vars)

    pybuda_module = JaxModule("JAX_dense", framework_module)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

def test_tvm_conv2d_transpose(test_kind, test_device):
    pytest.skip()

    class Conv2d(nn.Module):
        @nn.compact
        def __call__(self, img):
            weight = jax.numpy.ones((4, 4, 2048, 1024))
            img = jax.lax.conv_transpose(
                    img,
                    weight,
                    strides=[2, 2],
                    padding="SAME",
                )

            return img

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    compiler_config.enable_xla_jax_convert = True
    compiler_config.varify_tvm_compile = True

    # Initialize module
    input_shape = (1, 16, 16, 2048)
    framework_module = Conv2d()

    # Bind params to module
    key = jax.random.PRNGKey(0)
    act = jax.random.uniform(key, input_shape)
    vars = framework_module.init(key, act)
    framework_module = framework_module.bind(vars)
    
    # Run module
    # res = framework_module(act)

    pybuda_module = JaxModule("jax_conv2d_transpose_test", framework_module)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

def test_tvm_conv2d_dilated(test_kind, test_device):
    pytest.skip()

    class Conv2d(nn.Module):
        @nn.compact
        def __call__(self, img):
            weight = jax.numpy.ones((4, 4, 512, 1024))
            img = jax.lax.conv_general_dilated(
                img,
                weight,
                window_strides=[2, 2],
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )

            return img

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    else:
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    compiler_config.enable_xla_jax_convert = True

    # Initialize module
    input_shape1 = (1, 64, 64, 512)
    # input_shape2 = (4, 4, 512, 1024)
    framework_module = Conv2d()

    # Bind params to module
    key = jax.random.PRNGKey(0)
    act = jax.random.uniform(key, input_shape1)
    # weight = jax.random.uniform(key, input_shape2)
    vars = framework_module.init(key, act)
    framework_module = framework_module.bind(vars)
    
    # Run module
    # res = framework_module(act)

    pybuda_module = JaxModule("jax_conv2d_dilated_test", framework_module)
    verify_module(
        pybuda_module,
        (input_shape1,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

