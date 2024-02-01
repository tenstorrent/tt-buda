# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from pybuda.config import CompileDepth
from pybuda.op.eval.common import compare_tensor_to_golden
from pybuda.verify.backend import verify_module
from pybuda.verify.config import TestKind
import pytest

import tensorflow as tf

from pybuda import (
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import _get_global_compiler_config

input_shapes = [(1, 128, 64)]
dense_units = [64]


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "dense_units", dense_units, ids=[f"dense({str(d)})" for d in dense_units]
)
def test_tvm_linear(test_kind, test_device, input_shape, dense_units):
    class DoubleLinear(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(dense_units, use_bias=True)
            self.dense2 = tf.keras.layers.Dense(dense_units, use_bias=True)

        def call(self, x1, x2):
            m1 = self.dense1(x1)
            m2 = self.dense2(x2)

            return m1 + m2

    mod = TFModule("linear", DoubleLinear())
    verify_module(
        mod,
        (input_shape, input_shape),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "dense_units", dense_units, ids=[f"dense({str(d)})" for d in dense_units]
)
def test_tvm_gelu(test_kind, test_device, input_shape, dense_units):
    class Gelu(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(dense_units, use_bias=False)

        def call(self, x):
            x1 = self.dense1(x)

            return tf.keras.activations.gelu(x1)

    mod = TFModule("gelu", Gelu())

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "dense_units", dense_units, ids=[f"dense({str(d)})" for d in dense_units]
)
def test_tvm_power(test_kind, test_device, input_shape, dense_units):
    class Power(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(
                dense_units, kernel_initializer="ones", use_bias=False
            )

        def call(self, x):
            x = self.dense1(x)

            return tf.pow(x, -0.5)

    mod = TFModule("Power", Power())
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "dense_units", dense_units, ids=[f"dense({str(d)})" for d in dense_units]
)
def test_tvm_layernorm_tf(test_kind, test_device, input_shape, dense_units):

    class Layernorm(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(dense_units, use_bias=False)
            self.ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0)

        def call(self, x):
            x = self.dense1(x)

            return self.ln(x)

    mod = TFModule("Layernorm", Layernorm())
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "dense_units", dense_units, ids=[f"dense({str(d)})" for d in dense_units]
)
def test_tvm_softmax_tf(test_kind, test_device, input_shape, dense_units):

    class Softmax(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(dense_units, use_bias=True)

        def call(self, x):
            x1 = self.dense1(x)

            return tf.nn.softmax(x1, axis=-1)

    mod = TFModule("Softmax", Softmax())

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


input_shapes = [(1, 32, 32, 32)]
filter_size = [64]
kernel_size = [2, 3]
groups = [1, 2, 4]
channel_format = ["channels_first", "channels_last"]

@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"shape{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "filter_size", filter_size, ids=[f"fsize({str(f)})" for f in filter_size]
)
@pytest.mark.parametrize(
    "kernel_size", kernel_size, ids=[f"ksize({str(k)})" for k in kernel_size]
)
@pytest.mark.parametrize(
    "groups", groups, ids=[f"groups({str(k)})" for k in groups]
)
@pytest.mark.parametrize(
    "channel_format", channel_format, ids=[f"channel_format({k})" for k in channel_format]
)
def test_tvm_conv(test_kind, test_device, input_shape, filter_size, kernel_size, groups, channel_format,):
    if test_kind.is_training():
        pytest.xfail() # Backward is currently unsupported

    #TODO:
    if channel_format == "channels_first" and groups == 1:
        pytest.skip()

    class Convolution(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv = tf.keras.layers.Conv2D(
                filters=filter_size,
                kernel_size=kernel_size,
                activation="relu",
                input_shape=input_shape[1:],
                padding="same",
                groups=groups,
                data_format=channel_format,
            )

        def call(self, x1):
            return self.conv(x1)

    mod = TFModule("conv2d_tf", Convolution())
    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

input_shapes_first = [(1, 32, 64)]
input_shapes_second = [(1, 32, 128)]

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
def test_tvm_einsum(test_kind, test_device, input_shape_first, input_shape_second):
    if test_kind.is_training():
        pytest.xfail() # Backward is currently unsupported

    class Einsum(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x1, x2):
            return tf.einsum("bct,bcs->bts", x1, x2) 
    
    model = Einsum()
    mod = TFModule("Einsum", model)
    
    verify_module(
        mod,
        (input_shape_first, input_shape_second),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

def test_tvm_where(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip() # There's nothing to train

    _get_global_compiler_config().compile_depth = CompileDepth.PRE_LOWERING_PASS
    class Where(tf.keras.Model):
        def __init__(self):
            super().__init__()
        
        def call(self, cond, x, y):
            return tf.where(cond, x, y)

    model = Where()
    mod = TFModule("Where", model)
    inputs = (tf.convert_to_tensor([[True, False, False, True]]), tf.convert_to_tensor([[1, 2, 3, 4]]), tf.convert_to_tensor([[100, 200, 300, 400]]))

    verify_module(
        mod,
        ((1,4,), (1,4,), (1,4),),
        inputs=[inputs],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

input_shapes = [(1, 112, 112, 1), (1, 32, 32, 32)]

@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
def test_tvm_global_avgpool2d(training, input_shape):

    recompute = False

    compile_depth = CompileDepth.FULL

    class GlobalAvgPool2D(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.layer = tf.keras.layers.GlobalAveragePooling2D()

        def call(self, x):
            return self.layer(x)

    model = GlobalAvgPool2D()
    mod = TFModule("global_avg_pool2d_tf", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = tf.random.uniform(input_shape)

    res = pybuda_compile(
        tt0,
        "global_avg_pool2d_tf",
        act,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=compile_depth
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, res, act)

@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"input{str(s)}" for s in input_shapes]
)
def test_tvm_global_maxpool2d(input_shape):

    recompute = False

    compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS

    class GlobalMaxPool2D(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.layer = tf.keras.layers.GlobalMaxPooling2D()

        def call(self, x):
            return self.layer(x)

    model = GlobalMaxPool2D()
    mod = TFModule("global_max_pool2d_tf", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = tf.random.uniform(input_shape)

    res = pybuda_compile(
        tt0,
        "global_max_pool2d_tf",
        act,
        compiler_cfg=CompilerConfig(
            enable_training=False,
            enable_recompute=recompute,
            compile_depth=compile_depth
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, res, act)


input_shapes = [(1, 224, 224, 3)]
kernel_size = [(3, 3)]

@pytest.mark.parametrize("recompute", [False], ids=["no_recompute"])
@pytest.mark.parametrize(
    "input_shape", input_shapes, ids=[f"shape{str(s)}" for s in input_shapes]
)
@pytest.mark.parametrize(
    "kernel_size", kernel_size, ids=[f"ksize({str(k)})" for k in kernel_size]
)
@pytest.mark.parametrize(
    "channel_format", channel_format, ids=[f"channel_format({k})" for k in channel_format]
)
def test_tvm_depthwise_conv2d(training, recompute, input_shape, kernel_size, channel_format):
    
    if training:
        pytest.xfail()  # Backward is currently unsupported

    class Convolution(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                padding='same',
                input_shape=input_shape[1:],
                depth_multiplier=1,
                dilation_rate=(1,1),
                activation=tf.keras.activations.linear,
                use_bias=False,
                data_format=channel_format,
            )

        def call(self, x1):
            return self.conv(x1)
    model = Convolution()
    mod = TFModule("depthwise_conv2d_tf", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = tf.random.uniform(input_shape)

    res = pybuda_compile(
        tt0,
        "depthwise_conv2d_tf",
        act,
        compiler_cfg=CompilerConfig(
            enable_training=training,
            enable_recompute=recompute,
            compile_depth=CompileDepth.POST_INITIAL_GRAPH_PASS
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, res, act)

@pytest.mark.parametrize(
    "channel_format", channel_format, ids=[f"channel_format({k})" for k in channel_format]
)
def test_tvm_conv_uneven_padding(channel_format):

    if channel_format == "channels_first":
        pytest.skip()

    class Convolution(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                data_format=channel_format,
                dilation_rate=(1,1),
                groups=1,
                activation=tf.keras.activations.linear,
                use_bias=False,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",

            )

        def call(self, x1):
            return self.conv(x1)

    model = Convolution()
    mod = TFModule("conv2d_uneven_pad", model)

    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = tf.random.uniform((1, 224, 224, 3))

    res = pybuda_compile(
        tt0,
        "conv2d_uneven_pad",
        act,
        compiler_cfg=CompilerConfig(
            enable_training=False,
            enable_recompute=False,
            compile_depth=CompileDepth.BUDA_GRAPH_PRE_PLACER,
        ),
        verify_cfg=VerifyConfig(
            intermediates=True,
            verify_all=True,
        ),
    )
    evaluate_framework_vs_pybuda(model, res, act)

def test_multiout(test_kind, test_device):
    class MultiOut(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x1, x2):
            x1 = x1 + 7
            x2 = x2 * 9
            return x1, x2

    model = MultiOut()
    mod = TFModule("multi_out_tf", model)

    verify_module(
        mod,
        ((1, 64), (1, 64)),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )


def test_tvm_multiple_output(test_kind, test_device):
    class Linear(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(64, use_bias=True)
            self.dense2 = tf.keras.layers.Dense(32, use_bias=True)
            self.dense3 = tf.keras.layers.Dense(128, use_bias=True)

        def call(self, x):
            a = self.dense1(x)
            b = self.dense2(a)
            c = self.dense3(b)

            return a, b, c

    model = Linear()
    mod = TFModule("tf_multiple_output", model)

    verify_module(
        mod,
        ((1, 1, 32, 32),),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
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
    class CumulativeSum(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            return tf.math.cumsum(x, axis=dim, exclusive=exclusive, reverse=False, name=None)
        
    tensorflow_model = CumulativeSum()
    module = TFModule("tf_cumsum", tensorflow_model)

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


def test_tvm_floor_div(test_kind, test_device):
    class FloorDiv(tf.keras.Model):
        def __init__(self):
            super().__init__()

            self.dense = tf.keras.layers.Dense(9, use_bias=True)

        def call(self, x):
            a = tf.range(9, dtype=tf.float32)
            b = 10000 ** (2 * (a // 2) / 9)

            return self.dense(x) + b
        
    tensorflow_model = FloorDiv()
    module = TFModule("FloorDiv", tensorflow_model)

    input_shape = (1, 1, 1, 9)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


def test_tvm_logical_not(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.POST_INITIAL_GRAPH_PASS
    class LogicalNot(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            return ~x
        
    tensorflow_model = LogicalNot()
    module = TFModule("LogicalNot", tensorflow_model)

    input_shape = (1, 1, 9, 9)
    act = tf.cast(tf.zeros(input_shape), tf.bool)
    verify_module(
        module,
        (input_shape,),
        inputs = [[act]],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


def test_tvm_index(test_kind, test_device):

    _get_global_compiler_config().compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER
    class Index(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            return x[:, -1:, :]

    tensorflow_model = Index()
    module = TFModule("Index", tensorflow_model)

    input_shape = (1, 8, 8)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )


def test_list_input(training):

    class ListInputModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.linear1 = tf.keras.layers.Dense(3)
            self.linear2 = tf.keras.layers.Dense(3)
            self.linear3 = tf.keras.layers.Dense(3)
            self.linear4 = tf.keras.layers.Dense(3)
            self.smx = tf.keras.layers.Softmax(2)


        def call(self, inputs):
            x = inputs[0]
            y = inputs[1]

            x = self.linear1(x)
            x = self.linear2(x)
            y = self.linear3(y)
            y = self.linear4(y)
            return x + y

    model = ListInputModel()
    
    module = TFModule("ListInputModel", model)
    
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
   
    input_shape = (1, 1, 3)

    inputs = [(tf.random.uniform(input_shape), tf.random.uniform(input_shape))]
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    ret = pybuda_compile(
        tt0,
        "list_input",
        *inputs,
        compiler_cfg=_get_global_compiler_config(),
    )

    evaluate_framework_vs_pybuda(model, ret, *inputs)

def test_list_input_mixture(training):

    class ListInputModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.linear1 = tf.keras.layers.Dense(3)
            self.linear2 = tf.keras.layers.Dense(3)
            self.linear3 = tf.keras.layers.Dense(3)
            self.linear4 = tf.keras.layers.Dense(3)
            self.linear5 = tf.keras.layers.Dense(3)
            self.linear6 = tf.keras.layers.Dense(3)
            self.linear7 = tf.keras.layers.Dense(3)
            self.linear8 = tf.keras.layers.Dense(3)
            self.smx = tf.keras.layers.Softmax(axis=2)


        def call(self, inputs1, z, inputs2, w):
            x = inputs1[0]
            y = inputs1[1]

            a = inputs2[0]
            b = inputs2[1]
            c = inputs2[2]

            x = self.linear1(x)
            x = self.linear2(x)
            y = self.linear3(y)
            y = self.linear4(y)

            w = self.linear6(self.linear5(w))
            z = self.linear8(self.linear7(z))
            return x + y + w + z + a + b + c

    model = ListInputModel()
    
    module = TFModule("ListInputModel", model)
    
    sgd_optimizer = optimizers.SGD(learning_rate=0.5, device_params=True)
   
    input_shape = (1, 1, 3)


    # Inputs are: tensor, list[tensor], list[tensor], tensor
    inputs = [[2*tf.ones(input_shape), 3*tf.ones(input_shape)], tf.ones(input_shape), [4*tf.ones(input_shape), 5*tf.ones(input_shape), 6*tf.ones(input_shape)], 7*tf.ones(input_shape)]
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(module)

    ret = pybuda_compile(
        tt0,
        "list_input",
        *inputs,
        compiler_cfg=_get_global_compiler_config(),
    )

    evaluate_framework_vs_pybuda(model, ret, *inputs)


def test_list_input(test_kind, test_device):
    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend
    class ListInputModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.linear1 = tf.keras.layers.Dense(3)
            self.linear2 = tf.keras.layers.Dense(3)
            self.linear3 = tf.keras.layers.Dense(3)
            self.linear4 = tf.keras.layers.Dense(3)
            self.smx = tf.keras.layers.Softmax(2)


        def call(self, inputs):
            x = inputs[0]
            y = inputs[1]

            x = self.linear1(x)
            x = self.linear2(x)
            y = self.linear3(y)
            y = self.linear4(y)
            return x + y

    model = ListInputModel()
    module = TFModule("ListInputModel", model)
   
    input_shape = (1, 1, 3)
    inputs = [(tf.random.uniform(input_shape), tf.random.uniform(input_shape))]

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

    class ListInputModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.linear1 = tf.keras.layers.Dense(3)
            self.linear2 = tf.keras.layers.Dense(3)
            self.linear3 = tf.keras.layers.Dense(3)
            self.linear4 = tf.keras.layers.Dense(3)
            self.linear5 = tf.keras.layers.Dense(3)
            self.linear6 = tf.keras.layers.Dense(3)
            self.linear7 = tf.keras.layers.Dense(3)
            self.linear8 = tf.keras.layers.Dense(3)
            self.smx = tf.keras.layers.Softmax(axis=2)


        def call(self, inputs1, z, inputs2, w):
            x = inputs1[0]
            y = inputs1[1]

            a = inputs2[0]
            b = inputs2[1]
            c = inputs2[2]

            x = self.linear1(x)
            x = self.linear2(x)
            y = self.linear3(y)
            y = self.linear4(y)

            w = self.linear6(self.linear5(w))
            z = self.linear8(self.linear7(z))
            return x + y + w + z + a + b + c

    model = ListInputModel()
    module = TFModule("ListInputModel", model)

    input_shape = (1, 1, 3)
    # Inputs are: tensor, list[tensor], list[tensor], tensor
    inputs = [[2*tf.ones(input_shape), 3*tf.ones(input_shape)], tf.ones(input_shape), [4*tf.ones(input_shape), 5*tf.ones(input_shape), 6*tf.ones(input_shape)], 7*tf.ones(input_shape)]
    
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

    class DictInputModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.linear1 = tf.keras.layers.Dense(3)
            self.linear2 = tf.keras.layers.Dense(3)
            self.linear3 = tf.keras.layers.Dense(3)
            self.linear4 = tf.keras.layers.Dense(3)
            self.linear5 = tf.keras.layers.Dense(3)
            self.linear6 = tf.keras.layers.Dense(3)
            self.smx = tf.keras.layers.Softmax(axis=2)


        def call(self, inputs, x, z):
            q = inputs['x']
            y = inputs['y']

            a = x[0]
            b = x[1]

            q = self.linear1(q)
            q = self.linear2(q)
            y = self.linear3(y)
            y = self.linear4(y)

            z = self.linear5(self.linear6(z))
            return q + y + z + a + b

    model = DictInputModel()
    module = TFModule("DictInputModel", model)
   
    input_shape = (1, 1, 3)
    inputs = [{'x':tf.random.uniform(input_shape), 'y': tf.random.uniform(input_shape)}, [tf.random.uniform(input_shape), tf.random.uniform(input_shape)], tf.random.uniform(input_shape)]
    
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


@pytest.mark.parametrize("filter_size", [1, 3,])
@pytest.mark.parametrize("kernel_size", [3, 9,])
@pytest.mark.parametrize("stride", [1, 2,])
@pytest.mark.parametrize("groups", [1,])
@pytest.mark.parametrize("padding", ["valid", "same", ])
def test_tvm_conv1d(test_kind, test_device, filter_size, kernel_size, stride, groups, padding):
    if test_kind in (TestKind.TRAINING, TestKind.TRAINING_RECOMPUTE,) :  # only run recompute test in post-commit
        pytest.skip()

    # Broadcast on dimensions beyond 3rd is not supported
    compiler_config = _get_global_compiler_config()
    compiler_config.compile_depth = CompileDepth.GENERATE_INITIAL_GRAPH

    class Conv1d(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv = tf.keras.layers.Conv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                data_format='channels_last',
                dilation_rate=1,
                groups=groups,
                use_bias=True,
                bias_initializer="he_normal",                
            )

        def call(self, x):
            return self.conv(x)

    module = TFModule("conv1d_tf", Conv1d())
    input_shape = (1, 224, 224, 3)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind
        )
    )

def test_tvm_tf_erf_gelu(test_kind, test_device):

    if test_kind == TestKind.TRAINING:
        pytest.skip()


    class TFErfGeluModule(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            return tf.keras.activations.gelu(x)

    framework_model = TFErfGeluModule()
    module = TFModule("TFErfGeluModule", framework_model)

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

def test_tvm_tf_tanh_gelu(test_kind, test_device):

    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class TFTanhGeluModule(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            return tf.keras.activations.gelu(x, approximate=True)

    framework_model = TFTanhGeluModule()
    module = TFModule("TFTanhGeluModule", framework_model)

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
