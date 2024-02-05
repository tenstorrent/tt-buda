# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Padding Pass Tests
#
import os

import torch
import pytest

import pybuda
from pybuda import (
    TTDevice,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
)
from pybuda._C.backend_api import BackendType


class TestPaddingPassK(pybuda.PyBudaModule):

    # Convolutional Network
    
    def __init__(
        self, 
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features
    ):
        super().__init__(name)

        # Get from test
        self.name = name
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_features
        self.out_features = out_features
        
        # Fixed for particular test
        self.dilation = 1
        self.groups = 1
        self.bias = False
        self.padding = "same"

        # Note: kernel and stride depend on convolution

        # Auxiliary function that creates convolutional layer
        def conv2d(name, kernel, stride, padding=None, in_channels=None):
            return pybuda.op.nn.Conv2dModule(
                        name=f"{self.name}.{name}",
                        in_channels=in_channels if in_channels is not None else self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding if padding is not None else self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=self.bias,
                    )

        # Auxiliary function that creates maxpool layer
        def maxpool2d(name, kernel, stride):
            return pybuda.op.nn.MaxPool2dModule(
                        name=f"{self.name}.{name}",
                        kernel_size=kernel,
                        stride=stride,
                    )

        # Auxiliary function that creates FC layer(Linear) layer
        # We use this layer at the end of the CNN before Softmax
        def linear(name):
            return pybuda.op.nn.Linear(
                        name=f"{self.name}.{name}",
                        in_features=self.in_features,
                        out_features=self.out_features,
                        bias=False,
                    )

        # head
        # ... head flow 1 ...
        self.conv1 = conv2d(name="conv1", kernel=self.kernel, stride=2, in_channels=3)
        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)
        # ... head flow 2 ...
        self.conv2 = conv2d(name="conv2", kernel=self.kernel, stride=2, in_channels=3)
        self.maxpool2 = maxpool2d(name="maxpool2", kernel=3, stride=2)

        # block
        self.conv3 = conv2d(name="conv3", kernel=(3, 3), stride=1)
        self.conv4 = conv2d(name="conv4", kernel=(3, 3), stride=1)
        self.conv5 = conv2d(name="conv5", kernel=(3, 3), stride=1)

        # tail
        self.linear = linear("linear")

    def forward(self, x1, x2):
        
        # head
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x2)
        maxpool1 = self.maxpool1(conv1)
        maxpool2 = self.maxpool2(conv2)
        add1 = pybuda.op.Add("add1", maxpool1, maxpool2)

        # block
        relu1 = pybuda.op.Relu("relu1", add1)
        conv3 = self.conv3(relu1)
        relu2 = pybuda.op.Relu("relu2", conv3)
        conv4 = self.conv4(relu2)
        relu3 = pybuda.op.Relu("relu3", conv4)
        conv5 = self.conv5(relu3)
        add2 = pybuda.op.Add("add2", relu1, conv5)

        # tail
        W, Z, R, C = 1, 1, add2.shape[-3], add2.shape[-1] * add2.shape[-2]
        resh = pybuda.op.Reshape("resh", add2, (W, Z, R, C))
        tr = pybuda.op.Transpose("tr", resh, -1, -2)
        ra = pybuda.op.ReduceAvg("ra", tr, -2)
        lin = self.linear(ra) 
        sm = pybuda.op.Softmax("sm", lin, dim=-1, stable=True)

        return sm


class TestPaddingPassK_1(pybuda.PyBudaModule):

    def __init__(
        self,
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features
    ):
        super().__init__(name)

        # Get from test
        self.name = name
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_features
        self.out_features = out_features
        
        # Fixed for particular test
        self.dilation = 1
        self.groups = 1
        self.bias = False
        self.padding = "same"

        # Note: kernel and stride depend on convolution

        # Auxiliary function that creates convolutional layer
        def conv2d(name, kernel, stride, padding=None, in_channels=None):
            return pybuda.op.nn.Conv2dModule(
                        name=f"{self.name}.{name}",
                        in_channels=in_channels if in_channels is not None else self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding if padding is not None else self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=self.bias,
                    )

        # Auxiliary function that creates maxpool layer
        def maxpool2d(name, kernel, stride):
            return pybuda.op.nn.MaxPool2dModule(
                        name=f"{self.name}.{name}",
                        kernel_size=kernel,
                        stride=stride,
                    )

        # Auxiliary function that creates FC layer(Linear) layer
        # We use this layer at the end of the CNN before Softmax
        def linear(name):
            return pybuda.op.nn.Linear(
                        name=f"{self.name}.{name}",
                        in_features=self.in_features,
                        out_features=self.out_features,
                        bias=False,
                    )

        # head
        # ... head flow 1 ...
        self.conv1 = conv2d(name="conv1", kernel=self.kernel, stride=2, in_channels=3)
        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)
        # ... head flow 2 ...
        self.conv2 = conv2d(name="conv2", kernel=self.kernel, stride=2, in_channels=3)
        self.maxpool2 = maxpool2d(name="maxpool2", kernel=3, stride=2)

        # tail
        self.linear = linear("linear")

    def forward(self, x):

        # pre-head
        exp = pybuda.op.Exp("exp", x)
        relu = pybuda.op.Relu("relu", x)

        mul = pybuda.op.Multiply("mul", exp, relu)
        sub = pybuda.op.Multiply("sub", exp, relu)
        
        # head
        conv1 = self.conv1(mul)
        conv2 = self.conv2(sub)
        maxpool1 = self.maxpool1(conv1)
        maxpool2 = self.maxpool2(conv2)
        add1 = pybuda.op.Add("add1", maxpool1, maxpool2)

        # tail
        W, Z, R, C = 1, 1, add1.shape[-3], add1.shape[-1] * add1.shape[-2]
        resh = pybuda.op.Reshape("resh", add1, (W, Z, R, C))
        tr = pybuda.op.Transpose("tr", resh, -1, -2)
        ra = pybuda.op.ReduceAvg("ra", tr, -2)
        lin = self.linear(ra) 
        sm = pybuda.op.Softmax("sm", lin, dim=-1, stable=True)

        return sm

class TestPaddingPassK_2(pybuda.PyBudaModule):

    def __init__(
        self,
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features
    ):
        super().__init__(name)

        # Get from test
        self.name = name
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_features
        self.out_features = out_features
        
        # Fixed for particular test
        self.dilation = 1
        self.groups = 1
        self.bias = False
        self.padding = "same"

        # Note: kernel and stride depend on convolution

        # Auxiliary function that creates convolutional layer
        def conv2d(name, kernel, stride, padding=None, in_channels=None):
            return pybuda.op.nn.Conv2dModule(
                        name=f"{self.name}.{name}",
                        in_channels=in_channels if in_channels is not None else self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding if padding is not None else self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=self.bias,
                    )

        # Auxiliary function that creates maxpool layer
        def maxpool2d(name, kernel, stride):
            return pybuda.op.nn.MaxPool2dModule(
                        name=f"{self.name}.{name}",
                        kernel_size=kernel,
                        stride=stride,
                    )

        # Auxiliary function that creates FC layer(Linear) layer
        # We use this layer at the end of the CNN before Softmax
        def linear(name):
            return pybuda.op.nn.Linear(
                        name=f"{self.name}.{name}",
                        in_features=self.in_features,
                        out_features=self.out_features,
                        bias=False,
                    )

        # head
        self.conv1 = conv2d(name="conv1", kernel=self.kernel, stride=2, in_channels=3)
        self.maxpool1 = maxpool2d(name="maxpool1", kernel=3, stride=2)

        # tail
        self.linear = linear("linear")

    def forward(self, x):
        
        # head
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)

        # tail
        W, Z, R, C = 1, 1, maxpool1.shape[-3], maxpool1.shape[-1] * maxpool1.shape[-2]
        resh = pybuda.op.Reshape("resh", maxpool1, (W, Z, R, C))
        tr = pybuda.op.Transpose("tr", resh, -1, -2)
        ra = pybuda.op.ReduceAvg("ra", tr, -2)
        lin = self.linear(ra) 
        sm = pybuda.op.Softmax("sm", lin, dim=-1, stable=True)

        return sm

class TestPaddingPassK_3(pybuda.PyBudaModule):

    def __init__(
        self,
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features
    ):
        super().__init__(name)

        # Get from test
        self.name = name
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_features
        self.out_features = out_features
        
        # Fixed for particular test
        self.dilation = 1
        self.groups = 1
        self.bias = False
        self.padding = "same"

        # Note: kernel and stride depend on convolution

        # Auxiliary function that creates convolutional layer
        def conv2d(name, kernel, stride, padding=None, in_channels=None):
            return pybuda.op.nn.Conv2dModule(
                        name=f"{self.name}.{name}",
                        in_channels=in_channels if in_channels is not None else self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding if padding is not None else self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=self.bias,
                    )

        # head
        self.conv1 = conv2d(name="conv1", kernel=self.kernel, stride=2, in_channels=3)

    def forward(self, x):
        
        exp = pybuda.op.Exp("exp", x)
        conv = self.conv1(exp)

        return conv

class TestPaddingPassK_4(pybuda.PyBudaModule):

    def __init__(
        self,
        name,
        kernel,
        in_channels,
        out_channels,
        in_features,
        out_features
    ):
        super().__init__(name)

        # Get from test
        self.name = name
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_features
        self.out_features = out_features
        
        # Fixed for particular test
        self.dilation = 1
        self.groups = 1
        self.bias = False
        self.padding = "same"

        # Note: kernel and stride depend on convolution

        # Auxiliary function that creates convolutional layer
        def conv2d(name, kernel, stride, padding=None, in_channels=None):
            return pybuda.op.nn.Conv2dModule(
                        name=f"{self.name}.{name}",
                        in_channels=in_channels if in_channels is not None else self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding if padding is not None else self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=self.bias,
                    )

        # head
        self.conv1 = conv2d(name="conv1", kernel=self.kernel, stride=2, in_channels=3)

    def forward(self, x):
        
        conv = self.conv1(x)

        return conv


TEST_K_DISABLE_PADDING_PASS_FLAG = True
TEST_K_PRINT_GRAPH_VIZ_FLAG = False
TEST_K_PRINT_GRAPH_AT_FLAG = False
TEST_K_FRACTURING_FLAG = False

TEST_K_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = False
TEST_K_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = True
TEST_K_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = False
TEST_K_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_K_SHAPE_SIZE_1_FLAG = True
TEST_K_SHAPE_SIZE_2_FLAG = True
TEST_K_SHAPE_SIZE_3_FLAG = False
TEST_K_SHAPE_SIZE_4_FLAG = False

TEST_K_KERNEL_SQUARED_ODD = True
TEST_K_KERNEL_SQUARED_EVEN = False

TEST_K_MODEL_K_FLAG = False
TEST_K_MODEL_K_1_FLAG = False
TEST_K_MODEL_K_2_FLAG = False
TEST_K_MODEL_K_3_FLAG = False
TEST_K_MODEL_K_4_FLAG = False

TEST_K_LOGGER_LEVEL_TRACE = False
TEST_K_LOGGER_LEVEL_DEBUG = False

TEST_K_INPUT_NO = 1


def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_K_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"

    # Environment variable that allows printing a graph
    if TEST_K_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_K_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_K_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_K_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_K_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_K_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_K_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
        os.environ["PYBUDA_LEGALIZER_DEBUG_NODE_NAME"] = ""



# The main reason why we use shapes of different sizes is
# because convolutional neural networks can't always work with big shapes.

# Each size category is bigger than previous one 
# for one or half order of magnitude

# Shapes -> Size 1
shapes_size_1 = [
    # (12, 12),
    # (16, 16),
    # (17, 17),
    # (32, 32),
    # (37, 37),
    (64, 64),
    (73, 73),
    (80, 80),
    (96, 96)
]

# Shapes -> Size 2
shapes_size_2 = [
    (128, 128), 
    (192, 192), 
    (211, 211), 
    (212, 212),
    (224, 224)
]

# Shapes -> Size 3
shapes_size_3 = [
    (541, 541),
    (977, 977),
    (768, 768),
    (1024, 1024)
]

# Shapes -> Size 4
shapes_size_4 = [
    (2048, 2048),
    (3000, 3000),
    (4096, 4096)
]

original_shape = []
if TEST_K_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_K_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_K_SHAPE_SIZE_3_FLAG:
    original_shape += shapes_size_3
if TEST_K_SHAPE_SIZE_4_FLAG:
    original_shape += shapes_size_4

original_shape_ids = [
    f"shape={'x'.join([str(jtem) for jtem in item])}" for item in original_shape
]

kernel_squared_odd = [
    # (1, 1), 
    (3, 3), 
    (5, 5), 
    # (7, 7)
]
kernel_squared_even = [(2, 2), (4, 4), (6, 6)]

kernels = []
if TEST_K_KERNEL_SQUARED_ODD:
    kernels += kernel_squared_odd
if TEST_K_KERNEL_SQUARED_EVEN:
    kernels += kernel_squared_even

kernel_ids = [f"kernel={'x'.join([str(jtem) for jtem in item])}" for item in kernels]

test_model = []
if TEST_K_MODEL_K_FLAG:
    test_model.append(("TestPaddingPassK", TestPaddingPassK))
if TEST_K_MODEL_K_1_FLAG:
    test_model.append(("TestPaddingPassK_1", TestPaddingPassK_1))
if TEST_K_MODEL_K_2_FLAG:
    test_model.append(("TestPaddingPassK_2", TestPaddingPassK_2))
if TEST_K_MODEL_K_3_FLAG:
    test_model.append(("TestPaddingPassK_3", TestPaddingPassK_3))
if TEST_K_MODEL_K_4_FLAG:
    test_model.append(("TestPaddingPassK_4", TestPaddingPassK_4))


# @pytest.mark.xfail
@pytest.mark.parametrize("test_model", test_model, ids=[item[0] for item in test_model])
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
@pytest.mark.parametrize("out_features", [10])
@pytest.mark.parametrize("in_features", [32])
@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("kernel", kernels, ids=kernel_ids)
def test_padding_pass_k(
    test_kind,
    test_device,
    in_channels,
    out_channels,
    in_features,
    out_features,
    original_shape,
    test_model, 
    kernel
):

    if test_kind.is_training():
        pytest.skip()
        pass_name = "training"
    else:
        pass_name = "inference"

    set_environment()

    test_name = f"{test_model[0]}_{original_shape[0]}_{original_shape[1]}_{pass_name}"
    model = test_model[1](
        name=test_name,
        kernel=kernel,
        in_channels=in_channels,
        out_channels=out_channels,
        in_features=in_features,
        out_features=out_features
    )

    if test_kind.is_training() and test_device.devtype == BackendType.Silicon:
        relative_atol = 0.3
    else: 
        relative_atol = 0.1

    if test_device.devtype == BackendType.Silicon:
        pcc = 0.96
    else:
        pcc = 0.99

    # activations = ((torch.rand((1, 3, original_shape[0], original_shape[1])) + 0.00001) * 100000).detach()
    act_shape = (1, in_channels, original_shape[0], original_shape[1])

    compiler_cfg = CompilerConfig(
        enable_training=test_kind.is_training()
    )
    verify_cfg = VerifyConfig(
        pcc=pcc,
        relative_atol=relative_atol
    )
    
    inputs = [Tensor.create_from_torch(torch.rand(act_shape)) for _ in range(TEST_K_INPUT_NO)]

    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *inputs, 
        compiler_cfg=compiler_cfg, 
        verify_cfg=verify_cfg
    )