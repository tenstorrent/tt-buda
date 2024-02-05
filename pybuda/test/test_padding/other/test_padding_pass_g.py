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
from pybuda._C import DataFormat


class TestPaddingPassG(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on element-wise operation.

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):
        
        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x1, self.train_param2)
        mul3 = pybuda.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        add1 = pybuda.op.Add("add1", mul1, mul2)
        add2 = pybuda.op.Add("add2", mul2, mul3)
        add3 = pybuda.op.Add("add3", mul2, mul3)

        # Layer 4
        mul4 = pybuda.op.Multiply("mul4", add1, add2)
        mul5 = pybuda.op.Multiply("mul5", add1, add2)
        mul6 = pybuda.op.Multiply("mul6", add2, add3)

        # Layer 5
        exp1 = pybuda.op.Exp("exp1", mul4)
        exp2 = pybuda.op.Exp("exp2", mul5)
        exp3 = pybuda.op.Exp("exp3", mul6)

        # Layer 6
        relu1 = pybuda.op.Relu("relu1", exp1)
        relu2 = pybuda.op.Relu("relu2", exp2)
        relu3 = pybuda.op.Relu("relu3", exp3)

        # Layer 7
        exp4 = pybuda.op.Exp("exp4", relu1)
        exp5 = pybuda.op.Exp("exp5", relu2)
        exp6 = pybuda.op.Exp("exp6", relu3)

        return exp4, exp5, exp6


class TestPaddingPassG_1(pybuda.PyBudaModule):

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):
        
        # Layer 2
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)

        # Layer 3
        add = pybuda.op.Add("add", mul1, mul2)

        # Layer 4
        exp = pybuda.op.Exp("exp", add)

        return exp

class TestPaddingPassG_2(pybuda.PyBudaModule):

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):
        
        # Layer 2
        add = pybuda.op.Add("add", x1, x2)
        # Layer 3
        mul1 = pybuda.op.Multiply("mul1", add, self.train_param1)
        # Layer 4
        mul2 = pybuda.op.Multiply("mul2", mul1, self.train_param2)
        # Layer 5
        recip = pybuda.op.Reciprocal("recip", mul2)

        return recip


class TestPaddingPassG_3(pybuda.PyBudaModule):

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):
        
        # Layer 2
        add1 = pybuda.op.Add("add1", x1, x2)
        add2 = pybuda.op.Add("add2", self.train_param1, self.train_param2)

        # Layer 3
        mul = pybuda.op.Multiply("mul", add1, add2)

        return mul


TEST_G_DISABLE_PADDING_PASS_FLAG = False
TEST_G_PADDING_PASS_ELEMENT_WISE_FLAG = True
TEST_G_PADDING_PASS_MATMUL_FLAG = True

TEST_G_VERIFY_ALL_FLAG = True
TEST_G_PRINT_GRAPH_VIZ_FLAG = False
TEST_G_PRINT_GRAPH_AT_FLAG = False
TEST_G_FRACTURING_FLAG = False

TEST_G_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = True
TEST_G_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = True
TEST_G_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = True
TEST_G_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_G_MODEL_G_FLAG = False
TEST_G_MODEL_G_1_FLAG = True
TEST_G_MODEL_G_2_FLAG = True
TEST_G_MODEL_G_3_FLAG = True

TEST_G_SHAPE_SIZE_1_FLAG = True
TEST_G_SHAPE_SIZE_2_FLAG = True
TEST_G_SHAPE_SIZE_3_FLAG = False
TEST_G_SHAPE_SIZE_4_FLAG = False

TEST_G_LOGGER_LEVEL_TRACE = False
TEST_G_LOGGER_LEVEL_DEBUG = False

TEST_G_INPUT_NO = 2


def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_G_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
    if TEST_G_PADDING_PASS_ELEMENT_WISE_FLAG:
        os.environ["PYBUDA_PADDING_PASS_ELEMENT_WISE"] = "1"
    if TEST_G_PADDING_PASS_MATMUL_FLAG:
        os.environ["PYBUDA_PADDING_PASS_MATMUL"] = "1"

    # Environment variable that allows printing a graph
    if TEST_G_VERIFY_ALL_FLAG:
        os.environ["PYBUDA_FORCE_VERIFY_ALL"] = "1"
    # Environment variable that allows printing a graph
    if TEST_G_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_G_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_G_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_G_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_G_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_G_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_G_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
        os.environ["PYBUDA_LEGALIZER_DEBUG_NODE_NAME"] = ""



# The main reason why we use shapes of different sizes is
# because convolutional neural networks can't always work with big shapes.

# Each size category is bigger than previous one 
# for one or half order of magnitude

# Shapes -> Size 1
shapes_size_1 = [
    (12, 12),
    (16, 16),
    (17, 17),
    (73, 73),
    (64, 64),
    (32, 32),
    (37, 37)
]

# Shapes -> Size 2
shapes_size_2 = [
    (128, 128), 
    (192, 192), 
    (211, 211), 
    (212, 212),
    (224, 224),
    (337, 337),
    (359, 359)
]

# Shapes -> Size 3
shapes_size_3 = [
    (503, 503),
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

if TEST_G_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_G_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_G_SHAPE_SIZE_3_FLAG:
    original_shape += shapes_size_3
if TEST_G_SHAPE_SIZE_4_FLAG:
    original_shape += shapes_size_4

original_shape_ids = [
    f"shape={'x'.join([str(jtem) for jtem in item])}" for item in original_shape
]

test_model = []

if TEST_G_MODEL_G_FLAG:
    test_model.append(("TestPaddingPassG", TestPaddingPassG))
if TEST_G_MODEL_G_1_FLAG:
    test_model.append(("TestPaddingPassG_1", TestPaddingPassG_1))
if TEST_G_MODEL_G_2_FLAG:
    test_model.append(("TestPaddingPassG_2", TestPaddingPassG_2))
if TEST_G_MODEL_G_3_FLAG:
    test_model.append(("TestPaddingPassG_3", TestPaddingPassG_3))

@pytest.mark.parametrize("test_model", test_model, ids=[item[0] for item in test_model])
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
def test_padding_pass_g(
    test_kind,
    test_device,
    original_shape,
    test_model
):
    if test_device.arch == pybuda.BackendDevice.Wormhole_B0:
        pytest.skip("Skip until #731 is solved")

    if test_kind.is_training():
        pass_name = "training"
    else:
        pass_name = "inference"

    set_environment()

    inputs = [Tensor.create_from_torch(torch.rand(original_shape)) for _ in range(TEST_G_INPUT_NO)]

    test_name = f"{test_model[0]}_{original_shape[0]}_{original_shape[1]}_{pass_name}"
    model = test_model[1](name=test_name, shape=original_shape)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *inputs, 
        compiler_cfg=CompilerConfig(
            enable_training=test_kind.is_training()
        ), 
        verify_cfg=VerifyConfig()
    )