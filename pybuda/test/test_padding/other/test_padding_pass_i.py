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

class TestPaddingPassI(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on only matmul operation.

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(3)]

        self.set_parameter("train_param1", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param2", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param3", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        mm1 = pybuda.op.Matmul("mm1", x1, self.train_param1)
        mm2 = pybuda.op.Matmul("mm2", x2, self.train_param2)
        mm3 = pybuda.op.Matmul("mm3", x3, self.train_param3)

        # Layer 3
        mm4 = pybuda.op.Matmul("mm4", mm1, x2)
        mm5 = pybuda.op.Matmul("mm5", self.train_param1, mm2)
        mm6 = pybuda.op.Matmul("mm6", mm2, self.train_param3)
        mm7 = pybuda.op.Matmul("mm7", self.train_param2, mm3)

        # Layer 4
        mm8 = pybuda.op.Matmul("mm8", mm4, mm5)
        mm9 = pybuda.op.Matmul("mm9", mm6, mm7)

        # Layer 5
        mm10 = pybuda.op.Matmul("mm10", mm8, mm2)
        mm11 = pybuda.op.Matmul("mm11", mm2, mm9)

        # Layer 6
        mm12 = pybuda.op.Matmul("mm12", mm4, mm10)
        mm13 = pybuda.op.Matmul("mm13", mm10, mm11)
        mm14 = pybuda.op.Matmul("mm14", mm11, mm7)

        # Layer 7
        mm15 = pybuda.op.Matmul("mm15", mm12, mm13)
        mm16 = pybuda.op.Matmul("mm16", mm13, mm14)

        return mm15, mm16


class TestPaddingPassI_1(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on matmul operation and eltwise operations.

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(2)]

        self.set_parameter("train_param1", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param2", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mm1 = pybuda.op.Matmul("mm1", x1, self.train_param1)
        mm2 = pybuda.op.Matmul("mm2", x2, self.train_param2)

        # Layer 3
        mm3 = pybuda.op.Matmul("mm3", mm1, mm2)
        
        return mm3


class TestPaddingPassI_2(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on matmul operation and eltwise operations.

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)
        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]

        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        mm = pybuda.op.Matmul("mm", x, self.train_param)

        return mm


class TestPaddingPassI_3(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on matmul operation and eltwise operations.

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)
        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]

        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        mm1 = pybuda.op.Matmul("mm1", x, self.train_param)
        mm2 = pybuda.op.Matmul("mm2", x, self.train_param)
        mm3 = pybuda.op.Matmul("mm3", x, self.train_param)

        # Layer 3
        mm4 = pybuda.op.Matmul("mm4", mm1, mm2)
        mm5 = pybuda.op.Matmul("mm5", mm2, mm3)

        # Layer 4
        mm6 = pybuda.op.Matmul("mm6", mm1, mm4)
        mm7 = pybuda.op.Matmul("mm7", mm5, mm3)

        # Layer 5
        mm8 = pybuda.op.Matmul("mm8", mm6, mm7)

        return mm8


class TestPaddingPassI_4(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on matmul operation and eltwise operations.

    def __init__(self, name: str, shape: list):
        super().__init__(name)
        self.shape = shape
        self.train_param = pybuda.Parameter(*self.shape, requires_grad=True)
        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]

        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        mm1 = pybuda.op.Matmul("mm1", x, self.train_param)
        mm2 = pybuda.op.Matmul("mm2", x, self.train_param)

        mm3 = pybuda.op.Matmul("mm3", mm1, mm2)

        return mm3


TEST_I_DISABLE_PADDING_PASS_FLAG = False
TEST_I_PADDING_PASS_ELEMENT_WISE_FLAG = True
TEST_I_PADDING_PASS_MATMUL_FLAG = True

TEST_I_VERIFY_ALL_FLAG = True
TEST_I_PRINT_GRAPH_VIZ_FLAG = False
TEST_I_PRINT_GRAPH_AT_FLAG = False
TEST_I_FRACTURING_FLAG = False

TEST_I_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = True
TEST_I_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = True
TEST_I_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = True
TEST_I_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_I_SHAPE_SIZE_1_FLAG = True
TEST_I_SHAPE_SIZE_2_FLAG = True
TEST_I_SHAPE_SIZE_3_FLAG = False
TEST_I_SHAPE_SIZE_4_FLAG = False

TEST_I_MODEL_I_FLAG = False
TEST_I_MODEL_I_1_FLAG = True
TEST_I_MODEL_I_2_FLAG = True
TEST_I_MODEL_I_3_FLAG = False
TEST_I_MODEL_I_4_FLAG = True

TEST_I_LOGGER_LEVEL_TRACE = False
TEST_I_LOGGER_LEVEL_DEBUG = False


def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_I_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
    if TEST_I_PADDING_PASS_ELEMENT_WISE_FLAG:
        os.environ["PYBUDA_PADDING_PASS_ELEMENT_WISE"] = "1"
    if TEST_I_PADDING_PASS_MATMUL_FLAG:
        os.environ["PYBUDA_PADDING_PASS_MATMUL"] = "1"

    # Environment variable that allows printing a graph
    if TEST_I_VERIFY_ALL_FLAG:
        os.environ["PYBUDA_FORCE_VERIFY_ALL"] = "1"
    # Environment variable that allows printing a graph
    if TEST_I_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_I_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_I_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_I_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_I_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_I_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_I_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG:
        os.environ["PYBUDA_GRAPH_SOLVER_SELF_CUT_TYPE"] = "FastCut"
    if TEST_I_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
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

if TEST_I_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_I_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_I_SHAPE_SIZE_3_FLAG:
    original_shape += shapes_size_3
if TEST_I_SHAPE_SIZE_4_FLAG:
    original_shape += shapes_size_4

original_shape_ids = [
    f"shape={'x'.join([str(jtem) for jtem in item])}" for item in original_shape
]

test_model = []

if TEST_I_MODEL_I_FLAG:
    test_model.append(("TestPaddingPassI", TestPaddingPassI))
if TEST_I_MODEL_I_1_FLAG:
    test_model.append(("TestPaddingPassI_1", TestPaddingPassI_1))
if TEST_I_MODEL_I_2_FLAG:
    test_model.append(("TestPaddingPassI_2", TestPaddingPassI_2))
if TEST_I_MODEL_I_3_FLAG:
    test_model.append(("TestPaddingPassI_3", TestPaddingPassI_3))
if TEST_I_MODEL_I_4_FLAG:
    test_model.append(("TestPaddingPassI_4", TestPaddingPassI_4))


# @pytest.mark.xfail
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
@pytest.mark.parametrize("test_model", test_model, ids=[item[0] for item in test_model])
def test_padding_pass_i(
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

    test_name = f"{test_model[0]}_{original_shape[0]}_{original_shape[1]}_{pass_name}"
    model = test_model[1](name=test_name, shape=original_shape)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
            enable_training=test_kind.is_training()
        ), 
        verify_cfg=VerifyConfig()
    )