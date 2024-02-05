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

from pybuda._C.balancer import OpOverride
from pybuda._C import DataFormat

class TestPaddingPassH(pybuda.PyBudaModule):

    # Testing padding/unpadding pass.
    # This test is based on matmul operation and eltwise operations.

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
        mul1 = pybuda.op.Multiply("mul1", mm1, x2)
        mul2 = pybuda.op.Multiply("mul2", self.train_param1, mm2)
        mul3 = pybuda.op.Multiply("mul3", mm2, self.train_param3)
        mul4 = pybuda.op.Multiply("mul4", self.train_param2, mm3)

        # Layer 4
        mm4 = pybuda.op.Matmul("mm4", mul1, mul2)
        mm5 = pybuda.op.Matmul("mm5", mul3, mul4)

        # Layer 5
        mul5 = pybuda.op.Multiply("mul5", mm4, mm2)
        mul6 = pybuda.op.Multiply("mul6", mm2, mm5)

        # Layer 6
        mm6 = pybuda.op.Matmul("mm6", mul1, mul5)
        mm7 = pybuda.op.Matmul("mm7", mul6, mul4)
        add = pybuda.op.Add("add", mul5, mul6)

        # Layer 7
        mul7 = pybuda.op.Multiply("mul7", mm6, add)
        mul8 = pybuda.op.Multiply("mul8", add, mm7)

        return mul7, mul8


class TestPaddingPassH_1(pybuda.PyBudaModule):

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
        mul1 = pybuda.op.Multiply("mul1", x1, self.train_param1)
        mul2 = pybuda.op.Multiply("mul2", x2, self.train_param2)
        mm1 = pybuda.op.Matmul("mm1", self.train_param1, x2)

        # Layer 3
        mm2 = pybuda.op.Matmul("mm2", mul1, mm1)
        mm3 = pybuda.op.Matmul("mm3", mm1, mul2)

        # Layer 4
        mm4 = pybuda.op.Matmul("mm4", mm2, mm3)
        
        return mm4


class TestPaddingPassH_2(pybuda.PyBudaModule):

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

        # Layer 3
        exp = pybuda.op.Exp("exp", mm)

        return exp


class TestPaddingPassH_3(pybuda.PyBudaModule):

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
        mul1 = pybuda.op.Multiply("mul1", mm1, mm4)
        mul2 = pybuda.op.Multiply("mul2", mm5, mm3)

        # Layer 5
        mm6 = pybuda.op.Matmul("mm6", mul1, mul2)

        return mm6


TEST_H_DISABLE_PADDING_PASS_FLAG = False
TEST_H_PADDING_PASS_ELEMENT_WISE_FLAG = True
TEST_H_PADDING_PASS_MATMUL_FLAG = True
TEST_H_PADDING_PASS_BUFFER_QUEUE_FLAG = False
TEST_E_PADDING_PASS_DISABLE_BUDA_OP_FLAG = False

TEST_H_VERIFY_ALL_FLAG = True
TEST_H_PRINT_GRAPH_VIZ_FLAG = False
TEST_H_PRINT_GRAPH_AT_FLAG = False
TEST_H_FRACTURING_FLAG = False

TEST_H_CHIP_PLACEMENT_FORCE_INTERMED_FLAG = True
TEST_H_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG = True
TEST_H_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG = True
TEST_H_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG_FASTCUT = True
TEST_H_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG_PRODUCER = False
TEST_H_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG_CONSUMER = False
TEST_H_CHIP_PLACEMENT_LEGALIZER_NODE_NAME = False

TEST_H_SHAPE_SIZE_1_FLAG = False
TEST_H_SHAPE_SIZE_2_FLAG = False
TEST_H_SHAPE_SIZE_3_FLAG = False
TEST_H_SHAPE_SIZE_4_FLAG = False

TEST_H_MODEL_H_FLAG = False
TEST_H_MODEL_H_1_FLAG = False
TEST_H_MODEL_H_2_FLAG = False
TEST_H_MODEL_H_3_FLAG = False

TEST_H_LOGGER_LEVEL_TRACE = False
TEST_H_LOGGER_LEVEL_DEBUG = False



def set_environment():

    """
    This function set all environment variables used in the tests.
    """

    # Environment variable that adds padding pass
    if TEST_H_DISABLE_PADDING_PASS_FLAG:
        os.environ["PYBUDA_DISABLE_PADDING_PASS"] = "1"
    if TEST_H_PADDING_PASS_ELEMENT_WISE_FLAG:
        os.environ["PYBUDA_PADDING_PASS_ELEMENT_WISE"] = "1"
    if TEST_H_PADDING_PASS_MATMUL_FLAG:
        os.environ["PYBUDA_PADDING_PASS_MATMUL"] = "1"
    if TEST_H_PADDING_PASS_BUFFER_QUEUE_FLAG:
        os.environ["PYBUDA_PADDING_PASS_BUFFER_QUEUE"] = "1"

    # Environment variable that allows printing a graph
    if TEST_H_VERIFY_ALL_FLAG:
        os.environ["PYBUDA_FORCE_VERIFY_ALL"] = "1"
    # Environment variable that allows printing a graph
    if TEST_H_PRINT_GRAPH_VIZ_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_VIZ_FORMAT_DIR"] = "ALL"
    if TEST_H_PRINT_GRAPH_AT_FLAG:
        os.environ["PYBUDA_PRINT_GRAPH_AT"] = "ALL"

    # Environment variable that allows fracturing
    if TEST_H_FRACTURING_FLAG:
        os.environ["PYBUDA_FRACTURIZATION_DISABLE"] = "1"

    # Include or not environment variables for debugging the stack
    if TEST_H_LOGGER_LEVEL_TRACE:
        os.environ["LOGGER_LEVEL"] = "TRACE"
    if TEST_H_LOGGER_LEVEL_DEBUG:
        os.environ["LOGGER_LEVEL"] = "DEBUG"

    # Include or not environment variables for debugging chip placement module
    if TEST_H_CHIP_PLACEMENT_LEGALIZER_DETAILED_FLAG:
        os.environ["PYBUDA_LEGALIZER_DETAILED_DEBUGGING"] = "1"
    if TEST_H_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG:
        if TEST_H_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG_CONSUMER:
            os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"
        if TEST_H_CHIP_PLACEMENT_SELF_CUT_TYPE_FLAG_PRODUCER:
            os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ProducerUserDataEdgesFirst"

    if TEST_H_CHIP_PLACEMENT_LEGALIZER_NODE_NAME:
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
    (1, 6, 192, 192), 
    (1, 1, 256, 256), 
    (8, 192, 192),
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

if TEST_H_SHAPE_SIZE_1_FLAG:
    original_shape += shapes_size_1
if TEST_H_SHAPE_SIZE_2_FLAG:
    original_shape += shapes_size_2
if TEST_H_SHAPE_SIZE_3_FLAG:
    original_shape += shapes_size_3
if TEST_H_SHAPE_SIZE_4_FLAG:
    original_shape += shapes_size_4

original_shape_ids = [
    f"shape={'x'.join([str(jtem) for jtem in item])}" for item in original_shape
]

test_model = []
if TEST_H_MODEL_H_FLAG:
    test_model.append(("TestPaddingPassH", TestPaddingPassH))
if TEST_H_MODEL_H_1_FLAG:
    test_model.append(("TestPaddingPassH_1", TestPaddingPassH_1))
if TEST_H_MODEL_H_2_FLAG:
    test_model.append(("TestPaddingPassH_2", TestPaddingPassH_2))
if TEST_H_MODEL_H_3_FLAG:
    test_model.append(("TestPaddingPassH_3", TestPaddingPassH_3))
    

@pytest.mark.xfail(reason="tenstorrent/pybuda#1004#note_216172")
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
@pytest.mark.parametrize("test_model", test_model, ids=[item[0] for item in test_model])
def test_padding_pass_h(
    test_kind,
    test_device,
    original_shape,
    test_model
):

    if test_kind.is_training():
        pytest.skip()

    if test_kind.is_training():
        pass_name = "training"
    else:
        pass_name = "inference"

    set_environment()

    verify_cfg = VerifyConfig(
        # pcc=1.0,
        run_golden=False,
        verify_all=True
    )

    test_name = f"{test_model[0]}_{'shape=' + 'x'.join([str(item) for item in original_shape])}_{pass_name}"
    model = test_model[1](name=test_name, shape=original_shape)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *model.inputs, 
        compiler_cfg=CompilerConfig(
            enable_training=test_kind.is_training(),
        ), 
        verify_cfg=verify_cfg
    )

@pytest.mark.xfail(reason="tenstorrent/pybuda#1004#note_216172")
@pytest.mark.parametrize("original_shape", original_shape, ids=original_shape_ids)
def test_padding_pass_h_argument(
    test_kind,
    test_device,
    original_shape
):
    if test_device.arch == pybuda.BackendDevice.Grayskull:
        pytest.skip("Wait until #1004 is resolved")

    if test_kind.is_training():
        pass_name = "training"
    else:
        pass_name = "inference"

    set_environment()

    paddings = {
        'mm3': True,
        'mm5': True,
        'mul1': True
    }

    verify_cfg = VerifyConfig(
        # pcc=1.0,
        run_golden=False,
        verify_all=True
    )

    compiler_cfg = CompilerConfig(
        enable_training=test_kind.is_training(),
        paddings=paddings
    )

    # BudaDramQueue in combination with t-streaming raises an error on budabackend 
    if TEST_H_PADDING_PASS_BUFFER_QUEUE_FLAG:
        override = OpOverride()
        override.t_stream_dir = "n"
        compiler_cfg.balancer_op_overrides["mm3.unpadding.select_2"] = override

    test_name = f"TestPaddingPassH_3_{'shape=' + 'x'.join([str(item) for item in original_shape])}_{pass_name}"
    model = TestPaddingPassH_3(name=test_name, shape=original_shape)
    tt0 = TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch)
    tt0.place_module(model)
    pybuda_compile(
        tt0, 
        model.name, 
        *model.inputs,
        compiler_cfg=compiler_cfg, 
        verify_cfg=verify_cfg
    )