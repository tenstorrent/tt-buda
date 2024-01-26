# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Test data format control
"""
import pytest
import os
import torch

import pybuda
import pybuda.op
from pybuda import (
    PyBudaModule,
    TTDevice,
    BackendType,
    Tensor,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
    DataFormat,
)
from pybuda._C.backend_api import BackendDevice
from pybuda._C import NodeEpochType
from pybuda.verify import verify_module, TestKind, config
from test.common import ModuleBuilder, run, run_torch

verify_cfg = VerifyConfig(run_golden=True) # Run backend golden check on all tests in here

class BudaTest(PyBudaModule):
    """
    Simple buda module for basic testing
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name):
        super().__init__(name)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, act2)
        return m1

@pytest.mark.parametrize("pt_format", (torch.float32, torch.bfloat16, torch.float16), ids=["float32", "bfloat16", "float16"])
def test_input(pt_format):
    """
    Test basic pytorch types, with no explicit request for conversion
    """
    mod = BudaTest("test_module")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)

    act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, dtype=pt_format))
    act2 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, requires_grad=True, dtype=pt_format))

    pybuda_compile(tt0, "sanity", act1, act2, compiler_cfg=CompilerConfig(enable_training=False), verify_cfg=verify_cfg)


@pytest.mark.parametrize("pt_format", (torch.float32, torch.bfloat16, torch.float16, torch.int8), ids=["float32", "bfloat16", "float16", "int8"])
@pytest.mark.parametrize("target_format", (DataFormat.Float16, DataFormat.Float16_b, DataFormat.Bfp8, DataFormat.Bfp8_b), ids=["Float16", "Float16_b", "Bfp8", "Bfp8_b"])
def test_input_with_conversion(pt_format, target_format):
    """
    Test basic pytorch types, with explicit request for conversion to tt dataformat
    """
    if target_format == DataFormat.Bfp8 or target_format == DataFormat.Bfp8_b:
        # tenstorrent/budabackend#263
        pytest.skip()

    if pt_format == torch.int8:
        # Not supported through non-backend path
        pytest.skip()

    mod = BudaTest("test_module")
    tt0 = TTDevice("tt0", devtype=BackendType.Golden)
    tt0.place_module(mod)

    if pt_format == torch.int8:
        act1 = Tensor.create_from_torch(torch.randint(size=BudaTest.shape, high=100, dtype=pt_format), dev_data_format=target_format)
        act2 = Tensor.create_from_torch(torch.randint(size=BudaTest.shape, high=100, dtype=pt_format), dev_data_format=target_format)
    else:
        act1 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, dtype=pt_format), dev_data_format=target_format)
        act2 = Tensor.create_from_torch(torch.rand(*BudaTest.shape, requires_grad=True, dtype=pt_format), dev_data_format=target_format)

    pybuda_compile(tt0, "sanity", act1, act2, compiler_cfg=CompilerConfig(enable_training=False), verify_cfg=verify_cfg)

@pytest.mark.skip(reason="Still working on this")
@pytest.mark.parametrize("pt_format", (torch.float32, torch.bfloat16, torch.float16, torch.int8), ids=["float32", "bfloat16", "float16", "int8"])
@pytest.mark.parametrize("target_format", (DataFormat.Float16, DataFormat.Float16_b, DataFormat.Bfp8, DataFormat.Bfp8_b), ids=["Float16", "Float16_b", "Bfp8", "Bfp8_b"])
def test_input_with_conversion_backend(pt_format, target_format):
    """
    Test basic pytorch types, with explicit request for conversion to tt dataformat
    """

    verify_module(BudaTest("input_conversion"), [BudaTest.shape, BudaTest.shape],
            input_params = [
                {"data_format": pt_format,
                 "dev_data_format": target_format },
                {"data_format": pt_format,
                 "dev_data_format": target_format,
                 "requires_grad": pt_format != torch.int8} ],
            verify_cfg=VerifyConfig(test_kind=TestKind.INFERENCE))

class BudaDFTest(pybuda.PyBudaModule):
    """
    Simple buda module for basic testing
    """

    shape = (64, 64)

    def __init__(self, name: str):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(*self.shape, requires_grad=True)
        self.weights2 = pybuda.Parameter(*self.shape, requires_grad=True)

    def forward(self, act1, act2):
        m1 = pybuda.op.Matmul("matmul1", act1, self.weights1)
        m2 = pybuda.op.Matmul("matmul2", act2, self.weights2)
        m1e = pybuda.op.Sqrt("sqrt", m1)
        add = pybuda.op.Add("add", m1e, m2)
        return add

# Too many to run, ends up failing in pytest due to "too many open files"
#formats = [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Bfp8_b, DataFormat.Bfp8, DataFormat.Bfp4]
#format_ids = ["bfloat", "fp16", "bfp8b", "bfp8", "bfp4"]
formats = [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Bfp4_b]
format_ids = ["bfloat", "bfp8b", "bfp4b"]

@pytest.mark.parametrize("param1_df", formats, ids=format_ids)
@pytest.mark.parametrize("param2_df", formats, ids=format_ids)
@pytest.mark.parametrize("input1_df", formats, ids=format_ids)
@pytest.mark.parametrize("input2_df", formats, ids=format_ids)
def test_data_formats(test_kind, test_device, param1_df, param2_df, input1_df, input2_df):
    mod = BudaDFTest("data_formats")
    mod.weights1.set_data_format(param1_df)
    mod.weights2.set_data_format(param2_df)
    verify_module(mod, [(1, *BudaDFTest.shape), (1, *BudaDFTest.shape)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
            input_params=[{"dtype": input1_df}, {"dtype": input2_df}])


def test_bwd_op_format_promotion():
    def bwd_op_format_promotio(act, *, ff1_weights):
        return pybuda.op.Matmul(f"ff1", act, ff1_weights)

    pybuda.config.configure_mixed_precision(
        epoch_type=NodeEpochType.Backward, output_df=DataFormat.Float32, accumulate_df=DataFormat.Float16_b
    )

    module = ModuleBuilder(bwd_op_format_promotio, ff1_weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)], VerifyConfig(test_kind=config.TestKind.TRAINING))


def test_gradient_op_format_promotion():
    def gradient_op_format_promotion(act, *, ff1_weights):
        return pybuda.op.Matmul(f"ff1", act, ff1_weights)

    pybuda.config.configure_mixed_precision(
        is_gradient_op=True, output_df=DataFormat.Float32, accumulate_df=DataFormat.Float16_b
    )

    module = ModuleBuilder(gradient_op_format_promotion, ff1_weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)], VerifyConfig(test_kind=config.TestKind.TRAINING))

def test_bwd_fail():
    def bwd_op_format_promotio(act, *, ff1_weights):
        return pybuda.op.Matmul(f"ff1", act, ff1_weights)
    
    df = DataFormat.Float16

    pybuda.config.set_configuration_options(
        default_df_override=df,
        accumulate_df=df,
        enable_auto_transposing_placement=True,
        backend_opt_level=3
    )

    pybuda.config.configure_mixed_precision(
        op_type="matmul",
        accumulate_df=pybuda._C.DataFormat.Float32,
        intermediate_df=pybuda._C.DataFormat.Float32,
        is_gradient_op=False
    )

    module = ModuleBuilder(bwd_op_format_promotio, ff1_weights=pybuda.Parameter(1,1,64,64))
    verify_module(module, [(1, 1, 64, 64)], VerifyConfig(test_kind=config.TestKind.TRAINING))

def test_eltwise_binary_mixed_ab_inputs(test_device):
    shape = (1, 1, 32, 32)

    @run(
        VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch),
    )
    def mixed_ab_inputs(x, y):
        return pybuda.op.Add("add", x, y)

    x = Tensor.create_from_torch(torch.randn(shape, dtype=torch.bfloat16))
    y = Tensor.create_from_torch(torch.randn(shape, dtype=torch.float16))
    mixed_ab_inputs(x, y)


def test_matmul_large_mk_decoupled_acc_intermediate_df(test_device):
    if test_device.arch != BackendDevice.Wormhole_B0:
        pytest.skip("Only relevant for Wormhole_B0 with FP32 accumulation")
    shape = (1, 1, 768, 768)
    test_kind = TestKind.INFERENCE

    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
    )
    def simple_matmul_gradient_t(x, weight=None):
        return pybuda.op.Matmul("mm0", x, weight)

    x = Tensor.create_from_torch(torch.randn(shape, requires_grad=test_kind.is_training()))
    w = pybuda.Parameter(*shape, requires_grad=test_kind.is_training())

    pybuda.config.configure_mixed_precision(
        op_type="matmul",
        intermediate_df=pybuda._C.DataFormat.Float16_b,
        accumulate_df=pybuda._C.DataFormat.Float32,
    )
    simple_matmul_gradient_t(x, weight=w)


def test_gradient_matmul_decoupled_acc_intermediate_df(test_device):
    if test_device.arch != BackendDevice.Wormhole_B0:
        pytest.skip("Only relevant for Wormhole_B0 with FP32 accumulation")
    shape = (1, 1, 768, 768)
    test_kind = TestKind.TRAINING

    @run(
        VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch),
    )
    def simple_matmul_gradient_t(x, weight=None):
        return pybuda.op.Matmul("mm0", x, weight)

    x = Tensor.create_from_torch(torch.randn(shape, requires_grad=test_kind.is_training(), dtype=torch.bfloat16))
    w = pybuda.Parameter(torch.randn(shape, requires_grad=test_kind.is_training(), dtype=torch.bfloat16))

    pybuda.config.configure_mixed_precision(
        op_type="matmul",
        intermediate_df=pybuda._C.DataFormat.Float16_b,
        accumulate_df=pybuda._C.DataFormat.Float32,
        output_df=pybuda._C.DataFormat.Float16_b,
        is_gradient_op=True
    )
    simple_matmul_gradient_t(x, weight=w)


def test_stochastic_rounding(test_device):
    os.environ["PYBUDA_ENABLE_STOCHASTIC_ROUNDING"] = "1"
    if test_device.devtype == BackendType.Golden:
        os.environ["GOLDEN_WORMHOLE_B0"] = "1"
    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch),
    )
    def operation(x, y, z):
        matmul = pybuda.op.Matmul("matmul", x, y)
        gelu = pybuda.op.Gelu("gelu", matmul)
        add = pybuda.op.Add("add", gelu, z)
        return add

    dims = (1, 1, 128, 128)
    X = Tensor.create_from_torch(torch.randn(dims, dtype=torch.float16))
    Y = Tensor.create_from_torch(torch.randn(dims, dtype=torch.float16))
    Z = Tensor.create_from_torch(torch.randn(dims, dtype=torch.float16))

    operation(X,Y,Z)


def test_splice(test_device):
    pybuda.config.configure_mixed_precision(
        op_type="splice",
        input_df={
            0: [pybuda.DataFormat.Float32, True],
            1: [pybuda.DataFormat.Bfp8, True],
        }
    )

    @run_torch(
        VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch),
    )
    def splice(x, y):
        z = x[...,1:]
        return torch.cat((x, y), dim=-1)

    x = Tensor.create_from_torch(torch.randn(1, 1, 32, 32))
    y = Tensor.create_from_torch(torch.randn(1, 1, 32, 1))
    splice(x, y)


def test_format_conversion(test_device):
    """
    Grayskull, matmul, a-formats on inputs, b-format on output w/ bfp8_b
    """

    pybuda.config.configure_mixed_precision(
        op_type="matmul",
        input_df={
            0: [pybuda.DataFormat.Float16_b, True],
            1: [pybuda.DataFormat.Float16_b, True],
        },
        output_df=pybuda.DataFormat.Bfp8
    )

    @run_torch(
        VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch),
    )
    def matmul_exponent_conversion(x, y):
        return torch.matmul(x, y)

    x = Tensor.create_from_torch(torch.randn(1, 1, 32, 32))
    y = Tensor.create_from_torch(torch.randn(1, 1, 32, 32))
    matmul_exponent_conversion(x, y)
