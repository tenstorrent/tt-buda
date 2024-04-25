# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Emulation-related tests for end-to-end emulation
#
from pybuda import pybuda
from pybuda._C.backend_api import BackendType
from pybuda.module import PyTorchModule
from pybuda.verify.backend import verify_module
from pybuda.tensor import Tensor
from test.utils import download_model
from pybuda.verify.config import TestKind, VerifyConfig
import pytest
import torch
from test.common import run
from transformers import BertModel

def test_emulation_simple_matmul(test_device):
    # Run only emulation tests
    if test_device.devtype != BackendType.Emulation:
        pytest.skip()

    @run(
        verify_cfg=VerifyConfig(
            test_kind=TestKind.INFERENCE,
            devtype=test_device.devtype,
            arch=test_device.arch,
            pcc=0.99),
    )
    def simple_matmul(a, b):
        c = pybuda.op.Matmul("matmul0", a, b)
        return c

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.input_queues_on_host = False
    compiler_cfg.output_queues_on_host = False

    shape = (1, 1, 128, 128)
    a = Tensor.create_from_torch(torch.randn(shape))
    b = Tensor.create_from_torch(torch.randn(shape))
    simple_matmul(a, b)

def test_bert_tiny(test_device):
    # Run only emulation tests
    if test_device.devtype != BackendType.Emulation:
        pytest.skip()
        
    input_shape = (1, 128)
    model = download_model(BertModel.from_pretrained, "prajjwal1/bert-tiny", add_pooling_layer=False)

    pt_module = PyTorchModule("bert", model)

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.input_queues_on_host = False
    compiler_cfg.output_queues_on_host = False

    verify_module(
        pt_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=TestKind.INFERENCE,
            pcc=0.9,
        ),
        input_params=[{"requires_grad": False, "data_format": torch.int}],
    )