# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import os

import pytest
import onnx
import torch

from transformers import DistilBertModel

from pybuda import OnnxModule, VerifyConfig
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import CompileDepth, _get_global_compiler_config


def test_distilbert_onnx(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()
        
    # Data mismatch while comparing to backend golden (first iteration)
    compile_cfg = _get_global_compiler_config()
    compile_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    class Transformer(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.attn_mask = torch.ones((1, 128))
            self.module = module

        def forward(self, input_act):
            return self.module(input_act, self.attn_mask)

    # Load framework model
    framework_module = DistilBertModel.from_pretrained(
        "distilbert-base-cased-distilled-squad"
    )
    framework_module = Transformer(framework_module)

    # Generate example output
    input_act_shape = (1, 128)
    act_input = torch.randint(0, 25000, input_act_shape)
    # out_example = framework_module(act_input)

    # Export to ONNX
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/distilbert.onnx"
    torch.onnx.export(
        framework_module,
        (act_input,),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        # example_outputs=(out_example,),
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "distilbert_onnx",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (input_act_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={"attention.k_lin.bias"},
        ),
        input_params=[{"data_format": torch.int64}],
    )

    # Cleanup
    os.remove(save_path)
