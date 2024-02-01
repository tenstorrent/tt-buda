# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest

import torch
import onnx
from loguru import logger


from pybuda import (
    OnnxModule,
    VerifyConfig,
)
from pybuda.config import _get_global_compiler_config
from pybuda.verify import verify_module
from transformers import OPTModel, OPTConfig

def test_tvm_opt_fallback(test_kind, test_device):
    # Training fails due to unsupported op
    if test_kind.is_training():
        pytest.skip()
    

    config = OPTConfig()
    compiler_cfg = _get_global_compiler_config()

    input_shape = (1, 32,)
    config.num_hidden_layers = 1
    model = OPTModel(config)

    save_path = os.path.dirname(os.path.realpath(__file__)) + "/opt_onnx.onnx"

    torch.onnx.export(model,               # model being run
                        torch.randint(2000, input_shape), # model input (or a tuple for multiple inputs),
                        save_path,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=12,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        )

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    mod = OnnxModule(
        "opt_onnx",
        onnx_model,
        save_path,
    )
    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.retain_tvm_python_files = True

    verify_module(
        mod,
        (input_shape,),
        input_params=[{"data_format" :  torch.int64},],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )
    os.remove(save_path)
