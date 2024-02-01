# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
""" 
Test large matmuls needed in large language models on WH
# TODO: test large attention matmuls
"""
import pytest

import pybuda
from pybuda.verify import verify_module, VerifyConfig, TestKind
import torch


class MatmulTest(pybuda.PyBudaModule):
    """
    Simple matmul test
    """

    def __init__(self, name, shape):
        super().__init__(name)
        self.weight = pybuda.Parameter(*shape, requires_grad=True)

    def forward(self, act):
        out = pybuda.op.Matmul("matmul", act, self.weight)
        return out


matmul_sizes = {
            'bert_tiny_linear': (128, 128, 128),
            'bert_linear': (512, 768, 768),
            'bert_ff0': (512, 768, 3072),
            'bert_ff1': (512, 3072, 768),
            'bert_large_linear': (512, 1024, 1024),
            'bert_large_ff0': (512, 1024, 4096),
            'bert_large_ff1': (512, 4096, 1024),
            'gpt2_linear': (1024, 768, 768),
            'gpt2_ff0': (1024, 768, 3072),
            'gpt2_ff1': (1024, 3072, 768),
            #'resnet50_224x224': (50176, 32, 64 * 3 * 3),
            # 'gpt3_linear': (2048, 12288, 12288), # TODO: Temp skip due the CI Pipeline issues. 
            #'gpt3_ff0': (2048, 12288, 49152),  # currently don't need to fit on one chip
            #'gpt3_ff1': (2048, 49152, 12288),  # currently don't need to fit on one chip
            # 'mtnlg_linear': (2048, 20480, 20480), # TODO: Temp skip due the CI Pipeline issues. 
            #'mtnlg_ff0': (2048, 20480, 81920), # currently don't need to fit on one chip
            #'mtnlg_ff1': (2048, 81920, 20480)  # currently don't need to fit on one chip
}

@pytest.mark.parametrize("shape_id",  matmul_sizes.keys(), ids=matmul_sizes.keys())
def test_single_op(test_device, shape_id):
    shape = matmul_sizes[shape_id]
    in_shape = shape[:2]
    weight_shape = shape[1:]

    if in_shape[1] > 4096:
        #pytest.skip() # needs work
        pass

    scale_params = (3*in_shape[1])**.5 # how mt-nlg does weight init

    verify_module(MatmulTest(f"matmul_{shape_id}", weight_shape), [(1, *in_shape),],
            VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch),
            scale_params=scale_params)

