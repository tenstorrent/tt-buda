# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import pybuda

import onnx
import torch
import numpy as np

from tvm.relay.op.contrib import match_einsum_pattern

from pybuda import (
    PyTorchModule,
    VerifyConfig,
    OnnxModule,
)
from pybuda.verify.config import TestKind
from pybuda.verify.backend import verify_module
from pybuda.config import CompileDepth, _get_global_compiler_config
import pybuda


tensor_dims = [2, 3, 4, 5]
tries = 100
alphabet = "abcdefghijklmnopqrstuvwxyz"


@pytest.mark.parametrize(
    "tensor_dim",
    tensor_dims,
    ids=[f"dim{str(s)}" for s in tensor_dims],
)
def test_einsum_pattern_match(tensor_dim):
    for t in range(tries):

        # Generate random pattern
        tensor_a = ""
        tensor_b = ""
        tensor_c = ""

        letters_left = len(alphabet)
        useable_letters = alphabet

        for i in range(tensor_dim):
            rand_a = useable_letters[np.random.randint(0, letters_left)]
            rand_b = useable_letters[np.random.randint(0, letters_left)]
            useable_letters = useable_letters.replace(rand_a, "").replace(rand_b, "")

            letters_left -= 1
            if rand_a != rand_b:
                letters_left -= 1

            tensor_a = f"{tensor_a}{rand_a}"
            tensor_b = f"{tensor_b}{rand_b}"

        # set of letters describing equation, use this to make output tensor
        useable_letters = "".join(set(tensor_a + tensor_b))
        letters_left = len(useable_letters)

        for i in range(tensor_dim):
            rand_c = useable_letters[np.random.randint(0, letters_left)]
            useable_letters = useable_letters.replace(rand_c, "")
            letters_left -= 1

            tensor_c = f"{tensor_c}{rand_c}"

        pattern = f"{tensor_a}, {tensor_b} -> {tensor_c}"

        # Generate equivalent pattern
        char_map = {}
        pattern_chars = set(tensor_a + tensor_b)

        useable_letters = "".join(set(alphabet) - pattern_chars)
        letters_left = len(useable_letters)

        assert (
            letters_left >= 13
        ), "Writing a test that handles the case where the query pattern and pattern definition contain some of the same characters is cumbersome. Please use half the alphabet or less in the pattern definition."

        for char in pattern_chars:
            new_char = useable_letters[np.random.randint(0, letters_left)]
            useable_letters = useable_letters.replace(new_char, "")
            letters_left -= 1

            char_map[char] = new_char

        for old_char, new_char in char_map.items():
            tensor_a = tensor_a.replace(old_char, new_char)
            tensor_b = tensor_b.replace(old_char, new_char)
            tensor_c = tensor_c.replace(old_char, new_char)

        query = f"{tensor_a}, {tensor_b} -> {tensor_c}"

        assert match_einsum_pattern(
            pattern, query
        ), f"The conclusion should be that '{query}' matches the einsum pattern '{pattern}'."

        # Try a failing case

        # Find a character that is in both tensor_a or tensor_b and tensor_c
        replaced_char = ""
        in_tensor_a = False
        for char_a in tensor_a:
            if in_tensor_a:
                break
            for char_c in tensor_c:
                if char_a == char_c:
                    replaced_char = char_c
                    in_tensor_a = True
                    break

        in_tensor_b = False
        if not in_tensor_a:
            for char_b in tensor_b:
                if in_tensor_b:
                    break
                for char_c in tensor_c:
                    if char_b == char_c:
                        replaced_char = char_c
                        in_tensor_b = True
                        break

        useable_letters = list(set(alphabet) - set(replaced_char))
        if in_tensor_a:
            tensor_a = tensor_a.replace(
                replaced_char,
                useable_letters[np.random.randint(0, len(useable_letters))],
            )
        else:
            tensor_b = tensor_b.replace(
                replaced_char,
                useable_letters[np.random.randint(0, len(useable_letters))],
            )

        # A single character in either tensor_a or tensor_b will be off in this query, so the patterns should not match
        query = f"{tensor_a}, {tensor_b} -> {tensor_c}"
        assert not match_einsum_pattern(
            pattern, query
        ), f"The conclusion should be that '{query}' DOES NOT match the einsum pattern '{pattern}'."


def test_tvm_all_and_cast_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(5, 3)
            self.l2 = torch.nn.Linear(3, 5)

        def forward(self, x):
            x = torch.all(x.detach(), 1, False)
            x = x.type(torch.float)
            x = x.unsqueeze(0)
            x = self.l1(x)
            x = self.l2(x)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("all")
    compiler_cfg.cpu_fallback_ops.add("cast")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_all_and_cast_fallback", framework_module)

    input_shape = (1, 3, 5)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_broadcast_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bcast_shape = (1, 3, 5)
            self.l1 = torch.nn.Linear(5, 3)
            self.l2 = torch.nn.Linear(3, 5)

        def forward(self, x):
            x = torch.broadcast_to(x, self.bcast_shape)
            x = self.l1(x)
            x = self.l2(x)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("broadcast_to")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_broadcast_fallback", framework_module)

    input_shape = (1, 1, 5)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_reshape_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.new_shape = (1, 1, 2, 9)

        def forward(self, x):
            x = torch.reshape(x, self.new_shape)
            x = torch.add(x, x)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("reshape")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_reshape_fallback", framework_module)

    input_shape = (1, 1, 3, 6)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_cumsum_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.cumsum(x, 2)
            x = torch.add(x, x)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("cumsum")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_cumsum_fallback", framework_module)

    input_shape = (1, 1, 3, 6)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_log_softmax_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.log_softmax(x, 2)
            x = torch.add(x, x)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("nn.log_softmax")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_log_softmax_fallback", framework_module)

    input_shape = (1, 1, 3, 6)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_softmax_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.softmax(x, 2)
            x = torch.add(x, x)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("nn.softmax")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_softmax_fallback", framework_module)

    input_shape = (1, 1, 3, 6)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_transpose_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x.squeeze(0)
            x = torch.add(x, x)
            x = torch.transpose(x, 1, 0)
            x = torch.softmax(x, 1)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("transpose")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_transpose_fallback", framework_module)

    input_shape = (1, 3, 6)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_scatter_add_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()
        
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.src = torch.ones((1, 2, 5))
            self.index = torch.tensor([[[0, 1, 2, 0, 0]]])
            self.l1 = torch.nn.Linear(5, 3)
            self.l2 = torch.nn.Linear(3, 5)
            # self.softmax = torch.nn.LogSoftmax(dim=1)

        def forward(self, x):
            x = torch.scatter_add(x, 2, self.index, self.src)
            x = self.l1(x)
            x = self.l2(x)

            # if torch.equal(x, x):
            #     x = self.softmax(x)

            return x

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_scatter_add_fallback", framework_module)

    input_shape = (1, 3, 5)

    # Sanity run
    act = torch.zeros(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_scatter_add_fallback_inplace(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.src = torch.ones((1, 2, 5))
            self.index = torch.tensor([[[0, 1, 2, 0, 0]]])
            self.l1 = torch.nn.Linear(5, 3)
            self.l2 = torch.nn.Linear(3, 5)

        def forward(self, x):
            x = x.scatter_add(2, self.index, self.src)
            x = self.l1(x)
            x = self.l2(x)

            return x

    compiler_cfg = _get_global_compiler_config() 

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_scatter_add_fallback_inplace", framework_module)

    input_shape = (1, 3, 5)

    # Sanity run
    act = torch.zeros(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_max_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(1, 3)
            self.l2 = torch.nn.Linear(3, 5)

        def forward(self, x):
            x = torch.max(x)
            x = x.unsqueeze(0)
            x = self.l1(x)
            x = self.l2(x)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("max")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_max_fallback", framework_module)

    input_shape = (1, 3, 5)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_cat_fallback(test_kind, test_device):
    pytest.skip()
    #TODO: Fix tvm.14 regressions: tenstorrent/pybuda#2099
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.add(x, x)
            x = torch.cat((x, x), 1)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("concatenate")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_cat_fallback", framework_module)

    input_shape = (1, 3, 6)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_argmax_fallback(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.argmax(x, 1, False)
            x = torch.add(x, x)

            return x

    compiler_cfg = _get_global_compiler_config() 
    compiler_cfg.cpu_fallback_ops.add("argmax")

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_argmax_fallback", framework_module)

    input_shape = (1, 3, 6)

    # Sanity run
    # act = torch.rand(input_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_reshape_transpose_into_hslice(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.l1 = torch.nn.Linear(512, 512)
            self.l2 = torch.nn.Linear(64, 64)

        def forward(self, x):
            x = self.l1(x)
            
            reshape0 = torch.reshape(x, (1, 2048, 8, 64))
            x = torch.mul(reshape0, reshape0)
            x = torch.sum(x, -1)
            x = torch.reshape(x, (1, 2048, 8, 1))
            sqrt0 = torch.sqrt(x)

            const0 = torch.ones((1, 1))
            x = torch.max(const0, sqrt0)
            
            const1 = torch.ones((1, 1))
            x = torch.min(const1, x)
            
            x = torch.subtract(x, sqrt0)
            x = torch.add(sqrt0, x)

            x = torch.reciprocal(x)
            x = torch.mul(reshape0, x)

            x = torch.transpose(x, -3, -2)
            x = torch.add(x, x)
            
            x = self.l2(x)

            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_reshape_transpose_into_hslice", framework_module)

    input_shape = (1, 2048, 512)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_transpose_reshape_into_hstack(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.l2 = torch.nn.Linear(256, 256)

        def forward(self, x):
            x = torch.add(x, x)
            
            transpose0 = torch.transpose(x, -3, -2)
            x = torch.mul(transpose0, transpose0)
            sqrt0 = torch.sqrt(x)

            const0 = torch.ones((1, 1))
            x = torch.max(const0, sqrt0)
            
            const1 = torch.ones((1, 1))
            x = torch.min(const1, x)
            
            x = torch.subtract(x, sqrt0)
            x = torch.add(sqrt0, x)

            x = torch.reciprocal(x)
            x = torch.mul(transpose0, x)

            x = torch.reshape(x, (1, 1, 2048, 256))
            x = torch.softmax(x, -1)
            x = torch.add(x, x)
            
            x = self.l2(x)

            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_transpose_reshape_into_hstack", framework_module)

    input_shape = (1, 4, 2048, 64)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_reshape_into_vslice(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    #Fusing disabled due to tenstorrent/pybuda#784
    pybuda.set_configuration_options(enable_auto_fusing=False)

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.l1 = torch.nn.Linear(256, 256)

        def forward(self, x):
            x = torch.add(x, x)
            
            transpose0 = torch.transpose(x, -3, -2)
            x = torch.mul(transpose0, transpose0)
            sqrt0 = torch.sqrt(x)

            const0 = torch.ones((1, 1))
            x = torch.max(const0, sqrt0)
            
            const1 = torch.ones((1, 1))
            x = torch.min(const1, x)
            
            x = torch.subtract(x, sqrt0)
            x = torch.add(sqrt0, x)

            x = torch.reciprocal(x)
            x = torch.mul(transpose0, x)

            x = torch.reshape(x, (1, 1, 1024, 256))
            x = torch.softmax(x, -1)
            x = torch.reshape(x, (1, 2, 512, 256))
            x = x*2
            
            x = self.l1(x)

            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_reshape_into_vslice", framework_module)

    input_shape = (1, 4, 1024, 64)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_reshape_into_vstack(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.l1 = torch.nn.Linear(64, 64)

        def forward(self, x):
            x = torch.add(x, x)
            
            transpose0 = torch.transpose(x, -3, -2)
            x = torch.mul(transpose0, transpose0)
            sqrt0 = torch.sqrt(x)

            const0 = torch.ones((1, 1))
            x = torch.max(const0, sqrt0)
            
            const1 = torch.ones((1, 1))
            x = torch.min(const1, x)
            
            x = torch.subtract(x, sqrt0)
            x = torch.add(sqrt0, x)

            x = torch.reciprocal(x)
            x = torch.mul(transpose0, x)

            x = x.reshape(1, 1, 4096, 64)
            x = torch.add(x, x)

            x = self.l1(x)

            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_reshape_into_vslice", framework_module)

    input_shape = (1, 1024, 4, 64)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_cpu_fallback_when_more_performant(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.conv1 = torch.nn.Conv2d(3, 16, 3)

        def forward(self, x):
            x = self.conv1(x)
            
            tensors_along_t_dim = []
            b, t, r, c = x.shape
            for t_id in range(t):
                tensors_along_t_dim.append(
                    x[:, t_id].reshape((1, 1, 49))
                )

            x = torch.stack(tensors_along_t_dim, axis=1)
            x = x.reshape(b, t, r, c)
            
            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tm_cpu_fallback = True

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_cpu_fallback_when_more_performant", framework_module)

    input_shape = (1, 3, 9, 9)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_extended_tm_cpu_fallback_concat_variation(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.conv1 = torch.nn.Conv2d(3, 16, 3)

        def forward(self, x):
            x = self.conv1(x)
            
            tensors_along_t_dim = []
            b, t, r, c = x.shape
            for t_id in range(t):
                tensors_along_t_dim.append(
                    x[:, t_id].reshape((1, 1, 1, 49))
                )

            x = torch.cat(tensors_along_t_dim, dim=1)
            x = x.reshape(b, t, r, c)
            
            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.enable_tm_cpu_fallback = True

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_test_extended_tm_cpu_fallback_concat_variation", framework_module)

    input_shape = (1, 3, 9, 9)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )
    
    
def test_extended_tm_cpu_fallback_hslice_variation(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.conv1 = torch.nn.Conv2d(3, 16, 3)

        def forward(self, x):
            x = self.conv1(x)
            
            tensors_along_t_dim = []
            b, t, r, c = x.shape
            for t_id in range(t):
                inter_res = x[:, t_id]
                inter_res = inter_res.reshape((1, 12, 3, 4))
                inter_res = torch.transpose(inter_res, 1, 2)
                tensors_along_t_dim.append(
                    inter_res
                )

            x = torch.cat(tensors_along_t_dim, dim=1)
            x = x.reshape(b, t, r, c)
            
            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True
    compiler_cfg.enable_tm_cpu_fallback = True

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_test_extended_tm_cpu_fallback_hslice_variation", framework_module)

    input_shape = (1, 3, 14, 14)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )
    
def test_extended_tm_cpu_fallback_matmul_variation(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()            

        def forward(self, x):
            x = torch.matmul(x, torch.transpose(x, 2, 3))
            
            tensors_along_t_dim = []
            b, t, r, c = x.shape
            for t_id in range(t):
                tensors_along_t_dim.append(
                    x[:, t_id].reshape((1, 1, 1, 196))
                )

            x = torch.cat(tensors_along_t_dim, dim=1)
            x = x.reshape(b, t, r, c)
            
            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tm_cpu_fallback = True

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_test_extended_tm_cpu_fallback_matmul_variation", framework_module)

    input_shape = (1, 3, 14, 14)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )


def test_tvm_batch_matmul_with_1d_op1(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.const = torch.randn((1024,))

        def forward(self, x):
            x = torch.matmul(x, self.const)
            return x

    compiler_cfg = _get_global_compiler_config() 

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_batch_matmul_with_1d_op1", framework_module)

    input_shape = (1, 64, 32, 1024)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_batch_matmul_with_1d_op0(test_kind, test_device):
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.const = torch.randn((1024,))

        def forward(self, x):
            x = torch.matmul(self.const, x)
            return x

    compiler_cfg = _get_global_compiler_config() 

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_batch_matmul_with_1d_op0", framework_module)

    input_shape = (1, 32, 1024, 64)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

# Replicating following issue
# tenstorrent/pybuda#365
def test_group_conv2d(test_kind, test_device):
    pytest.skip() # Skipping for now until fix is in place
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    if not test_kind.is_training():
        pytest.skip()
        
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(
                768,
                768,
                (7, 7),
                stride=1,
                dilation=1,
                groups=768,
                padding=(3, 3),
            )

        def forward(self, x):
            x = self.conv1(x)

            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.retain_tvm_python_files = True

    framework_model = Model()
    module = PyTorchModule("pt_group_conv2d", framework_model)

    input_shape = (1, 768, 2, 2)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )

def test_invalid_reshape_transpose_into_hslice(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.lin1 = torch.nn.Linear(256, 256, bias=False)
            self.lin2 = torch.nn.Linear(64, 64, bias=False)

        def forward(self, x):
            x = self.lin1(x)
            
            x = x.reshape([1, 5, 16, 64])
            x = x.transpose(-3, -2)
            
            x = self.lin2(x)
            
            return x

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_invalid_reshape_transpose_into_hslice", framework_module)

    input_shape = (1, 5, 4, 256)

    # Sanity run
    act = torch.rand(input_shape)
    out = framework_module(act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )

def test_neg_inf_const_pt(test_kind, test_device):   
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Mask filled with zeros and ones
            self.mask = torch.randint(0, 2, (1, 128, 768), dtype=torch.bool)

        def forward(self, act):
            a = act.masked_fill(self.mask, torch.tensor(-float("inf")))
            b = torch.nn.functional.softmax(a, dim=-1)
            return b

    framework_module = Transformer()
    pybuda_module = PyTorchModule("pt_neg_inf_const_onnx", framework_module)

    # Input shapes
    inp_shape = (1, 128, 768)

    # Sanity check
    # act = torch.rand(inp_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (inp_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95,
        ),
    )


def test_neg_inf_const_onnx(test_kind, test_device):   
    # Too small values to compare
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY
    
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Mask filled with zeros and ones
            self.mask = torch.randint(0, 2, (1, 128, 768), dtype=torch.bool)

        def forward(self, act):
            a = act.masked_fill(self.mask, torch.tensor(-float("inf")))
            b = torch.nn.functional.softmax(a, dim=-1)
            return b

    # PyTorch module
    framework_module = Transformer()
    
    # Input shapes
    inp_shape = (1, 128, 768)
    
    # Sanity check
    act = torch.rand(inp_shape)
    out = framework_module(act)

    # Export to ONNX
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/neg_inf_const.onnx"
    torch.onnx.export(
        framework_module,
        (act,),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX module
    onnx_module = onnx.load(save_path)
    onnx.checker.check_model(onnx_module)
    onnx_module = OnnxModule(
        "onnx_neg_inf_const",
        onnx_module,
        save_path,
    )

    verify_module(
        onnx_module,
        (inp_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=0.95,
        ),
    )
    
    # Cleanup
    os.remove(save_path)


def test_maximum_bwd(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, act):
            return torch.max(act, torch.tensor(torch.finfo(act.dtype).min))

    framework_module = Model()
    pybuda_module = PyTorchModule("pt_maximum_bwd", framework_module)

    # Input shapes
    inp_shape = (1, 8, 128, 256)

    # Sanity check
    # act = torch.rand(inp_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (inp_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_minimum_bwd(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, act):
            return torch.min(act, torch.tensor(torch.finfo(act.dtype).max))

    framework_module = Model()
    pybuda_module = PyTorchModule("pt_minimum_bwd", framework_module)

    # Input shapes
    inp_shape = (1, 8, 128, 256)

    # Sanity check
    # act = torch.rand(inp_shape)
    # out = framework_module(act)

    verify_module(
        pybuda_module,
        (inp_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_tvm_scatter_nd(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()
        
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    
    compiler_cfg.enable_tm_cpu_fallback = True
    compiler_cfg.cpu_fallback_ops.add("adv_index")
    # compiler_cfg.enable_tvm_constant_prop = True
    
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()
        
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.l1 = torch.nn.Linear(18, 18)

        def forward(self, input_act, threshold):
            input_act = self.l1(input_act)
            input_act = torch.sigmoid(input_act)

            threshold = threshold.expand(input_act.shape[0], -1)

            input_act_new = torch.zeros(input_act.shape, device=input_act.device) + 0.5

            mask_leq = (input_act < threshold) & ~torch.isnan(threshold)
            mask_gt = ~(input_act < threshold) & ~torch.isnan(threshold)

            input_act_new[mask_leq] = input_act[mask_leq] / (threshold[mask_leq] * 2)
            input_act_new[mask_gt] = 1.0 - ((1.0 - input_act[mask_gt]) / ((1 - threshold[mask_gt]) * 2))
            
            return input_act_new

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_tvm_scatter_nd", framework_module)

    input_shape = (1, 18)

    # Sanity run
    # input_act = torch.rand(input_shape)
    # threshold = torch.rand(input_shape)
    # threshold[0, [1, 3, 7]] = torch.nan
    # out = framework_module(input_act, threshold)

    verify_module(
        pybuda_module,
        (input_shape,input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_tvm_invalid_dtype(test_kind, test_device):
    pytest.skip()
    #TODO: Fix tvm.14 regressions: tenstorrent/pybuda#2099
    if test_kind.is_training():
        pytest.skip()
        
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:
        pytest.skip()
        
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    
    compiler_cfg.enable_tm_cpu_fallback = True
    compiler_cfg.tm_cpu_fallback_max_depth = 5
    compiler_cfg.cpu_fallback_ops.add("adv_index")
    compiler_cfg.cpu_fallback_ops.add("argwhere")
    compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.l1 = torch.nn.Linear(18, 18)

        def forward(self, input_act):
            input_act = self.l1(input_act)

            mask_leq = (input_act < 0.001)
            input_act[mask_leq] = input_act[mask_leq]

            return input_act

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_tvm_invalid_dtype", framework_module)

    input_shape = (1, 18)

    # Sanity run
    input_act = torch.rand(input_shape)
    out = framework_module(input_act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )
    

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.l1 = torch.nn.Linear(18, 18)

        def forward(self, input_act):
            input_act = self.l1(input_act)
            
            mask_leq = (input_act < 0.001)
            input_act[mask_leq] = torch.tensor(1.0)

            return input_act

    framework_module = Module()
    pybuda_module = PyTorchModule("pt_tvm_where", framework_module)

    input_shape = (1, 18)

    # Sanity run
    # input_act = torch.rand(input_shape)
    # out = framework_module(input_act)

    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
            # verify_each_buda_pass=True,
        ),
    )


def test_conv2d_with_merged_bias(test_kind, test_device):
    if test_kind.is_training():
        pytest.skip()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 7)

        def forward(self, x):
            x = self.conv(x)
            x = torch.nn.functional.relu(x)

            return x

    compiler_cfg = _get_global_compiler_config()
    # compiler_cfg.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER

    framework_model = Model()
    module = PyTorchModule("pt_conv_with_merged_bias", framework_model)

    input_shape = (1, 3, 64, 64)
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            verify_all=True,
        ),
    )
