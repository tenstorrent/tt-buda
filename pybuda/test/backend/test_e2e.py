# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Testing various types of e2e connections (see testplan for details)
"""

from typing import List

import pytest

import pybuda
from pybuda.verify import verify_module, VerifyConfig, TestKind

shape = (256, 256)

input_types = ["input", "unary", "matmul"]
binary_types = ["eltwise", "matmul"]

class SkipConnections(pybuda.PyBudaModule):

    def __init__(self, name, input_type: str, binary_type: str):
        super().__init__(name)
        self.weights_input = pybuda.Parameter(1, *shape, requires_grad=True) if input_type == "matmul" else None
        self._input_type = input_type
        self._binary_type = binary_type

    def forward(self, act1, act2):

        if self._input_type == "unary":
            input1 = pybuda.op.Buffer("input1", act1)
        elif self._input_type == "matmul":
            assert self.weights_input
            input1 = pybuda.op.Matmul("input1", act1, self.weights_input)
        else:
            input1 = act1
        
        input2 = pybuda.op.Buffer("input2", act2)

        stage2 = pybuda.op.Buffer("stage2", input2)
        stage3 = pybuda.op.Buffer("stage3", stage2)

        if self._binary_type == "matmul":
            output = pybuda.op.Matmul("binary", input1, stage3)
        else:
            output = pybuda.op.Add("binary", input1, stage3)

        return output


@pytest.mark.parametrize("input_type", input_types)
@pytest.mark.parametrize("binary_type", binary_types)
def test_skip_connections(test_kind, test_device, input_type, binary_type):
    # TODO: maybe introduce breaks on bwd pass, too?
    verify_module(SkipConnections("skip_con", input_type, binary_type), [(2, *shape), (2, *shape)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch,
                 microbatch_count=10, epoch_breaks=["stage2", "stage3", "binary"]))


class VConnection(pybuda.PyBudaModule):

    def __init__(self, name, depth: int, binary_types: List[str]):
        super().__init__(name)
        self.depth = depth
        self.binary_types = binary_types

    def forward(self, act):

        fwd = [act]
        for i in range(self.depth):
            fwd.append( pybuda.op.Buffer(f"down{i}", fwd[i] ))

        bwd = fwd[-1]
        for i in range(1, self.depth):
            if self.binary_types[i-1] == "matmul":
                bwd = pybuda.op.Matmul(f"up{i-1}", bwd, fwd[-(i+1)])
            else:
                bwd = pybuda.op.Add(f"up{i-1}", bwd, fwd[-(i+1)])

        return bwd

@pytest.mark.parametrize("depth", [2, 3, 5, 10])
@pytest.mark.parametrize("pattern",
        [
            ["matmul"],
            ["eltwise"],
            ["matmul", "eltwise"],
            ["matmul", "matmul", "eltwise"],
        ])
def test_v_connections(test_kind, test_device, depth, pattern):
    # TODO: maybe introduce breaks on bwd pass, too?


    epoch_breaks = [f"down{i}" for i in range(1, depth)] + [f"up{i}" for i in range(0, depth-1)]
    binary_types = [pattern[i%len(pattern)] for i in range(depth)]

    verify_module(VConnection("v_con", depth, binary_types), [(2, *shape)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch,
                 microbatch_count=min(10, 2*depth), epoch_breaks=epoch_breaks),
            inputs_centered_on_zero=True)

class ForkSkipConnections(pybuda.PyBudaModule):

    def __init__(self, name, input_type: str, binary_type1: str, binary_type2):
        super().__init__(name)
        self.weights_input = pybuda.Parameter(1, *shape, requires_grad=True) if input_type == "matmul" else None
        self._input_type = input_type
        self._binary_type1 = binary_type1
        self._binary_type2 = binary_type2

    def forward(self, act):

        if self._input_type == "unary":
            input1 = pybuda.op.Buffer("input1", act)
        elif self._input_type == "matmul":
            assert self.weights_input
            input1 = pybuda.op.Matmul("input1", act, self.weights_input)
        else:
            input1 = act
        
        stage2 = pybuda.op.Buffer("stage2", input1)

        if self._binary_type1 == "matmul":
            stage3 = pybuda.op.Matmul("binary1", input1, stage2)
        else:
            stage3 = pybuda.op.Add("binary1", input1, stage2)

        stage4 = pybuda.op.Buffer("stage4", stage3)

        if self._binary_type2 == "matmul":
            output = pybuda.op.Matmul("binary2", input1, stage4)
        else:
            output = pybuda.op.Add("binary2", input1, stage4)

        return output


@pytest.mark.parametrize("input_type", input_types)
@pytest.mark.parametrize("binary_type1", binary_types)
@pytest.mark.parametrize("binary_type2", binary_types)
def test_fork_skip_connections(test_kind, test_device, input_type, binary_type1, binary_type2):
    # TODO: maybe introduce breaks on bwd pass, too?
    epoch_breaks=["stage2", "binary1", "stage4", "binary2"]
    verify_module(ForkSkipConnections("fork_skip_con", input_type, binary_type1, binary_type2), [(2, *shape)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch,
                 microbatch_count=10, epoch_breaks=epoch_breaks))

class MultiInputRead(pybuda.PyBudaModule):
    def forward(self, act):
        const = pybuda.op.Constant("const", constant=1)
        stage1 = pybuda.op.Add("add1", act, const)
        stage2 = pybuda.op.Add("add2", act, const)
        stage3 = pybuda.op.Add("add3", act, const)

        return stage1, stage2, stage3

def test_multi_read_input():
    # Test input that's read from multiple forward epochs
    epoch_breaks=["add2", "add3"]
    verify_module(MultiInputRead("multi_input_read"), [(2, *shape)],
            VerifyConfig(test_kind=TestKind.INFERENCE, microbatch_count=10, epoch_breaks=epoch_breaks))


# Test many parameters with adam optimizer, so that we get optimizer e2e queues
import torch
class ManyParams(pybuda.PyBudaModule):
    def __init__(self, name):
        super().__init__(name)
        self.size = 10
        self.params = [pybuda.Parameter(torch.normal(mean=0.0, std=0.1, size=(128, 128)), requires_grad=True) for _ in range(self.size)]

    def forward(self, act):
        for i in range(self.size):
            act = pybuda.op.Matmul(f"matmul_{i}", act, self.params[i])
        return act

def test_optimizer_e2e():
    verify_module(ManyParams("many_params"), [(16, 128, 128)],
        VerifyConfig(test_kind=TestKind.TRAINING, 
            optimizer={"type": "adam", "params": {"learning_rate": 5.0}}))
        


