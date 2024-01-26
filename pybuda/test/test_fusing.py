# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import torch

import pytest
import pybuda
from pybuda.verify import verify_module, VerifyConfig, TestKind
from pybuda import BackendDevice

def get_relaxed_atol_pcc(is_training, test_device, size = 1):
    """
    Figure out reasonable pcc/atol for training on silicon
    """
    training_atol = 0.3
    training_pcc = 0.89
    inference_atol = 0.1
    inference_pcc = 0.95
    if size > 1:
        training_atol = 0.5
    relative_atol = training_atol if is_training else inference_atol
    if test_device.is_silicon() and is_training:
        relative_atol *= 2.5
    pcc = training_pcc if is_training else inference_pcc

    return relative_atol, pcc

def check_fused_result(exp_count, *ops):
    # Check that fused counts are as expected
    g = pybuda.pybudaglobal.get_devices()[0]._compile_output.lowered_graph
    fused_ops = g.get_fused_ops()
    count = len(fused_ops)
    assert exp_count == count, f"Fused op count mismatch"

    for i, f in enumerate(fused_ops):
        input_count = f[0]
        assert ops[i]["inputs"] == input_count, f"Input count mismatch on op {i}"

        schedules = f[1]
        assert len(ops[i]["schedules"]) == len(schedules), f"Schedule count mismatch on op {i}"

        for j, s in enumerate(schedules):
            assert ops[i]["schedules"][j] == len(s), f"Op count mismatch on op {i}, schedule {j}"


class FuseEltwise(pybuda.PyBudaModule):
    """
    Simple module with 2 eltwise ops to be fused
    """

    shape = (1, 1, 32, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)

    def forward(self, act1, act2):
        a1 = pybuda.op.Matmul("matmul", act1, self.weights1)

        # Expecting fusing of a2 and a3
        a2 = pybuda.op.Add("add", act2, a1)
        a3 = pybuda.op.Reciprocal("reciprocal", a2)
        return a3


def test_fuse_eltwise(test_device, test_kind):

    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(FuseEltwise("fuse_eltwise"), [FuseEltwise.shape, FuseEltwise.shape], 
            VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                relative_atol=relative_atol,
                pcc=pcc))

    # on schedule, two ops in it
    if test_kind == TestKind.INFERENCE:
        check_fused_result(1, {"inputs": 2, "schedules": [2]})

def test_dont_fuse(test_device):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.dont_fuse("add")
    verify_module(FuseEltwise("dont_fuse_eltwise"), [FuseEltwise.shape, FuseEltwise.shape],
                  VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch))
    check_fused_result(0)

def test_manual_fuse(test_device):
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.manual_fuse(["a.*", "r.*"])
    verify_module(FuseEltwise("manual_fuse_eltwise"), [FuseEltwise.shape, FuseEltwise.shape],
                  VerifyConfig(test_kind=TestKind.INFERENCE, devtype=test_device.devtype, arch=test_device.arch))
    check_fused_result(1, {"inputs": 2, "schedules": [2]})

class FuseForkJoin(pybuda.PyBudaModule):
    """
    Simple module with a fork
    """

    shape = (1, 1, 32, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)

    def forward(self, act1, act2):
        a1 = pybuda.op.Matmul("matmul", act1, self.weights1)

        # Expecting fusing of a2, a3, a4
        a2 = pybuda.op.Add("add", act2, a1)
        a3 = pybuda.op.Reciprocal("reciprocal", a2)
        a4 = pybuda.op.Sqrt("sqrt", a2)
        return pybuda.op.Multiply("mul", a3, a4)

def test_fuse_fork(test_device, test_kind):

    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(FuseForkJoin("fuse_fork"), [FuseForkJoin.shape, FuseForkJoin.shape], 
            VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                relative_atol=relative_atol,
                pcc=pcc))

    if test_kind == TestKind.INFERENCE:
        check_fused_result(1, {"inputs": 2, "schedules": [4]})

class FuseReduce(pybuda.PyBudaModule):
    """
    Simple module with a reduce in the middle, or at the end, of a fused sequence
    """

    shape = (1, 1, 32, 128)

    def __init__(self, name, middle: bool):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.middle = middle

    def forward(self, act1, act2):
        a1 = pybuda.op.Matmul("matmul", act1, self.weights1)

        # Expecting fusing of ops below
        a2 = pybuda.op.Add("add", act2, a1)
        a3 = pybuda.op.ReduceAvg("reduce_avg", a2, dim=-1)
        if (self.middle):
            a3 = pybuda.op.Reciprocal("reciprocal", a3)
        return a3

@pytest.mark.parametrize("middle", (True, False), ids=["middle", "end"])
def test_fuse_reduce(test_device, middle, test_kind):

    os.environ["PYBUDA_FUSE_REDUCE"] = "1"
    os.environ["PYBUDA_LEGACY_UBLOCK_SHAPE"] = "1"
    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseReduce("fuse_reduce_" + ("middle" if middle else "end"), middle), [FuseReduce.shape, FuseReduce.shape], 
                VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))

        if test_kind == TestKind.INFERENCE:
            if middle:
                check_fused_result(1, {"inputs": 3, "schedules": [2, 1]})
            else:
                check_fused_result(1, {"inputs": 3, "schedules": [2]})

    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"]

class FuseReduceRC(pybuda.PyBudaModule):
    """
    Simple module with reduces in both R and C dimensions.
    """

    shape1 = (1, 1, 128, 256)
    shape2 = (1, 1, 1, 256)

    def __init__(self, name):
        super().__init__(name)
        self.weights = pybuda.Parameter(self.shape1[-1], self.shape1[-1], requires_grad=True)

    def forward(self, act1, act2):
        a1 = pybuda.op.Matmul("matmul", act1, self.weights)

        # Expecting ops below to be fused into two fused ops -
        # because we cannot have reduces with both R and C dims in one fused op.
        a2 = pybuda.op.ReduceAvg("reduce_avg_r", a1, dim=-2)
        a3 = pybuda.op.Add("add", act2, a2)

        a4 = pybuda.op.ReduceAvg("reduce_avg_c", a3, dim=-1)
        a5 = pybuda.op.Multiply("multiply", a4, act2)

        return a5

def test_fuse_reduce_rc(test_device, test_kind):
    pytest.skip("Skipping pending fix to #1258")

    if test_kind.is_training():
        # Training is currently not working, issue:
        # tenstorrent/pybuda#1211
        pytest.skip()

    os.environ["PYBUDA_FUSE_REDUCE"] = "1"
    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseReduceRC("fuse_reduce_RC"), [FuseReduceRC.shape1, FuseReduceRC.shape2],
                VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))

        if test_kind == TestKind.INFERENCE:
                check_fused_result(2, {"inputs": 3, "schedules": [1, 1]}, {"inputs": 3, "schedules": [1, 1]})

    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"]

class FuseForkBroadcast(pybuda.PyBudaModule):
    """
    Module with fork/reduce/broadcast/join, and optional second fused op
    """

    shape = (1, 1, 32, 128)

    def __init__(self, name, second_op: bool):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.second_op = second_op
        if self.second_op:
            self.weights2 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)

    def forward(self, act1, act2):
        a1 = pybuda.op.Matmul("matmul1", act1, self.weights1)

        # Expecting fusing of ops below
        a2 = pybuda.op.Add("add", act2, a1)
        a3 = pybuda.op.ReduceAvg("reduce_avg", a2, dim=-1)
        a4 = pybuda.op.Add("add_join", a2, a3)

        if not self.second_op:
            return a4

        a5 = pybuda.op.Matmul("matmul2", a4, self.weights2)

        # Expecting fusing of a different op here
        a6 = pybuda.op.Gelu("gelu1", a5)
        a7 = pybuda.op.Gelu("gelu2", a6)

        return a7

def test_fork_broadcast_join(test_device, test_kind):

    # Broadcast inputs that are reused
    os.environ["PYBUDA_FUSE_REDUCE"] = "1"
    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseForkBroadcast("fuse_fork_broadcast", second_op=False), [FuseForkBroadcast.shape, FuseForkBroadcast.shape], 
                VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))
        if test_kind == TestKind.INFERENCE:
            check_fused_result(1, {"inputs": 5, "schedules": [2, 2]})
    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"]

def test_two_fuse_ops(test_device, test_kind):

    # Add second op to the test
    os.environ["PYBUDA_FUSE_REDUCE"] = "1"
    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseForkBroadcast("two_fuse_ops", second_op=True), [FuseForkBroadcast.shape, FuseForkBroadcast.shape], 
                VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))
        if test_kind == TestKind.INFERENCE:
            check_fused_result(2, 
                    {"inputs": 5, "schedules": [2, 2]},
                    {"inputs": 1, "schedules": [2]})
    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"]


class FuseTileBroadcast(pybuda.PyBudaModule):
    """
    Module with a tile broadcast
    """

    shape = (1, 1, 256, 512)

    def __init__(self, name, dim):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        if dim == "r":
            self.bias1 = pybuda.Parameter(1, self.shape[-1], requires_grad=True)
        else:
            self.bias1 = pybuda.Parameter(self.shape[-2], 1, requires_grad=True)

    def forward(self, act1):
        a1 = pybuda.op.Matmul("matmul1", act1, self.weights1)

        # Expecting fusing of op below, which will have a tile broadcast added
        a2 = pybuda.op.Add("add", a1, self.bias1)
        return a2

@pytest.mark.parametrize("dim", ["r", "c"])
def test_tile_broadcast(test_device, test_kind, dim):

    # Broadcast inputs that are reused
    os.environ["PYBUDA_FUSE_REDUCE"] = "1"
    os.environ["PYBUDA_NO_FUSE_MATMUL_BIAS"] = "1"
    os.environ["PYBUDA_DISABLE_TILE_BROADCAST_CONSTEVAL"] = "1"
    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseTileBroadcast("fuse_tile_broadcast", dim), [FuseTileBroadcast.shape], 
                VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))
        if test_kind == TestKind.INFERENCE:
            check_fused_result(1, {"inputs": 2, "schedules": [1]})
    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"]
        del os.environ["PYBUDA_NO_FUSE_MATMUL_BIAS"]
        del os.environ["PYBUDA_DISABLE_TILE_BROADCAST_CONSTEVAL"]


class FuseSoftmax(pybuda.PyBudaModule):
    """
    Module with a softmax
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name, dim):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.dim = dim

    def forward(self, act1):
        a1 = pybuda.op.Matmul("matmul1", act1, self.weights1)

        # Expecting fusing of op below
        a2 = pybuda.op.Softmax("softmax", a1, dim=self.dim)
        return a2

@pytest.mark.parametrize("dim", ["r", "c"])
def test_softmax(test_device, test_kind, dim):

    pybuda.set_configuration_options(enable_stable_softmax=False)

    os.environ["PYBUDA_FUSE_REDUCE"] = "1"

    dim_index = -1 if dim == "c" else -2
    try: 
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseSoftmax("fuse_softmax", dim_index), [FuseSoftmax.shape], 
                VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))
        if test_kind == TestKind.INFERENCE:
            check_fused_result(1, {"inputs": 4, "schedules": [1, 3]})
    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"]

class FuseLayernorm(pybuda.PyBudaModule):
    """
    Module with a layernorm
    """

    def __init__(self, name, shape):
        super().__init__(name)
        self.shape = shape
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.ln_weights = pybuda.Parameter(1, self.shape[-1], requires_grad=True)
        self.ln_bias = pybuda.Parameter(1, self.shape[-1], requires_grad=True)

    def forward(self, act1):
        a1 = pybuda.op.Matmul("matmul1", act1, self.weights1)

        # Expecting fusing of op below
        a2 = pybuda.op.Layernorm("layernorm", a1, self.ln_weights, self.ln_bias)
        return a2

@pytest.mark.parametrize("fuse_reduce", [True, False], ids=["fuse_reduce", "no_reduce"])
@pytest.mark.parametrize("rows", [1, 2])
def test_layernorm(test_device, test_kind, fuse_reduce, rows):

    shape = (1, 1, 32*rows, 256)
    if fuse_reduce:
        os.environ["PYBUDA_FUSE_REDUCE"] = "1"

    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseLayernorm(f"fuse_layernorm_{'reduce' if fuse_reduce else 'no_reduce'}", shape), [shape], 
                VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))

        if test_kind == TestKind.INFERENCE:
            if fuse_reduce:
                check_fused_result(1, {"inputs": 11, "schedules": [1, 4, 9]})
            else:
                check_fused_result(2, {"inputs": 3, "schedules": [2]}, {"inputs": 6, "schedules": [7]})

    finally:
        if fuse_reduce:
            del os.environ["PYBUDA_FUSE_REDUCE"] 

class FuseMatmulBias(pybuda.PyBudaModule):
    """
    Module with matmuls+bias
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.weights2 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.bias1 = pybuda.Parameter((100.0 * torch.rand(1, self.shape[-1])).detach(), requires_grad=True)
        self.bias2 = pybuda.Parameter((100.0 * torch.rand(1, self.shape[-1])).detach(), requires_grad=True)

    def forward(self, act1):
        a1 = pybuda.op.Matmul("matmul1", act1, self.weights1) + self.bias1
        a2 = pybuda.op.Sqrt("sqrt", a1)
        a3 = pybuda.op.Matmul("matmul2", a2, self.weights2) + self.bias2
        return a3

def test_matmul_bias(test_device, test_kind):
    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(FuseMatmulBias("fuse_matmul_bias"), [FuseMatmulBias.shape], 
        VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                relative_atol=relative_atol,
                pcc=pcc))

class FuseMatmulGelu(pybuda.PyBudaModule):
    """
    Module with matmuls+gelu
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.weights2 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.weights3 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.bias1 = pybuda.Parameter((100.0 * torch.rand(1, self.shape[-1])).detach(), requires_grad=True)

    def forward(self, act1):
        a1 = pybuda.op.Matmul("matmul1", act1, self.weights1, self.bias1)
        a2 = pybuda.op.Gelu("gelu1", a1)
        a3 = pybuda.op.Matmul("matmul2", a2, self.weights2)
        a4 = pybuda.op.Gelu("gelu2", a3)
        a5 = pybuda.op.Matmul("matmul3", a4, self.weights3)
        return a5

def test_matmul_gelu(test_device, test_kind):
    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(FuseMatmulGelu("fuse_matmul_gelu"), [FuseMatmulGelu.shape],
        VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                relative_atol=relative_atol,
                pcc=pcc))

class FuseTwoLayernorm(pybuda.PyBudaModule):
    """
    Module with two layernorms
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.weights2 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.ln_weights1 = pybuda.Parameter(1, self.shape[-1], requires_grad=True)
        self.ln_bias1 = pybuda.Parameter(1, self.shape[-1], requires_grad=True)
        self.ln_weights2 = pybuda.Parameter(1, self.shape[-1], requires_grad=True)
        self.ln_bias2 = pybuda.Parameter(1, self.shape[-1], requires_grad=True)

    def forward(self, act1):
        a1 = pybuda.op.Matmul("matmul1", act1, self.weights1)

        # Two layernorms with matmul in between. We want to see the fused op reused twice.
        a2 = pybuda.op.Layernorm("layernorm1", a1, self.ln_weights1, self.ln_bias1)
        a3 = pybuda.op.Matmul("matmul2", a2, self.weights2)
        a4 = pybuda.op.Layernorm("layernorm2", a3, self.ln_weights2, self.ln_bias2)
        return a4

def test_layernorm_reuse(test_device):

    os.environ["PYBUDA_FUSE_REDUCE"] = "1"

    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseTwoLayernorm(f"fuse_two_layernorm"), [FuseTwoLayernorm.shape], 
                VerifyConfig(test_kind=TestKind.INFERENCE, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))

        # reuse happens in lowering to netlist, so the check has to be post-netlist.... 
        #check_fused_result(1, {"inputs": 9, "schedules": [1, 3, 10]})

    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"] 


class FuseReduceMax(pybuda.PyBudaModule):
    """
    Simple module with a reduce_max in the middle, or at the end, of a fused sequence
    """

    shape = (1, 1, 32, 128)

    def __init__(self, name, middle: bool):
        super().__init__(name)
        self.weights1 = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.middle = middle

    def forward(self, act1, act2):
        a1 = pybuda.op.Matmul("matmul", act1, self.weights1)

        # Expecting fusing of ops below
        a2 = pybuda.op.Add("add", act2, a1)
        a3 = pybuda.op.ReduceMax("reduce_max", a2, dim=-1)
        if (self.middle):
            a3 = pybuda.op.Reciprocal("reciprocal", a3)
        return a3

@pytest.mark.parametrize("middle", (True, False), ids=["middle", "end"])
@pytest.mark.skip(reason="Reduce max not supported yet in fusing")
def test_fuse_reduce_max(test_device, middle, test_kind):

    os.environ["PYBUDA_FUSE_REDUCE"] = "1"
    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseReduceMax("fuse_reduce_max_" + ("middle" if middle else "end"), middle), [FuseReduce.shape, FuseReduce.shape], 
                VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))

        if test_kind == TestKind.INFERENCE:
            if middle:
                check_fused_result(1, {"inputs": 3, "schedules": [2, 1]})
            else:
                check_fused_result(1, {"inputs": 3, "schedules": [2]})

    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"]


class FuseSelect(pybuda.PyBudaModule):
    """
    Simple model with one select/splice operation in it. This op should not be fused.
    """

    shape = (1, 2, 64, 128)

    def __init__(self, name, dim, index, length):
        super().__init__(name)
        self.dim = dim
        self.index = index
        self.length = length

    def forward(self, act1, act2):
        #These two ops shouldn't be fused since select op is not allowed for fusing.
        a1 = pybuda.op.Add("add0", act1, act2)
        a2 = pybuda.op.Select("select0", a1, self.dim, (self.index, self.length)) 
        return a2

def test_fuse_select(test_device, test_kind):
    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(FuseSelect("fuse_select", 2, 0, 32), [FuseSelect.shape, FuseSelect.shape],
        VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
            relative_atol=relative_atol, pcc=pcc))

    #Expectation is that ops are not fused since select op is not supported for fusing.
    check_fused_result(0)


class FuseBuffer(pybuda.PyBudaModule):
    """
    Simple model with buffer operation in it. This op shouln't be fused
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, act1, act2):
        #These two ops shouldn't be fused since buffer op is not allowed for fusing.
        a1 = pybuda.op.Add("add0", act1, act2)
        a2 = pybuda.op.Buffer("buffer0", a1)
        return a2

def test_fuse_buffer(test_device, test_kind):
    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(FuseBuffer("fuse_buffer"), [FuseBuffer.shape, FuseBuffer.shape],
        VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
            relative_atol=relative_atol, pcc=pcc))

    #Expectation is that ops are not fused since buffer op is not supported for fusing.
    check_fused_result(0)


class FuseEpochAndChipBreak(pybuda.PyBudaModule):
    """
    Simple module with epoch and chip breaks.
    """

    shape = (1, 1, 64, 64)

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, act1, act2):
        # Make model with chip and epoch breaks
        a1 = pybuda.op.Add("add0", act1, act2)
        a2 = pybuda.op.Buffer("buffer_a", a1)
        a3 = pybuda.op.Buffer("buffer_b", a2)
        a4 = pybuda.op.Buffer("buffer_c", a2)
        a5 = pybuda.op.Add("add1", a3, a4)

        pybuda.set_epoch_break("buffer_a")
        pybuda.set_chip_break("buffer_b")
        pybuda.set_epoch_break("buffer_c")

        return a5

def test_fuse_epoch_and_chip_break(test_device, test_kind):
    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(FuseEpochAndChipBreak("fuse_epoch_and_chip_break"), [FuseEpochAndChipBreak.shape, FuseEpochAndChipBreak.shape],
        VerifyConfig(test_kind=test_kind, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
            relative_atol=relative_atol, pcc=pcc))

    # At the moment expectation is that model run was successful and that break ops are not fused.
    check_fused_result(0)
    

class FuseSimpleTileBroadcast(pybuda.PyBudaModule):
    """
    Module with a tile broadcast
    """
    
    def __init__(self, name):
        super().__init__(name)
        self.bias1 = pybuda.Parameter(1, 1, requires_grad=False)

    def forward(self, act1):
        a1 = pybuda.op.Subtract("sub", act1, self.bias1)
        return a1

def test_simple_tile_broadcast_RC(test_device):
    shape = (1, 1, 64, 64)
    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    os.environ["PYBUDA_DISABLE_TILE_BROADCAST_CONSTEVAL"] = "1"
    try:
        verify_module(FuseSimpleTileBroadcast("fuse_simple_tile_broadcast_RC"), [shape], 
                VerifyConfig(test_kind=TestKind.INFERENCE, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))
        
        # Due to shape this will lower to 2 tile broadcasts and 2 broadcasts op.
        # Since sub has 2 broadcasts it can't be fused nor it can cosume tile broadcast.
        # Only tile broadcast ops will be fused together.
        check_fused_result(1, {"inputs": 3, "schedules": [1, 1]})
    finally:
        del os.environ["PYBUDA_DISABLE_TILE_BROADCAST_CONSTEVAL"]

def test_simple_tile_broadcast_C(test_device):
    shape = (1, 1, 1, 64)
    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    os.environ["PYBUDA_DISABLE_TILE_BROADCAST_CONSTEVAL"] = "1"
    try:
        verify_module(FuseSimpleTileBroadcast("fuse_simple_tile_broadcast_C"), [shape], 
                VerifyConfig(test_kind=TestKind.INFERENCE, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol,
                    pcc=pcc))
        
        # Due to shape this will lower to 1 tile broadcast and 1 broadcast op.
        # Tile broadcast should be merged to sub and sub shoud be fused.
        check_fused_result(1, {"inputs": 2, "schedules": [1]})
    finally:
        del os.environ["PYBUDA_DISABLE_TILE_BROADCAST_CONSTEVAL"]

class FuseBroadcastAsLHSOfMatmul(pybuda.PyBudaModule):
    """
    Module with brodacast C, that forces u_block_order R
    """

    shape = (1, 1, 32, 128)

    def __init__(self, name):
        super().__init__(name)
        self.weights = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)

    def forward(self, act1, act2):

        # Expecting fusing of ops below and having broadcast C
        a1 = pybuda.op.Add("add", act1, act2)
        a2 = pybuda.op.ReduceAvg("reduce_avg", a1, dim=-1)
        a3 = pybuda.op.Add("add_join", a1, a2)

        # Have fused op as LHS argument
        a4 = pybuda.op.Matmul("matmul", a3, self.weights)
        return a4

# Test that fusing broacast C as LHS argument of matmul will work.
def test_fuse_broadcast_c_as_lhs_matmul(test_device):

    os.environ["PYBUDA_FUSE_REDUCE"] = "1"
    try:
        relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
        verify_module(FuseBroadcastAsLHSOfMatmul("fuse_broadcast_c_as_lhs_matmul"), [FuseBroadcastAsLHSOfMatmul.shape, FuseBroadcastAsLHSOfMatmul.shape], 
                VerifyConfig(test_kind=TestKind.INFERENCE, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                    relative_atol=relative_atol, pcc=pcc))
    finally:
        del os.environ["PYBUDA_FUSE_REDUCE"]

class FuseBroadcastOutputOp(pybuda.PyBudaModule):
    """
    Module with both broadcast and tile broadcast on output op of the fused op.
    Tests handling of tms in fused op shape calculation and in evaluation of fused ops.
    """

    shape = (1, 1, 32, 128)

    def __init__(self, name):
        super().__init__(name)
        self.weights = pybuda.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)

    def forward(self, act1, act2):

        act1_reduced = pybuda.op.ReduceAvg("reduce_avg_0", act1, dim=-1)
        act2_reduced = pybuda.op.ReduceAvg("reduce_avg_1", act2, dim=-1)

        # Inputs to the multiply operation (which will be the output op of the fused op)
        # will have both the "broadcast" and "tile_broadcast" tms.
        a1 = pybuda.op.Add("add", act1_reduced, act2_reduced)
        a2 = pybuda.op.Reciprocal("reciprocal", a1)
        a3 = pybuda.op.Broadcast("broadcast", a2, -1, self.shape[-1])
        a4 = pybuda.op.Multiply("multiply", a3, act1_reduced)

        a5 = pybuda.op.Matmul("matmul", a4, self.weights)
        return a5

# Test that all broadcast operations inside the fused op will be treated correctly.
def test_fuse_broadcast_output_op(test_device):

    relative_atol, pcc = get_relaxed_atol_pcc(True, test_device)
    verify_module(FuseBroadcastOutputOp("fuse_broadcast_output_op"), [FuseBroadcastOutputOp.shape, FuseBroadcastOutputOp.shape],
            VerifyConfig(test_kind=TestKind.INFERENCE, skip_shutdown=True, arch=test_device.arch, devtype=test_device.devtype,
                relative_atol=relative_atol, pcc=pcc, verify_all=True))
