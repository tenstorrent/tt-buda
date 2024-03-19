# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import pybuda
import pybuda.op
from pybuda import PyBudaModule, Tensor
from pybuda._C.balancer import BalancerConfig, PolicyType
from pybuda._C import run_pre_placer_buda_passes
from .common import compile, ModuleBuilder
from pybuda.verify import verify_module, VerifyConfig, TestKind
from pybuda._C.backend_api import BackendDevice


@pytest.mark.parametrize("mode", ["inference"])
@pytest.mark.parametrize("microbatch_size", (1, 8), ids=("mb1", "mb8"))
def test_intra_epoch_relay_queue(mode, microbatch_size):
    def linked_list(activations):
        activations = pybuda.op.Buffer(f"buffer_pre", activations)
        # num_entries=microbatch_size, so if the queue is statically allocated, it still has enough memory
        activations = pybuda.op.DRAMQueue(f"buffering_queue", activations, num_entries=microbatch_size)
        activations = pybuda.op.Buffer(f"buffer_post", activations)
        return activations

    module = ModuleBuilder(linked_list)
    verify_module(module, [(microbatch_size, 64, 64)],
            VerifyConfig(test_kind=TestKind.INFERENCE, run_net2pipe=True, arch=BackendDevice.Grayskull))


@pytest.mark.parametrize("mode", ["inference", "training"])
def test_sanity(mode):
    shape = (1, 1, 64, 64)
    training = mode == "training"
    parameters = {
        "weights1": pybuda.Parameter(
            torch.rand(*shape, requires_grad=True)
        ),
        "weights2": pybuda.Parameter(
            torch.rand(*shape, requires_grad=True)
        ),
    }

    @compile(
        compiler_cfg=pybuda.CompilerConfig(enable_training=training),
        verify_cfg=pybuda.VerifyConfig(),
    )
    def test(act1, *, weights1, weights2):
        m1 = pybuda.op.Matmul("matmul1", act1, weights1)
        m2 = pybuda.op.Matmul("matmul2", act1, weights2)
        m1e = pybuda.op.Exp("exp", m1)
        return pybuda.op.Add("add", m1e, m2)

    act1 = Tensor.create_from_torch(torch.rand(*shape))
    outputs = test(act1, weights1=parameters["weights1"], weights2=parameters["weights2"])


@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize("num_ops_left_branch", [0, 1])
@pytest.mark.parametrize("num_ops_right_branch", [2])
def test_two_branch_fork_join_branch_asymmetry(
    mode, num_ops_left_branch, num_ops_right_branch
):
    training = mode == "training"
    shape = (1, 1, 64, 64)

    @compile(
        compiler_cfg=pybuda.CompilerConfig(enable_training=training),
        verify_cfg=pybuda.VerifyConfig(),
    )
    def two_branch_fork_join_branch_asymmetry(act1):
        left_branch = act1
        right_branch = act1

        for i in range(num_ops_left_branch):
            left_branch = pybuda.op.Buffer(f"buffer_left_{i}", left_branch)

        for i in range(num_ops_right_branch):
            right_branch = pybuda.op.Buffer(f"buffer_right_{i}", right_branch)

        return pybuda.op.Add("add", left_branch, right_branch)

    act1 = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    outputs = two_branch_fork_join_branch_asymmetry(act1)



@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize("num_ops_left_branch", [1])
@pytest.mark.parametrize("num_ops_right_branch", [2])
def test_two_branch_fork_join_branch_asymmetry_with_buffering_queue(
    mode, num_ops_left_branch, num_ops_right_branch
):
    training = mode == "training"
    microbatch_size = 1
    shape = (microbatch_size, 1 , 64, 64)

    @compile(
        compiler_cfg=pybuda.CompilerConfig(enable_training=training),
        verify_cfg=pybuda.VerifyConfig(),
    )
    def two_branch_fork_join_branch_asymmetry(act1):
        left_branch = act1
        right_branch = act1

        for i in range(num_ops_left_branch):
            left_branch = pybuda.op.Buffer(f"buffer_left_{i}", left_branch)

        # num_entries=microbatch_size, so if the queue is statically allocated, it still has enough memory
        left_branch = pybuda.op.DRAMQueue(f"buffering_queue", left_branch, num_entries=microbatch_size)

        for i in range(num_ops_right_branch):
            right_branch = pybuda.op.Buffer(f"buffer_right_{i}", right_branch)

        return pybuda.op.Add("add", left_branch, right_branch)

    act1 = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    outputs = two_branch_fork_join_branch_asymmetry(act1)


# TODO: parametrize training mode
# TODO: sweep over different op-types, reduce, matmul etc
@pytest.mark.skip(reason="TODO: outstanding support")
@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize("num_ops_first_branch", [0])
@pytest.mark.parametrize("num_ops_second_branch", [2])
@pytest.mark.parametrize("num_ops_third_branch", [4])
def test_three_branch_fork_join_branch_asymmetry(
    mode, num_ops_first_branch, num_ops_second_branch, num_ops_third_branch
):
    training = mode == "training"
    shape = (1, 1, 64, 64)

    @compile(
        compiler_cfg=pybuda.CompilerConfig(enable_training=training),
        verify_cfg=pybuda.VerifyConfig(),
    )
    def three_branch_fork_join_branch_asymmetry(act1):
        first_branch = act1
        second_branch = act1
        third_branch = act1

        for i in range(num_ops_first_branch):
            first_branch = pybuda.op.Buffer(f"branch0_buffer_{i}", first_branch)

        for i in range(num_ops_second_branch):
            second_branch = pybuda.op.Buffer(f"branch1_buffer_{i}", second_branch)

        for i in range(num_ops_third_branch):
            third_branch = pybuda.op.Buffer(f"branch2_buffer_{i}", third_branch)

        partial_sum = pybuda.op.Add("partial_add", first_branch, second_branch)
        sum = pybuda.op.Add("final_add", partial_sum, third_branch)
        return sum

    act1 = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    outputs = three_branch_fork_join_branch_asymmetry(act1)


@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize("nesting_level_left_branch", [1])
@pytest.mark.parametrize("num_ops_left_branch", [1])
@pytest.mark.parametrize("num_ops_right_branch", [2])
def test_nested_fork(
    mode, nesting_level_left_branch, num_ops_left_branch, num_ops_right_branch
):
    training = mode == "training"
    shape = (1, 1, 64, 64)

    @compile(
        compiler_cfg=pybuda.CompilerConfig(enable_training=training),
        verify_cfg=pybuda.VerifyConfig(),
    )
    def nested_fork(act1):
        def instantiate_fork_join(nesting_level, fork_node):
            if nesting_level <= 0:
                return fork_node

            left_branch = fork_node
            right_branch = fork_node

            for i in range(num_ops_left_branch):
                left_branch = pybuda.op.Buffer(
                    f"nesting_level_{nesting_level}_buffer_left_{i}", left_branch
                )

            left_branch = instantiate_fork_join(nesting_level - 1, left_branch)

            for i in range(num_ops_right_branch):
                right_branch = pybuda.op.Buffer(
                    f"nesting_level_{nesting_level}_buffer_right_{i}", right_branch
                )

            return pybuda.op.Add(
                f"nesting_level_{nesting_level}_add", left_branch, right_branch
            )

        return instantiate_fork_join(nesting_level_left_branch + 1, act1)

    act1 = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))
    outputs = nested_fork(act1)
