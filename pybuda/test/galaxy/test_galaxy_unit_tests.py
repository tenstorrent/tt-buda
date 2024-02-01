# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from inspect import signature
import pytest
import copy
import random
import math

import torch
from transformers import (
    BertModel,
    BertConfig,
    BertForPreTraining,
    BertTokenizer,
    BertForQuestionAnswering,
)

import pybuda
import pybuda.op
from pybuda import (
    Tensor,
    CompilerConfig,
    DataFormat,
    PyTorchModule,
)
from test.bert.modules import (
    PyBudaBertMHA,
    PyBudaBertEncoder,
    PyBudaFeedForward,
    PyBudaPredictionHeadDecoder,
    PyBudaPredictionHeadTransform,
    get_bert_parameters,
)
from pybuda.verify import verify_module, verify_module_pipeline, VerifyConfig, TestKind
from pybuda._C.backend_api import BackendDevice, BackendType
from test.common import compile, device, ModuleBuilder
from pybuda.config import _get_global_compiler_config, _clear_global_compiler_config
from pybuda.ttdevice import get_device_config

from loguru import logger

random.seed(456)
ONE_SHELF_RUNTIME_PARAMS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "one_shelf_runtime_params.yaml"
)
ONE_SHELF_ETH_CONNECTIONS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "one_shelf_eth_connections.yaml"
)

BACKEND_CLUSTER_DESC = os.path.join(os.getenv("BUDA_HOME"), "cluster_desc.yaml")


############################################################################################
################################ TWO CHIP UNIT TESTS #######################################
############################################################################################
def get_two_chip_op_tests(subset, c0, c1):
    # Keep names unique and return same number of outputs as inputs to enable pipelining unit tests
    def two_chip_simple_unary_to_unary(act):
        pybuda.set_chip_break("unary1_A")
        pybuda.override_dram_queue_placement("inputs", chip_id=c0, channel=0)

        unary0 = pybuda.op.Exp("unary0_A", act)
        unary1 = pybuda.op.Gelu("unary1_A", unary0)
        return unary1

    def two_chip_eth_gather(act):
        pybuda.set_chip_break("unary1_B")
        pybuda.override_dram_queue_placement("inputs", chip_id=c0, channel=0)
        pybuda.override_op_size("unary0_B", (2, 2))
        pybuda.override_op_size("unary1_B", (1, 1))

        unary0 = pybuda.op.Buffer("unary0_B", act)
        unary1 = pybuda.op.Gelu("unary1_B", unary0)
        return unary1

    def two_chip_eth_multicast(act0, act1):
        pybuda.set_chip_break("unary1_C")
        pybuda.override_dram_queue_placement("input_0_unary0_C", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_0_unary1_C", chip_id=c1, channel=0)
        pybuda.override_op_size("unary0_C", (2, 1))
        pybuda.override_op_size("unary1_C", (1, 2))
        pybuda.override_op_size("matmul0_C", (2, 2))

        unary0 = pybuda.op.Sqrt("unary0_C", act0)
        unary1 = pybuda.op.Gelu("unary1_C", act1)
        matmul0 = pybuda.op.Matmul("matmul0_C", unary0, unary1)
        return matmul0, unary0

    def two_chip_eth_gather_multicast(act0, act1):
        pybuda.set_chip_break("unary1_D")
        pybuda.override_dram_queue_placement("input_0_unary0_D", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_0_unary1_D", chip_id=c1, channel=0)
        pybuda.override_op_size("unary0_D", (2, 1))
        pybuda.override_op_size("unary1_D", (2, 2))
        pybuda.override_op_size("matmul0_D", (2, 2))

        unary0 = pybuda.op.Gelu("unary0_D", act0)
        unary1 = pybuda.op.Buffer("unary1_D", act1)
        matmul0 = pybuda.op.Matmul("matmul0_D", unary0, unary1)
        return matmul0, unary1

    def two_chip_dram_buf_fork_c0_to_c0c1(act):
        pybuda.set_chip_break("unary1_E")
        pybuda.override_dram_queue_placement("inputs", chip_id=c0, channel=0)

        unary0 = pybuda.op.Gelu("unary0_E", act)
        unary1 = pybuda.op.Buffer("unary1_E", unary0)
        add0 = pybuda.op.Add("add_E", unary1, act)
        return add0

    def two_chip_l1_buf_fork_c0_to_c1c1_same_consumer(act):
        pybuda.set_chip_break("add0_F")
        pybuda.override_dram_queue_placement("inputs", chip_id=c0, channel=0)

        unary0 = pybuda.op.Buffer("unary0_F", act)
        add0 = pybuda.op.Add("add0_F", unary0, unary0)
        return add0

    def two_chip_binary_inputs_c1_tensix_c1_dram(act0, act1):
        pybuda.set_chip_break("unary0_G")
        pybuda.override_dram_queue_placement("input_0_nop0_G", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_1_add0_G", chip_id=c1, channel=0)

        nop0 = pybuda.op.Buffer("nop0_G", act0)
        unary0 = pybuda.op.Buffer("unary0_G", nop0)
        add0 = pybuda.op.Add("add0_G", unary0, act1)
        return add0, nop0

    def two_chip_multiply_inputs_c0_tensix_c1_dram(act0, act1):
        pybuda.set_chip_break("multiply0_H")
        pybuda.override_dram_queue_placement("input_0_nop0_H", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement(
            "input_1_multiply0_H", chip_id=c1, channel=0
        )

        nop0 = pybuda.op.Buffer("nop0_H", act0)
        multiply0 = pybuda.op.Multiply("multiply0_H", nop0, act1)
        return multiply0, nop0

    def two_chip_binary_inputs_c0_tensix_c1_tensix(act0, act1, act2):
        pybuda.set_chip_break("nop0_I")
        pybuda.override_dram_queue_placement("input_0_add0_I", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_1_add0_I", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_0_nop0_I", chip_id=c1, channel=0)

        add0 = pybuda.op.Add("add0_I", act0, act1)
        nop0 = pybuda.op.Buffer("nop0_I", act2)
        add1 = pybuda.op.Add("add1_I", add0, nop0)
        return add1, add0, nop0

    def two_chip_multiply_inputs_c0_tensix_c0_tensix(act0, act1, act2):
        pybuda.set_chip_break("multiply0_J")
        pybuda.override_dram_queue_placement("input_0_add0_J", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_1_add0_J", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_0_nop0_J", chip_id=c0, channel=0)

        add0 = pybuda.op.Add("add0_J", act0, act1)
        nop0 = pybuda.op.Buffer("nop0_J", act2)
        multiply0 = pybuda.op.Multiply("multiply0_J", add0, nop0)
        return multiply0, add0, nop0

    def two_chip_matmul_inputs_c0_dram_c1_tensix(act0, act1, act2):
        pybuda.set_chip_break("add0_K")
        pybuda.override_dram_queue_placement("input_0_nop0_K", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_0_nop1_K", chip_id=c0, channel=0)
        pybuda.override_dram_queue_placement("input_0_matmul0_K", chip_id=c0, channel=0)

        nop0 = pybuda.op.Buffer("nop0_K", act0)
        nop1 = pybuda.op.Buffer("nop1_K", act1)
        add0 = pybuda.op.Add("add0_K", nop0, nop1)
        matmul0 = pybuda.op.Matmul("matmul0_K", act2, add0)
        return matmul0, add0, nop1

    if subset == "sanity":
        return [two_chip_simple_unary_to_unary]
    elif subset == "full":
        test_list = [
            two_chip_simple_unary_to_unary,
            two_chip_eth_gather,
            two_chip_eth_multicast,
            two_chip_eth_gather_multicast,
            two_chip_dram_buf_fork_c0_to_c0c1,
            two_chip_l1_buf_fork_c0_to_c1c1_same_consumer,
            two_chip_binary_inputs_c1_tensix_c1_dram,
            two_chip_multiply_inputs_c0_tensix_c1_dram,
            two_chip_binary_inputs_c0_tensix_c1_tensix,
            two_chip_multiply_inputs_c0_tensix_c0_tensix,
            two_chip_matmul_inputs_c0_dram_c1_tensix,
        ]
        return test_list
    else:
        assert False, "Unrecognized two chip test list"


############################################################################################
################################ FOUR CHIP UNIT TESTS ######################################
############################################################################################
def create_four_chip_test_modules(c0, c1, c2, c3):
    # create pipelines out of the two chip unit tests (input shapes need to match between the tests)
    full_test_list = get_two_chip_op_tests("full", c0, c1)

    random.seed(int(f"{c0}{c1}{c2}{c3}"))
    random_test = random.choice(full_test_list)
    matching_input_shape_tests = [
        test
        for test in full_test_list
        if len(signature(test).parameters) == len(signature(random_test).parameters)
    ]

    return random.sample(matching_input_shape_tests, 3)


def get_four_chip_layouts(chip_locations):
    # Layouts generated for a 2x4 grid of modules
    # 0            1           2           3
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ░░███░███░░░░░▒▒▒░▒▒▒░░░░░███░▒▒▒░░░░░▒▒▒░▒▒▒░░
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ░░▒▒▒░▒▒▒░░░░░███░███░░░░░███░▒▒▒░░░░░███░▒▒▒░░
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ░░▒▒▒░▒▒▒░░░░░███░███░░░░░▒▒▒░███░░░░░███░███░░
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ░░███░███░░░░░▒▒▒░▒▒▒░░░░░▒▒▒░███░░░░░▒▒▒░███░░
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    #
    #  4           5           6           7
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ░░███░▒▒▒░░░░░▒▒▒░▒▒▒░░░░░███░▒▒▒░░░░░▒▒▒░▒▒▒░░
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ░░▒▒▒░███░░░░░███░▒▒▒░░░░░███░▒▒▒░░░░░███░▒▒▒░░
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ░░███░▒▒▒░░░░░▒▒▒░███░░░░░███░███░░░░░███░███░░
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    # ░░▒▒▒░███░░░░░███░███░░░░░▒▒▒░▒▒▒░░░░░███░▒▒▒░░
    # ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

    # TODO: ask for api to return device_cfg.get_ethernet_connections()
    return 0


############################################################################################
######################################### UTILS ############################################
############################################################################################
def get_galaxy_chip_adjacency_list(eth_connections):
    # {chan_a: (chip_b, chan_b) ... }... }
    galaxy_adjacent_chips = {}
    for chip_a, channels_a in eth_connections.items():
        if chip_a not in galaxy_adjacent_chips.keys():
            galaxy_adjacent_chips[chip_a] = set()
        for chip_b_chan_pair in channels_a.values():
            galaxy_adjacent_chips[chip_a].add(chip_b_chan_pair[0])
    return galaxy_adjacent_chips


def get_chip_ids_for_galaxy(chip_ids, full_galaxy=False, galaxy_adjacent_chips=None):
    # have to insert mmio chip to beginning of chip_ids
    # insert all chips if full_galaxy is true, useful when chip_ids might require empty graphs on all chips
    mmio_chip = 0
    if mmio_chip in chip_ids:
        chip_ids.remove(mmio_chip)
    chip_ids.insert(0, mmio_chip)

    if full_galaxy:
        assert galaxy_adjacent_chips is not None
        chip_ids.extend(
            [
                empty_chip_id
                for empty_chip_id in galaxy_adjacent_chips.keys()
                if empty_chip_id not in chip_ids
            ]
        )
    return chip_ids


def reset_pybuda_between_tests():
    pybuda.pybuda_reset()
    pybuda.set_configuration_options(
        backend_cluster_descriptor_path=BACKEND_CLUSTER_DESC,
        output_queues_on_host=False,
        enable_consteval=False,
    )


@pytest.fixture
def scan_chip(request):
    return request.config.getoption("--scan_chip")


@pytest.mark.parametrize("test_level", ["sanity", "full"])
def test_galaxy_scan_chip_pairs(scan_chip, test_level, microbatch):
    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0

    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )

    galaxy_adjacent_chips = get_galaxy_chip_adjacency_list(
        device_cfg.get_ethernet_connections()
    )

    skip_mmio = False
    if scan_chip == "all_non_mmio_chips":
        skip_mmio = True
    if scan_chip in ["all_chips", "all_non_mmio_chips"]:
        chips_to_scan = list(galaxy_adjacent_chips.keys())
    else:
        chips_to_scan = [int(chip) for chip in scan_chip.split(",")]
        assert all(chip in galaxy_adjacent_chips.keys() for chip in chips_to_scan)

    logger.info(f"Running tests on chips {chips_to_scan}")
    for chip_a in chips_to_scan:
        for chip_b in galaxy_adjacent_chips[chip_a]:
            if skip_mmio and 0 in [chip_a, chip_b]:
                continue
            test_list = get_two_chip_op_tests(test_level, chip_a, chip_b)
            for test in test_list:
                chip_ids = get_chip_ids_for_galaxy([chip_a, chip_b])
                reset_pybuda_between_tests()

                module = ModuleBuilder(test)
                num_inputs = len(signature(test).parameters)
                inputs_shape = [(microbatch, 1, 64, 64)] * num_inputs
                verify_module(
                    module,
                    inputs_shape,
                    VerifyConfig(
                        test_kind=TestKind.INFERENCE,
                        pcc=0.95,
                        devtype=devtype,
                        arch=arch,
                        run_net2pipe=True,
                        chip_ids=chip_ids,
                    ),
                    inputs_centered_on_zero=True,
                )


@pytest.mark.parametrize("test_level", ["sanity", "full"])
def test_galaxy_two_hop_two_chip_tests(test_level, microbatch):
    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0

    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )

    galaxy_adjacent_chips = get_galaxy_chip_adjacency_list(
        device_cfg.get_ethernet_connections()
    )
    galaxy_two_hop_pairs = {}
    for chip_a in galaxy_adjacent_chips.keys():
        if chip_a not in galaxy_two_hop_pairs.keys():
            galaxy_two_hop_pairs[chip_a] = set()
        for one_hop_chip in galaxy_adjacent_chips[chip_a]:
            for two_hop_chip in galaxy_adjacent_chips[one_hop_chip]:
                if two_hop_chip != chip_a:
                    galaxy_two_hop_pairs[chip_a].add(two_hop_chip)

    for chip_a, two_hop_chips in galaxy_two_hop_pairs.items():
        for chip_b in two_hop_chips:
            test_list = get_two_chip_op_tests(test_level, chip_a, chip_b)
            for test in test_list:
                chip_ids = get_chip_ids_for_galaxy(
                    [chip_a, chip_b], True, galaxy_adjacent_chips
                )
                reset_pybuda_between_tests()

                module = ModuleBuilder(test)
                num_inputs = len(signature(test).parameters)
                inputs_shape = [(microbatch, 1, 64, 64)] * num_inputs
                verify_module(
                    module,
                    inputs_shape,
                    VerifyConfig(
                        test_kind=TestKind.INFERENCE,
                        pcc=0.95,
                        devtype=devtype,
                        arch=arch,
                        run_net2pipe=True,
                        chip_ids=chip_ids,
                    ),
                    inputs_centered_on_zero=True,
                )


def test_galaxy_four_chip_layouts(microbatch):
    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0

    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )

    galaxy_adjacent_chips = get_galaxy_chip_adjacency_list(
        device_cfg.get_ethernet_connections()
    )
    random.seed(213)

    four_chip_layouts = []
    for i in range(100):
        four_chip_layouts.append(random.sample(galaxy_adjacent_chips.keys(), 4))

    for chips in four_chip_layouts:
        logger.info(
            f"Running tests on chips {chips[0]}, {chips[1]}, {chips[2]}, {chips[3]}"
        )
        chip_ids = get_chip_ids_for_galaxy(chips, True, galaxy_adjacent_chips)

        reset_pybuda_between_tests()

        tests = create_four_chip_test_modules(chips[0], chips[1], chips[2], chips[3])
        modules = [ModuleBuilder(test) for test in tests]
        num_inputs = len(signature(tests[0]).parameters)
        inputs_shape = [(microbatch, 1, 64, 64)] * num_inputs
        verify_module_pipeline(
            modules,
            inputs_shape,
            VerifyConfig(
                test_kind=TestKind.INFERENCE,
                pcc=0.95,
                devtype=devtype,
                arch=arch,
                run_net2pipe=True,
                chip_ids=chip_ids,
            ),
            inputs_centered_on_zero=True,
        )


def test_galaxy_dram_buf_fork(microbatch):
    MAX_BUF_FORK = 8

    def dram_buf_fork(act):
        pybuda.override_dram_queue_placement("input_unary0", chip_id=0, channel=0)
        for i in range(MAX_BUF_FORK):
            pybuda.set_chip_break(f"unary{i}")

        op_list = [pybuda.op.Buffer(f"unary{i}", act) for i in range(MAX_BUF_FORK)]
        return op_list

    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0

    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )
    galaxy_adjacent_chips = get_galaxy_chip_adjacency_list(
        device_cfg.get_ethernet_connections()
    )

    eight_chip_layouts = []
    for i in range(10):
        eight_chip_layouts.append(
            random.sample(galaxy_adjacent_chips.keys(), MAX_BUF_FORK)
        )

    for chips in eight_chip_layouts:
        chip_ids = get_chip_ids_for_galaxy(chips, True, galaxy_adjacent_chips)

        reset_pybuda_between_tests()

        module = ModuleBuilder(dram_buf_fork)
        inputs_shape = [(microbatch, 1, 64, 64)]
        verify_module(
            module,
            inputs_shape,
            VerifyConfig(
                test_kind=TestKind.INFERENCE,
                pcc=0.95,
                devtype=devtype,
                arch=arch,
                run_net2pipe=True,
                chip_ids=chip_ids,
            ),
            inputs_centered_on_zero=True,
        )


def test_galaxy_l1_buf_fork(microbatch):
    MAX_BUF_FORK = 8

    def l1_buf_fork(act):
        pybuda.override_dram_queue_placement("input_unary0", chip_id=0, channel=0)
        unary0 = pybuda.op.Gelu("unary0", act)
        for i in range(MAX_BUF_FORK):
            pybuda.set_chip_break(f"unary{i}")

        op_list = [
            pybuda.op.Buffer(f"unary{i}", unary0) for i in range(1, MAX_BUF_FORK)
        ]
        return op_list

    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0

    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )
    galaxy_adjacent_chips = get_galaxy_chip_adjacency_list(
        device_cfg.get_ethernet_connections()
    )

    eight_chip_layouts = []
    for i in range(10):
        eight_chip_layouts.append(
            random.sample(galaxy_adjacent_chips.keys(), MAX_BUF_FORK)
        )

    for chips in eight_chip_layouts:
        chip_ids = get_chip_ids_for_galaxy(chips, True, galaxy_adjacent_chips)

        reset_pybuda_between_tests()

        module = ModuleBuilder(l1_buf_fork)
        inputs_shape = [(microbatch, 1, 64, 64)]
        verify_module(
            module,
            inputs_shape,
            VerifyConfig(
                test_kind=TestKind.INFERENCE,
                pcc=0.95,
                devtype=devtype,
                arch=arch,
                run_net2pipe=True,
                chip_ids=chip_ids,
            ),
            inputs_centered_on_zero=True,
        )


@pytest.mark.parametrize(
    "num_chips",
    [2, 4, 8, 12, 32, 64],
    ids=["chip2", "chip4", "chip8", "chip12", "chip32", "chip64"],
)
def test_full_grid_unary_nops_multichip(microbatch, num_chips):
    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0

    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )

    num_rows = device_cfg.grid_size.r
    num_cols = device_cfg.grid_size.c

    def full_grid_unary_nop(act):
        op_names = [
            f"nop{chip}_{op}" for chip in range(num_chips) for op in range(num_rows)
        ]
        for op in op_names:
            pybuda.override_op_size(op, (1, num_cols))

        nop = pybuda.op.Buffer(op_names[0], act)

        for i in range(1, len(op_names)):
            # set chip break after one chip is full of ops
            if i % num_rows == 0:
                pybuda.set_chip_break(op_names[i])
            nop = pybuda.op.Buffer(op_names[i], nop)
        return nop

    galaxy_adjacent_chips = get_galaxy_chip_adjacency_list(
        device_cfg.get_ethernet_connections()
    )
    chip_ids = get_chip_ids_for_galaxy(
        list(range(num_chips)), True, galaxy_adjacent_chips
    )

    reset_pybuda_between_tests()

    module = ModuleBuilder(full_grid_unary_nop)
    inputs_shape = [(microbatch, 4, 64, num_cols * 32)]
    verify_module(
        module,
        inputs_shape,
        VerifyConfig(
            test_kind=TestKind.INFERENCE,
            pcc=0.95,
            devtype=devtype,
            arch=arch,
            run_net2pipe=True,
            chip_ids=chip_ids,
        ),
        inputs_centered_on_zero=True,
    )

def test_vf_chip(microbatch, scan_chip):
    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0

    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )

    num_rows = 8
    num_cols = 8

    def large_matmul(act0, act1):
        matmul0 = pybuda.op.Matmul("matmul", act0, act1)
        pybuda.override_op_size("matmul", (num_rows, num_cols))
        return matmul0

    galaxy_adjacent_chips = get_galaxy_chip_adjacency_list(
        device_cfg.get_ethernet_connections()
    )
    if scan_chip == "all_chips":
        chips_to_scan = list(galaxy_adjacent_chips.keys())
    else:
        chips_to_scan = [int(chip) for chip in scan_chip.split(",")]
        assert all(chip in galaxy_adjacent_chips.keys() for chip in chips_to_scan)

    logger.info(f"Running tests on chips {chips_to_scan}")
    for chip in chips_to_scan:
        chip_ids = get_chip_ids_for_galaxy(
            [int(chip)], False, galaxy_adjacent_chips
        )

        reset_pybuda_between_tests()

        module = ModuleBuilder(large_matmul)
        inputs_shape = [(microbatch, 4, num_rows * 32, num_cols * 32), (microbatch, 4, num_rows * 32, num_cols * 32)]
        verify_module(
            module,
            inputs_shape,
            VerifyConfig(
                test_kind=TestKind.INFERENCE,
                pcc=0.95,
                devtype=devtype,
                arch=arch,
                run_net2pipe=True,
                chip_ids=chip_ids,
            ),
            inputs_centered_on_zero=True,
        )
