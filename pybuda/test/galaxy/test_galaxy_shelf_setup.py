# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from inspect import signature
import pytest
import copy
from collections import Counter
import random
import math


import pybuda
import pybuda.op
from pybuda import (
    Tensor,
    CompilerConfig,
    DataFormat,
    PyTorchModule,
)
from pybuda.verify import verify_module, verify_module_pipeline, VerifyConfig, TestKind
from pybuda._C.backend_api import BackendDevice, BackendType
from test.common import compile, device, ModuleBuilder
from pybuda.config import _get_global_compiler_config, _clear_global_compiler_config
from pybuda.ttdevice import get_device_config
from test_galaxy_unit_tests import (
    get_two_chip_op_tests,
    get_galaxy_chip_adjacency_list,
    get_chip_ids_for_galaxy,
    reset_pybuda_between_tests,
)

from loguru import logger

random.seed(456)
TWO_SHELF_RUNTIME_PARAMS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "two_shelf_runtime_params.yaml"
)
TWO_SHELF_ETH_CONNECTIONS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "two_shelf_eth_connections.yaml"
)
TWO_SHELF_GALAXY_IDS = [0, 15, 16, 13, 12, 11, 6, 5, 14, 3, 4, 7, 24, 39, 2, 40, 38, 25, 26, 37, 41, 42, 36, 27, 28, 31, 49, 50, 30, 29, 22, 21, 51, 52, 20, 23, 8, 19, 53, 54, 18, 9, 10, 17, 55, 56, 57, 58, 59, 60, 61, 62, 63, 48, 32, 33, 34, 35, 43, 44, 45, 46, 47, 1]


############################################################################################
######################################### UTILS ############################################
############################################################################################
def get_num_links_between_chips(device_cfg):
    eth_connections = device_cfg.get_ethernet_connections()
    links = {}
    for chip_a, channels_a in eth_connections.items():
        chip_a_connections = [x[0] for x in channels_a.values()]
        links[chip_a] = Counter(chip_a_connections)
    return links


def get_chip_connections_with_two_links(device_cfg):
    links = get_num_links_between_chips(device_cfg)
    chip_ids = []
    for chip_a, chip_a_links in links.items():
        for chip_b, count in chip_a_links.items():
            if count == 2:
                chip_ids.append([chip_a, chip_b])
    return chip_ids


@pytest.mark.parametrize("test_level", ["sanity", "full"])
def test_galaxy_shelf_connection_modules(test_level, microbatch):
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

    chip_ids = get_chip_connections_with_two_links(device_cfg)

    logger.info(f"Running tests on chips {chip_ids}")
    for chip_a, chip_b in chip_ids:
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


def test_two_shelves_full_grid_unary_ops(microbatch, test_device):
    compiler_cfg = _get_global_compiler_config()
    devtype = test_device.devtype
    arch = test_device.arch
    pybuda.pybuda_reset()
    if devtype == BackendType.Golden:
        pybuda.set_configuration_options(
            backend_runtime_params_path=TWO_SHELF_RUNTIME_PARAMS,
            backend_cluster_descriptor_path=TWO_SHELF_ETH_CONNECTIONS,
        )

    pybuda.set_configuration_options(
        output_queues_on_host=False, enable_consteval=False
    )

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

    chip_ids = TWO_SHELF_GALAXY_IDS

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


def test_two_shelves_full_eth_link_usage(microbatch, test_device):
    compiler_cfg = _get_global_compiler_config()
    devtype = test_device.devtype
    arch = test_device.arch
    num_chips = 64

    pybuda.pybuda_reset()
    if devtype == BackendType.Golden:
        pybuda.set_configuration_options(
            backend_runtime_params_path=TWO_SHELF_RUNTIME_PARAMS,
            backend_cluster_descriptor_path=TWO_SHELF_ETH_CONNECTIONS,
        )

    pybuda.set_configuration_options(
        output_queues_on_host=False, enable_consteval=False
    )

    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )

    get_num_links_between_chips(device_cfg)
    chip_ids = TWO_SHELF_GALAXY_IDS

    def full_eth_links_unary_nop(act):
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

    chip_ids = TWO_SHELF_GALAXY_IDS

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

