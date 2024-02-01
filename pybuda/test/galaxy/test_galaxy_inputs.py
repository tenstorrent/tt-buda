# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from inspect import signature
import pytest
import copy
import random
import torch
import math

import pybuda
from pybuda import (
    Tensor,
    CompilerConfig,
    DataFormat,
    PyTorchModule,
)
from pybuda.verify import verify_module, verify_module_pipeline, VerifyConfig, TestKind
from pybuda._C.backend_api import BackendDevice, BackendType
from test.common import compile, run, device, ModuleBuilder
from pybuda.config import _get_global_compiler_config, _clear_global_compiler_config
from pybuda.ttdevice import get_device_config
from test_galaxy_unit_tests import get_galaxy_chip_adjacency_list

from loguru import logger

random.seed(456)
RUNTIME_PARAMS_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "one_shelf_runtime_params.yaml"
)
ETH_CONNECTIONS_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "one_shelf_eth_connections.yaml"
)

BACKEND_CLUSTER_DESC = os.path.join(os.getenv("BUDA_HOME"), "cluster_desc.yaml")

############################################################################################
############################### INPUT SHAPES UNIT TESTS ####################################
############################################################################################


def test_galaxy_large_inputs():
    # This test crashes intel CPUs as of Feb 10
    def test_large_mb_bert_size_host_input(act):
        mmio_chip = 0
        pybuda.override_dram_queue_placement("input_0_unary0", chip_id=mmio_chip, channel=0)

        unary0 = pybuda.op.Exp("unary0", act)
        return unary0

    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0

    # Only run this on WH silicon, where create-ethernet-map can be called
    if devtype == BackendType.Golden:
        compiler_cfg.backend_runtime_params_path = RUNTIME_PARAMS_FILE

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
    chip_ids = list(galaxy_adjacent_chips.keys())
    chip_ids.sort()

    pybuda.pybuda_reset()
    compiler_cfg.enable_consteval = False
    pybuda.set_configuration_options(output_queues_on_host=False)

    module = ModuleBuilder(test_large_mb_bert_size_host_input)
    inputs_shape = [(256, 1, 1028, 384)]  # bert large input shape, 256 microbatch

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

############################################################################################
################################## I/O BANDWIDTH TESTS #####################################
############################################################################################
def get_bandwidth_test(input_chip, dram_channel, verify_cfg):

    @run(
          verify_cfg=verify_cfg,
          num_inputs=2,
    )
    def test_unary_op_input_on_host(act):
        pybuda.override_dram_queue_placement("inputs", chip_id=input_chip, channel=dram_channel)
        #pybuda.override_op_size("unary0", (1,1))

        unary0 = pybuda.op.Buffer("unary0", act)
        return unary0

    return test_unary_op_input_on_host

@pytest.fixture
def test_chips(request):
    return request.config.getoption("--test_chips", default="0")

@pytest.mark.parametrize("output_on_host", [True, False], ids=["output_on_host", "output_on_device"])
@pytest.mark.parametrize("input_shape", [(32, 1, 128, 128), (256, 1, 1028, 384)], ids=["shape1", "shape2"])
@pytest.mark.parametrize("input_df", [torch.float32, torch.float16, torch.bfloat16], ids=["Float32", "Float16", "BFloat16"])
def test_galaxy_bandwidth_sweep(test_chips, input_shape, output_on_host, input_df):
    def reset_pybuda_between_tests():
        pybuda.pybuda_reset()
        pybuda.set_configuration_options(
            backend_cluster_descriptor_path=BACKEND_CLUSTER_DESC,
            output_queues_on_host=output_on_host,
            enable_consteval=False,
        )
    mmio_chip = 0
    dram_channels = list(range(6))
    input_shapes = [input_shape]

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
    chip_ids = [int(chip) for chip in test_chips.split(",")]

    for input_chip in chip_ids:
        dram_channel = 1
        logger.info(f"Running {input_df} bandwidth test with input on chip {input_chip} dram chan {dram_channel}");
        reset_pybuda_between_tests()

        test = get_bandwidth_test(input_chip, dram_channel, VerifyConfig(
                 test_kind=TestKind.INFERENCE,
                 pcc=0.95,
                 devtype=devtype,
                 arch=arch,
                 run_net2pipe=True,
                 chip_ids=list(set([mmio_chip, input_chip])),
             ))
        x = Tensor.create_from_torch(torch.rand(*input_shapes[0], requires_grad=False, dtype=input_df))
        test(x)
