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


def test_galaxy_two_chip_module(microbatch):
    compiler_cfg = _get_global_compiler_config()
    devtype = BackendType.Golden
    arch = BackendDevice.Wormhole_B0
    chip_a = 31

    def two_chip_simple_unary_to_unary(act):
        pybuda.set_chip_break("unary1_A")
        pybuda.override_dram_queue_placement("inputs", chip_id=chip_a, channel=0)

        unary0 = pybuda.op.Exp("unary0_A", act)
        unary1 = pybuda.op.Gelu("unary1_A", unary0)
        return unary1

    pybuda.set_configuration_options(
        backend_cluster_descriptor_path=ONE_SHELF_ETH_CONNECTIONS,
        backend_runtime_params_path=ONE_SHELF_RUNTIME_PARAMS,
    )
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
    for chip_b in galaxy_adjacent_chips[chip_a]:
        test = two_chip_simple_unary_to_unary
        chip_ids = get_chip_ids_for_galaxy([chip_a, chip_b])
        pybuda.pybuda_reset()
        pybuda.set_configuration_options(
            backend_cluster_descriptor_path=ONE_SHELF_ETH_CONNECTIONS,
            backend_runtime_params_path=ONE_SHELF_RUNTIME_PARAMS,
            output_queues_on_host=False,
            enable_consteval=False,
        )

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
