# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from inspect import signature
import pytest
import copy
from collections import Counter
import random
import math
import torch
from transformers import BertForQuestionAnswering


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

ONE_SHELF_RUNTIME_PARAMS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "one_shelf_runtime_params.yaml"
)
ONE_SHELF_ETH_CONNECTIONS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "one_shelf_eth_connections.yaml"
)
ONE_SHELF_GALAXY_IDS = [0,30,31,28,27,26,6,5,29,3,4,7,8,17,2,18,16,9,10,15,19,20,14,11,12,13,21,22,23,24,25,1]

TWO_SHELF_RUNTIME_PARAMS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "two_shelf_runtime_params.yaml"
)
TWO_SHELF_ETH_CONNECTIONS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "two_shelf_eth_connections.yaml"
)
TWO_SHELF_GALAXY_IDS = [0, 15, 16, 13, 12, 11, 10, 17, 55, 56, 57, 54, 18, 19, 53, 58, 59, 52, 20, 21, 51, 60, 61, 50, 30, 31, 49, 62, 63, 48, 32, 33, 28, 29, 22, 23, 8, 9, 6, 5, 14, 3, 4, 7, 24, 39, 2, 40, 38, 25, 26, 37, 41, 42, 36, 27, 34, 35, 43, 44, 45, 46, 47, 1]

class BertEncoderLMHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.bert.encoder
        self.lm_head = model.qa_outputs

    def forward(self, hidden_state, attention_mask):
        return self.lm_head(
            self.encoder(hidden_state, attention_mask).last_hidden_state
        )

def set_bert_demo_env_settings():
    os.environ['PYBUDA_EXP_APPROX'] = '1'
    os.environ['PYBUDA_FUSE_OPS'] = '1'
    #os.environ['PYBUDA_NO_FUSE_MATMUL_BIAS'] = '1'
    #os.environ['PYBUDA_NO_FUSE_MATMUL_GELU'] = '1'
    os.environ['PYBUDA_NLP_MANUAL_TARGET'] = '185000'
    os.environ['PYBUDA_DISABLE_DRAM0'] = '1'
    os.environ['PYBUDA_EXTRA_L1_MARGIN'] = '131072'
    os.environ['PYBUDA_DISABLE_FORK_JOIN_NOPS'] = '1'
    os.environ['ENABLE_ETH_SERIALIZATON'] = '1'



####################### BERT DEMO ON ONE SHELF ##########################

def test_one_shelf_bert_large_demo(test_device):
    num_enc = 24
    def apply_galaxy_am_buffering(config):
        am_consumer_ops = ["attention_mask"]
        for i in range(num_enc):
            am_consumer_ops.append(f"add_{17+53*i}")

        config.insert_nop("attention_mask", am_consumer_ops)

    def apply_config_overrides(config):
        for i in range(num_enc):
            config.set_chip_break(f"matmul_{55+53*i}")

        pybuda.config.set_configuration_options(
            default_df_override=pybuda.DataFormat.Float16_b
        )
        pybuda.set_configuration_options(
            math_fidelity=pybuda.MathFidelity.HiFi3,
            backend_opt_level=3,
            enable_auto_transposing_placement=True,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            enable_consteval = False,
        )

    compiler_cfg = _get_global_compiler_config()
    devtype = test_device.devtype
    arch = test_device.arch

    pybuda.pybuda_reset()
    set_bert_demo_env_settings()
    apply_config_overrides(pybuda.config)
    apply_galaxy_am_buffering(pybuda.config)

    if devtype == BackendType.Golden:
        pybuda.set_configuration_options(
            backend_runtime_params_path = ONE_SHELF_RUNTIME_PARAMS,
            backend_cluster_descriptor_path = ONE_SHELF_ETH_CONNECTIONS
        )

    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(dtype=torch.float32)
    model.eval()
    model = BertEncoderLMHeadWrapper(model)
    module = pybuda.PyTorchModule("bert_squad", model)

    microbatch = 1

    verify_module(
        module,
        [(microbatch, 384, 1024), (microbatch, 1, 384, 384)],
        VerifyConfig(
            test_kind=TestKind.INFERENCE,
            pcc=0.95,
            devtype=devtype,
            arch=arch,
            run_net2pipe=True,
            chip_ids=ONE_SHELF_GALAXY_IDS,
        ),
        inputs_centered_on_zero=True,
    )

####################### BERT DEMO ON TWO SHELVES #######################

def test_two_shelf_bert_large_demo(test_device):
    num_enc = 24
    def apply_galaxy_am_buffering(config):
        am_consumer_ops = ["attention_mask"]
        for i in range(num_enc):
            am_consumer_ops.append(f"add_{17+53*i}")

        config.insert_nop("attention_mask", am_consumer_ops)
        curr_buffer_name = am_consumer_ops[0]
        for i in range(num_enc):
            config.insert_nop(curr_buffer_name, am_consumer_ops[i+1:], hoist_tms=False)
            curr_buffer_name = f"buffer_0_{curr_buffer_name}_{am_consumer_ops[i+1]}"

    def apply_config_overrides(config):
        for i in range(num_enc):
            config.override_op_size(f"matmul_{2+53*i}", [3, 1])
            config.override_op_size(f"matmul_{8+53*i}", [3, 1])
            config.override_op_size(f"matmul_{14+53*i}", [1, 2])
            config.override_op_size(f"matmul_{22+53*i}", [1, 4])
            config.override_op_size(f"matmul_{29+53*i}", [2, 1])
            config.override_op_size(f"matmul_{33+53*i}", [2, 1])
            config.override_op_size(f"matmul_{41+53*i}", [2, 4])
            config.override_op_size(f"matmul_{47+53*i}", [2, 4])
            # matmuls
            config.set_chip_break(f"matmul_{41+53*i}")
            config.set_chip_break(f"matmul_{55+53*i}")

        pybuda.config.set_configuration_options(
            default_df_override=pybuda.DataFormat.Float16_b
        )
        pybuda.set_configuration_options(
            math_fidelity=pybuda.MathFidelity.HiFi3,
            backend_opt_level=3,
            enable_auto_transposing_placement=True,
            accumulate_df=pybuda._C.DataFormat.Float16_b,
            enable_consteval = False,
        )

    compiler_cfg = _get_global_compiler_config()
    devtype = test_device.devtype
    arch = test_device.arch

    pybuda.pybuda_reset()
    set_bert_demo_env_settings()
    apply_config_overrides(pybuda.config)
    apply_galaxy_am_buffering(pybuda.config)

    if devtype == BackendType.Golden:
        pybuda.set_configuration_options(
            backend_runtime_params_path = TWO_SHELF_RUNTIME_PARAMS,
            backend_cluster_descriptor_path = TWO_SHELF_ETH_CONNECTIONS
        )

    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(dtype=torch.float32)
    model.eval()
    model = BertEncoderLMHeadWrapper(model)
    module = pybuda.PyTorchModule("bert_squad", model)

    microbatch = 1

    verify_module(
        module,
        [(microbatch, 384, 1024), (microbatch, 1, 384, 384)],
        VerifyConfig(
            test_kind=TestKind.INFERENCE,
            pcc=0.95,
            devtype=devtype,
            arch=arch,
            run_net2pipe=True,
            chip_ids=TWO_SHELF_GALAXY_IDS,
        ),
        inputs_centered_on_zero=True,
    )
