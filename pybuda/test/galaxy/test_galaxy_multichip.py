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

random.seed(123)
ONE_SHELF_RUNTIME_PARAMS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "one_shelf_runtime_params.yaml"
)
ONE_SHELF_ETH_CONNECTIONS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "one_shelf_eth_connections.yaml"
)

############################################################################################
################################## GALAXY BERT TESTS #######################################
############################################################################################


def get_relaxed_atol_pcc(test_kind, test_device, size="tiny", microbatch_size=1):
    """
    Figure out reasonable pcc/atol for training on silicon
    """
    training_atol = 0.3
    training_pcc = 0.95
    if test_device.is_silicon():
        training_pcc = 0.85
        if size != "tiny" or microbatch_size > 1:
            training_atol = 0.55
            training_pcc = 0.8
    inference_atol = 0.1
    inference_pcc = 0.95
    relative_atol = training_atol if test_kind.is_training() else inference_atol
    if test_device.is_silicon() and test_kind.is_training():
        relative_atol *= 3.5
    pcc = training_pcc if test_kind.is_training() else inference_pcc

    if size == "large":
        relative_atol = 0.15
    return relative_atol, pcc


@pytest.mark.parametrize("size", ["tiny", "base", "large"])
@pytest.mark.parametrize(
    "encoder_count", [1, 2, 4, 12, 24,], ids=["enc1", "enc2", "enc4", "enc12", "enc24"]
)
@pytest.mark.parametrize(
    "num_chips", [2, 4, 8, 12, 32], ids=["chip2", "chip4", "chip8", "chip12", "chip32"],
)
def test_pt_encoder(test_kind, test_device, size, encoder_count, num_chips):
    optimizer = {"type": "sgd", "params": {"learning_rate": 50.0}}
    if size == "tiny":
        model_name = "prajjwal1/bert-tiny"
        seq_len = 128
    elif size == "base":
        model_name = "bert-base-uncased"
        seq_len = 128
        if test_device.is_silicon() and test_kind.is_training():
            _get_global_compiler_config().enable_broadcast_splitting = (
                True  # fork error workaround
            )
            pybuda.config.override_op_size(
                "bw_in0_matmul_128_matmul_1", (1, 2)
            )  # tenstorrent/budabackend#667
            pytest.skip(
                "Issue 667"
            )  # unsure why, but CI fails even with the workaround above, while it passes in interactive runs
    elif size == "large":
        model_name = "bert-large-uncased"
        seq_len = 384
        if test_device.is_silicon() and test_kind.is_training():
            _get_global_compiler_config().enable_broadcast_splitting = (
                True  # fork error workaround
            )
    else:
        raise RuntimeError("Unknown size")

    if test_device.is_silicon() and test_kind.is_recompute():
        pytest.skip()  # intermittent hang on silicon

    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = encoder_count
    model = BertModel(config=config)
    encoder = PyTorchModule("bert_encoder", model.encoder)
    microbatch = 2

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device, size, microbatch)

    if test_device.is_silicon() and test_kind.is_training() and size == "base":
        if test_device.is_wormhole_b0():
            pcc = 0.9

    if test_device.is_silicon() and test_kind.is_training() and size == "large":
        # Revert when issue is closed: tenstorrent/pybuda#207
        import os

        os.environ["PYBUDA_NO_FUSE_MATMUL_BIAS"] = "1"
        os.environ["PYBUDA_ENABLE_BROADCAST_SPLITTING"] = "1"

    waive_gradient_errors = {"attention.self.key.bias"}

    compiler_cfg = _get_global_compiler_config()
    pybuda.set_configuration_options(
        backend_cluster_descriptor_path=eth_connections_file
    )

    verify_module(
        encoder,
        [(microbatch, seq_len, config.hidden_size), (microbatch, 1, seq_len, seq_len)],
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            relative_atol=relative_atol,
            pcc=pcc,
            optimizer=optimizer,
            accumulation_steps=1,
            microbatch_count=1,
            epochs=1,
            waive_gradient_errors=waive_gradient_errors,
            chip_ids=list(range(num_chips)),
        ),
        # Input gradient is really hard to match, so setting requires_grad to false.
        # Will need another way to say what's "correct"
        input_params=[{"requires_grad": False}, {"requires_grad": False}],
        uniform_inputs=True,
    )


@pytest.mark.parametrize(
    "cfg",
    [(128, 2, 128, "tiny"), (768, 12, 128, "base"), (1024, 16, 384, "large")],
    ids=["tiny", "base", "large"],
)
@pytest.mark.parametrize(
    "encoder_count", [1, 2, 4, 12, 24,], ids=["enc1", "enc2", "enc4", "enc12", "enc24"]
)
@pytest.mark.parametrize(
    "num_chips", [2, 4, 8, 12, 32], ids=["chip2", "chip4", "chip8", "chip12", "chip32"],
)
def test_multichip_wormhole_b0_multi_encoder_split_concurrent(
    test_kind, cfg, test_device, encoder_count, num_chips
):
    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]
    size = cfg[3]

    microbatch_size = 1

    config = {"num_heads": num_heads, "encoder_index": 0}

    compiler_cfg = _get_global_compiler_config()

    pybuda.set_configuration_options(
        backend_cluster_descriptor_path=ONE_SHELF_ETH_CONNECTIONS
    )
    relative_atol, pcc = get_relaxed_atol_pcc(
        test_kind, test_device, size, microbatch_size
    )

    modules = []
    for encoder_index in range(encoder_count):
        enc_params = get_bert_parameters(
            "encoder", hidden_dim=hidden_dim, encoder_index=encoder_index
        )
        config = copy.copy(config)
        config["encoder_index"] = encoder_index
        config["passthrough_attn_mask"] = bool(encoder_index != (encoder_count - 1))

        mod = PyBudaBertEncoder(f"encoder{encoder_index}", enc_params, config)

        enc_params[f"reciprocal_of_sqrt_of_head_size_{encoder_index}"].set_value(
            torch.full((1, 1, 1, 1), 1 / math.sqrt(num_heads))
        )
        if encoder_index > 0:
            compiler_cfg.place_on_new_epoch(f"mha_{encoder_index}_query")

        modules.append(mod)

    verify_module_pipeline(
        modules,
        [(microbatch_size, 1, seq_len, hidden_dim), (microbatch_size, 1, 1, seq_len)],
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=BackendDevice.Wormhole_B0,
            relative_atol=relative_atol,
            pcc=pcc,
            accumulation_steps=1,
            chip_ids=list(range(num_chips)),
            waive_gradient_errors={"ff.bert.encoder.layer.0.attention.self.key.bias"},
        ),
        input_params=[{}, {"requires_grad": False}],
        scale_params=100,
        inputs_centered_on_zero=True,
        params_centered_on_zero=True,
    )


############################################################################################
################################ FULL GALAXY OP TESTS ######################################
############################################################################################
@pytest.mark.parametrize(
    "num_chips", [2, 4, 8, 16, 32,], ids=["2", "4", "8", "16", "32"]
)
def test_galaxy_linked_unary_ops(test_kind, test_device, num_chips):
    op_names = list(f"op{i}" for i in range(num_chips))

    def linked_list_32_chips(act):
        unary_op_list = [
            pybuda.op.Gelu,
            pybuda.op.Log,
            pybuda.op.Buffer,
            pybuda.op.Exp,
            pybuda.op.Sqrt,
        ]
        op = pybuda.op.Gelu(op_names[0], act)
        for i in range(1, num_chips):
            pybuda_op = unary_op_list[i % len(unary_op_list)]
            op = pybuda_op(op_names[i], op)
        return op

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_consteval = False
    for op in op_names:
        pybuda.set_chip_break(op)

    if test_device.devtype == BackendType.Golden:
        pybuda.set_configuration_options(
            backend_cluster_descriptor_path=ONE_SHELF_ETH_CONNECTIONS,
            backend_runtime_params_path=ONE_SHELF_RUNTIME_PARAMS,
        )

    pybuda.set_configuration_options(accumulate_df=DataFormat.Float32)

    module = ModuleBuilder(linked_list_32_chips)
    verify_module(
        module,
        [(1, 1, 64, 64)],
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            relative_atol=0.22,
            run_net2pipe=True,
            chip_ids=list(range(num_chips)),
        ),
        inputs_centered_on_zero=True,
    )


@pytest.mark.parametrize(
    "num_chips", [2, 4, 8, 16, 32], ids=["2", "4", "8", "16", "32"]
)
def test_galaxy_linked_8_unaries_per_chip(test_kind, test_device, num_chips):
    MAX_STREAMS_PER_CHAN = 8
    op_names = list(f"op{i}" for i in range(num_chips * MAX_STREAMS_PER_CHAN))

    def linked_list_8_unaries_per_chip(act0, act1, act2, act3, act4, act5, act6, act7):
        unary_op_list = [
            pybuda.op.Gelu,
            pybuda.op.Log,
            pybuda.op.Buffer,
            pybuda.op.Exp,
            pybuda.op.Sqrt,
        ]
        op0 = pybuda.op.Buffer(op_names[0], act0)
        op1 = pybuda.op.Buffer(op_names[1], act1)
        op2 = pybuda.op.Buffer(op_names[2], act2)
        op3 = pybuda.op.Buffer(op_names[3], act3)
        op4 = pybuda.op.Buffer(op_names[4], act4)
        op5 = pybuda.op.Buffer(op_names[5], act5)
        op6 = pybuda.op.Buffer(op_names[6], act6)
        op7 = pybuda.op.Buffer(op_names[7], act7)
        for i in range(1, num_chips):
            pybuda_op = unary_op_list[i % len(unary_op_list)]
            op0 = pybuda_op(op_names[MAX_STREAMS_PER_CHAN * i], op0)
            op1 = pybuda_op(op_names[MAX_STREAMS_PER_CHAN * i + 1], op1)
            op2 = pybuda_op(op_names[MAX_STREAMS_PER_CHAN * i + 2], op2)
            op3 = pybuda_op(op_names[MAX_STREAMS_PER_CHAN * i + 3], op3)
            op4 = pybuda_op(op_names[MAX_STREAMS_PER_CHAN * i + 4], op4)
            op5 = pybuda_op(op_names[MAX_STREAMS_PER_CHAN * i + 5], op5)
            op6 = pybuda_op(op_names[MAX_STREAMS_PER_CHAN * i + 6], op6)
            op7 = pybuda_op(op_names[MAX_STREAMS_PER_CHAN * i + 7], op7)
        return op0, op1, op2, op3, op4, op5, op6, op7

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_consteval = False
    for i in range(MAX_STREAMS_PER_CHAN, len(op_names), MAX_STREAMS_PER_CHAN):
        pybuda.set_chip_break(op_names[i])

    pybuda.set_configuration_options(
        # backend_cluster_descriptor_path=eth_connections_file,
        accumulate_df=DataFormat.Float32
    )

    module = ModuleBuilder(linked_list_8_unaries_per_chip)
    verify_module(
        module,
        [(1, 1, 64, 64),
         (1, 1, 64, 64),
         (1, 1, 64, 64),
         (1, 1, 64, 64),
         (1, 1, 64, 64),
         (1, 1, 64, 64),
         (1, 1, 64, 64),
         (1, 1, 64, 64),
        ],
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            relative_atol=0.22,
            run_net2pipe=True,
            chip_ids=list(range(num_chips)),
        ),
        inputs_centered_on_zero=True,
    )


def test_galaxy_bert_large_simple_graph_test(test_kind, test_device):
    num_chips = 32

    # ops match dimensions of cross chip ops in bert large, for debugging
    # essentially, a larger 32 chip test
    def bert_ops(hidden, attention_q_weights, attention_q_bias, input_5):
        pybuda.override_op_size("matmul1", (2, 4))
        pybuda.override_op_size("op1_a", (2, 1))
        matmul1 = pybuda.op.Matmul(
            "matmul1", hidden, attention_q_weights, attention_q_bias
        )
        op1_a = pybuda.op.Gelu("op1_a", matmul1)
        op1_b = pybuda.op.Gelu("op1_b", matmul1)

        pybuda.set_chip_break("multiply2")
        pybuda.override_op_size("multiply2", (2, 1))
        pybuda.override_op_size("op2_a", (2, 1))
        multiply2 = pybuda.op.Multiply("multiply2", op1_a, op1_b)
        op2_a = pybuda.op.Gelu("op2_a", multiply2)
        op2_b = pybuda.op.Sqrt("op2_b", multiply2)

        pybuda.set_chip_break("multiply3")
        pybuda.override_op_size("multiply3", (2, 1))
        pybuda.override_op_size("op3_a", (2, 1))
        multiply3 = pybuda.op.Multiply("multiply3", op2_a, op2_b)
        op3_a = pybuda.op.Gelu("op3_a", multiply3)
        op3_b = pybuda.op.Sqrt("op3_b", multiply3)

        pybuda.set_chip_break("multiply4")
        pybuda.override_op_size("multiply4", (2, 1))
        pybuda.override_op_size("op4_a", (2, 4))
        multiply4 = pybuda.op.Multiply("multiply4", op3_a, op3_b)
        op4_a = pybuda.op.Gelu("op4_a", multiply4)

        pybuda.set_chip_break("multiply4")
        pybuda.override_op_size("matmul5", (2, 8))
        pybuda.override_op_size("op5_a", (2, 1))
        matmul5 = pybuda.op.Matmul("matmul5", op4_a, input_5)
        op5_a = pybuda.op.Gelu("op5_a", matmul1)
        op5_b = pybuda.op.Gelu("op5_b", matmul1)

        return matmul5

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_consteval = False

    pybuda.set_configuration_options(
        backend_cluster_descriptor_path=ONE_SHELF_ETH_CONNECTIONS
    )

    module = ModuleBuilder(bert_ops)
    verify_module(
        module,
        [
            (1, 1, 32 * 12, 32 * 32),
            (1, 1, 32 * 32, 32 * 32),
            (1, 1, 1, 32 * 32),
            (1, 1, 32 * 32, 32 * 32),
        ],
        VerifyConfig(
            test_kind=test_kind,
            devtype=test_device.devtype,
            arch=test_device.arch,
            relative_atol=0.22,
            run_net2pipe=True,
            chip_ids=list(range(num_chips)),
        ),
        inputs_centered_on_zero=True,
    )


def test_galaxy_scan_chip_pairs(scan_chip):
    def two_chip_simple_unary_to_unary(act):
        pybuda.set_chip_break("unary1")

        unary0 = pybuda.op.Exp("unary0", act)
        unary1 = pybuda.op.Gelu("unary1", unary0)
        return unary1

    def two_chip_eth_gather(act):
        pybuda.set_chip_break("unary1")
        pybuda.override_op_size("unary0", (2, 2))
        pybuda.override_op_size("unary1", (1, 1))

        unary0 = pybuda.op.Exp("unary0", act)
        unary1 = pybuda.op.Gelu("unary1", unary0)
        return unary1

    def two_chip_eth_multicast(act0, act1):
        pybuda.set_chip_break("unary1")
        pybuda.override_op_size("unary0", (2, 1))
        pybuda.override_op_size("unary1", (1, 2))
        pybuda.override_op_size("matmul0", (2, 2))

        unary0 = pybuda.op.Sqrt("unary0", act0)
        unary1 = pybuda.op.Gelu("unary1", act1)
        matmul0 = pybuda.op.Matmul("matmul0", unary0, unary1)
        return matmul0

    def two_chip_eth_gather_multicast(act0, act1):
        pybuda.set_chip_break("unary1")
        pybuda.override_op_size("unary0", (2, 1))
        pybuda.override_op_size("unary1", (2, 2))
        pybuda.override_op_size("matmul0", (2, 2))

        unary0 = pybuda.op.Exp("unary0", act0)
        unary1 = pybuda.op.Exp("unary1", act1)
        matmul0 = pybuda.op.Matmul("matmul0", unary0, unary1)
        return matmul0

    def two_chip_dram_buf_fork_c0_to_c0c1(act):
        pybuda.set_chip_break("unary1")

        unary0 = pybuda.op.Gelu("unary0", act)
        unary1 = pybuda.op.Exp("unary1", unary0)
        return pybuda.op.Add("add", unary1, act)

    def two_chip_l1_buf_fork_c0_to_c1c1_same_consumer(act):
        pybuda.set_chip_break("matmul0")

        unary0 = pybuda.op.Exp("unary0", act)
        matmul0 = pybuda.op.Matmul("matmul0", unary0, unary0)
        return matmul0

    def two_chip_binary_inputs_c1_tensix_c1_dram(act0, act1):
        pybuda.set_chip_break("unary0")

        nop0 = pybuda.op.Buffer("nop0", act0)
        unary0 = pybuda.op.Exp("unary0", nop0)
        add0 = pybuda.op.Add("add0", unary0, act1)
        return add0

    def two_chip_matmul_inputs_c0_tensix_c1_dram(act0, act1):
        pybuda.set_chip_break("multiply0")

        nop0 = pybuda.op.Buffer("nop0", act0)
        multiply0 = pybuda.op.Multiply("multiply0", nop0, act1)
        return multiply0

    def two_chip_binary_inputs_c0_tensix_c1_tensix(act0, act1, act2):
        pybuda.set_chip_break("nop0")

        add0 = pybuda.op.Add("add0", act0, act1)
        nop0 = pybuda.op.Buffer("nop0", act2)
        add1 = pybuda.op.Add("add1", add0, nop0)
        return add1

    def two_chip_matmul_inputs_c0_tensix_c0_tensix(act0, act1, act2):
        pybuda.set_chip_break("multiply0")

        add0 = pybuda.op.Add("add0", act0, act1)
        nop0 = pybuda.op.Buffer("nop0", act2)
        multiply0 = pybuda.op.Multiply("multiply0", add0, nop0)
        return multiply0

    def two_chip_multi_temporal_unary_to_unary(act):
        # TODO: placement doesn't work for two non-mmio chips
        pybuda.set_chip_break("unary1")
        pybuda.set_epoch_break("unary2")
        pybuda.set_chip_break("unary3")

        unary0 = pybuda.op.Sqrt("unary0", act)
        unary1 = pybuda.op.Gelu("unary1", unary0)
        unary2 = pybuda.op.Exp("unary2", unary1)
        unary3 = pybuda.op.Log("unary3", unary2)
        return unary3

    test_list = [
        two_chip_simple_unary_to_unary,
        two_chip_eth_gather,
        two_chip_eth_multicast,
        two_chip_eth_gather_multicast,
    ]
    test_list += [
        two_chip_dram_buf_fork_c0_to_c0c1,
        two_chip_l1_buf_fork_c0_to_c1c1_same_consumer,
    ]
    test_list += [
        two_chip_binary_inputs_c1_tensix_c1_dram,
        two_chip_matmul_inputs_c0_tensix_c1_dram,
        two_chip_binary_inputs_c0_tensix_c1_tensix,
        two_chip_matmul_inputs_c0_tensix_c0_tensix,
    ]

    devtype = BackendType.Silicon
    arch = BackendDevice.Wormhole_B0
    compiler_cfg = _get_global_compiler_config()
    # pybuda.set_configuration_options(
    #        backend_cluster_descriptor_path=eth_connections_file
    # )

    # Only run this on WH_B0 silicon, where create-ethernet-map can be called
    device_cfg = get_device_config(
        arch,
        [], # chip_ids
        compiler_cfg.backend_cluster_descriptor_path,
        compiler_cfg.backend_runtime_params_path,
        compiler_cfg.store_backend_db_to_yaml,
        devtype,
    )
    eth_connections = (
        device_cfg.get_ethernet_connections()
    )  # {chip_a: {chan_a: (chip_b, chan_b) ... }... }
    compiler_cfg.enable_consteval = False
    pybuda.set_configuration_options(output_queues_on_host=False)

    galaxy_adjacent_chips = {}
    for chip_a, channels_a in eth_connections.items():
        if chip_a not in galaxy_adjacent_chips.keys():
            galaxy_adjacent_chips[chip_a] = set()
        for chip_b_chan_pair in channels_a.values():
            galaxy_adjacent_chips[chip_a].add(chip_b_chan_pair[0])

    if scan_chip == "full":
        chips_to_scan = list(galaxy_adjacent_chips.keys())
    else:
        chips_to_scan = [int(chip) for chip in scan_chip.split(",")]
        assert all(chip in galaxy_adjacent_chips.keys() for chip in chips_to_scan)

    logger.info(f"Running tests on chips {chips_to_scan}")
    for chip_a in chips_to_scan:
        for chip_b in galaxy_adjacent_chips[chip_a]:
            for test in test_list:
                chip_ids = [chip_a, chip_b]
                # add the mmio chip if it is not one of the used chip pairs
                if 0 not in chip_ids:
                    chip_ids.append(0)
                chip_ids.sort()

                pybuda.pybuda_reset()
                compiler_cfg.enable_consteval = False
                pybuda.set_configuration_options(output_queues_on_host=False)

                module = ModuleBuilder(test)
                num_inputs = len(signature(test).parameters)
                inputs_shape = [(1, 1, 64, 64)] * num_inputs
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
