# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Bert tests on backend
"""

import math
import copy

import torch
import pytest
import inspect

from pybuda.verify import verify_module, verify_module_pipeline, VerifyConfig

from test.bert.modules import (
    PyBudaBertMHA,
    PyBudaBertEncoder,
    PyBudaFeedForward,
    PyBudaPredictionHeadDecoder,
    PyBudaPredictionHeadTransform,
    get_bert_parameters
)

from pybuda import Tensor, DataFormat, BackendType, BackendDevice
from pybuda.config import _get_global_compiler_config
from pybuda._C.placer import DRAMPlacementAlgorithm

def get_relaxed_atol_pcc(test_kind, test_device, size = "tiny", microbatch_size = 1):
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

    return relative_atol, pcc



# Big config runs out of memory - to be debugged
#@pytest.mark.parametrize("cfg", [(128, 4, 128), (384, 6, 384), (768, 12, 512), (4096, 16, 2048)])
@pytest.mark.parametrize("cfg", [(128, 4, 128), (384, 6, 384), (768, 12, 512), (1024, 16, 512), (12288, 96, 2048), (20480, 128, 2048)])
def test_mha(test_kind, cfg, test_device):
    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)

    if test_device.devtype == BackendType.Silicon:
        if hidden_dim > 128 or test_kind.is_training():
            pytest.skip() # failing on silicon, need to be debugged
    if hidden_dim > 768:
        pytest.skip() # Large MHA fail FE

    microbatch_size = 1

    params = get_bert_parameters("mha", hidden_dim=hidden_dim)
    config =  { "num_heads": num_heads, "encoder_index": 0 }
    mod = PyBudaBertMHA("mha", params, config)

    verify_module(mod, [(microbatch_size, seq_len, hidden_dim), (microbatch_size, 1, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1,
                relative_atol=relative_atol, pcc=pcc, 
                waive_gradient_errors={"mha.bert.encoder.layer.0.attention.self.key.bias"}),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
    )

# Big config runs out of memory - to be debugged
#@pytest.mark.parametrize("cfg", [(128, 128), (384, 384), (768, 512), (4096, 2048)])
@pytest.mark.parametrize("cfg", [(128, 128), (384, 384), (768, 512)])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_ff(test_kind, cfg, test_device, optimizer):
    hidden_dim = cfg[0]
    seq_len = cfg[1]

    if optimizer == "adam":
        pytest.skip()

    if test_device.devtype == BackendType.Silicon:
        if hidden_dim > 384 and test_kind.is_training():
            pytest.skip() # pipegen forks

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device)

    microbatch_size = 1

    params = get_bert_parameters("ff", hidden_dim=hidden_dim)
    config =  { "encoder_index": 0 }
    mod = PyBudaFeedForward("ff", params, config)

    verify_module(mod, [(microbatch_size, seq_len, hidden_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1,
                relative_atol=relative_atol, pcc=pcc,
                optimizer = {"type": optimizer, "params": {"learning_rate": 50.0 } },
            ),
            uniform_inputs=True,
            params_centered_on_zero=True,
    )

@pytest.mark.parametrize("cfg", [(128, 128)])
def test_ff_fp16(test_kind, cfg, test_device):
    hidden_dim = cfg[0]
    seq_len = cfg[1]

    microbatch_size = 1

    if test_device.devtype == BackendType.Silicon:
        if test_kind.is_training():
            pytest.skip() # failing on silicon, need to be debugged

    params = get_bert_parameters("ff", hidden_dim=hidden_dim)
    for v in params.values():
        v.set_data_format(DataFormat.Float16)
    config =  { "encoder_index": 0 }
    mod = PyBudaFeedForward("ff", params, config)

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    verify_module(mod, [(microbatch_size, seq_len, hidden_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1,
                relative_atol=relative_atol, golden_ignore_df_precision=False, scale_loss=1.0, # Use actual formats since we're running FP16
                pcc=0.95),
            input_params=[{"data_format": torch.float16}],
            scale_params=10,
    )


# Big config runs out of memory - to be debugged
#@pytest.mark.parametrize("cfg", [(128, 128), (384, 384), (768, 512), (4096, 2048)])
@pytest.mark.parametrize("cfg", [(128, 128), (384, 384), (768, 512)])
def test_pred_transform(test_kind, cfg, test_device):
    hidden_dim = cfg[0]
    seq_len = cfg[1]

    microbatch_size = 1


    if test_device.devtype == BackendType.Silicon:
        if hidden_dim > 128 and test_kind.is_recompute():
            pytest.skip() # failing on silicon, need to be tweaked in precision, or debugged

    relative_atol = 0.3 if test_kind.is_training() or test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.93 if test_device.devtype == BackendType.Silicon and hidden_dim > 128 else 0.99

    params = get_bert_parameters("pred_transform", hidden_dim=hidden_dim)
    config =  { }
    mod = PyBudaPredictionHeadTransform("pred_transform", params, config)

    verify_module(mod, [(microbatch_size, seq_len, hidden_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1, relative_atol=relative_atol, pcc=pcc),
            scale_params=100,
    )

# Big config runs out of memory - to be debugged
#@pytest.mark.parametrize("cfg", [(128, 128), (384, 384), (768, 512), (4096, 2048)])
@pytest.mark.parametrize("cfg", [(128, 128, 30522), (384, 384, 30522), (768, 512, 30522)])
@pytest.mark.skip(reason="Working on it")
def test_pred_decoder(test_kind, cfg, test_device):
    hidden_dim = cfg[0]
    seq_len = cfg[1]
    vocab_size = cfg[2]

    microbatch_size = 1

    if test_device.devtype == BackendType.Silicon:
        if hidden_dim > 128:
            pytest.skip() # failing on silicon, need to be tweaked in precision, or debugged

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    pcc = 0.93 if test_device.devtype == BackendType.Silicon and hidden_dim > 128 else 0.99

    params = get_bert_parameters("pred_decoder", hidden_dim=hidden_dim, vocab_size=vocab_size)
    config =  { }
    mod = PyBudaPredictionHeadDecoder("pred_decoder", params, config)

    verify_module(mod, [(microbatch_size, seq_len, hidden_dim)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1, relative_atol=relative_atol, pcc=pcc),
            scale_params=100,
    )

# Big config runs out of memory - to be debugged
#@pytest.mark.parametrize("cfg", [(128, 4, 128), (384, 6, 384), (768, 12, 512), (4096, 16, 2048)])
@pytest.mark.parametrize("cfg", [(128, 4, 128), (384, 6, 128), (768, 12, 512)])
@pytest.mark.parametrize("dram_allocator", [DRAMPlacementAlgorithm.ROUND_ROBIN, DRAMPlacementAlgorithm.GREATEST_CAPACITY, DRAMPlacementAlgorithm.CLOSEST])
def test_encoder(test_kind, cfg, test_device, dram_allocator):
    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    if hidden_dim > 384 and test_kind.is_training():
        pytest.skip() # still working on this

    if test_device.devtype == BackendType.Silicon:
        if hidden_dim > 128:
            pytest.skip() # failing on silicon, need to be tweaked in precision, or debugged

    relative_atol = 0.3 if test_kind.is_training() else 0.1
    if test_device.devtype == BackendType.Silicon:
        relative_atol = 0.3
        pcc = 0.93 if hidden_dim > 128 else 0.99
    else:
        pcc = 0.95 if hidden_dim > 128 else 0.99

    microbatch_size = 1

    params = get_bert_parameters("encoder", hidden_dim=hidden_dim)
    config =  { "num_heads": num_heads, "encoder_index": 0 }
    mod = PyBudaBertEncoder("encoder", params, config)

    params["reciprocal_of_sqrt_of_head_size_0"].set_value(torch.full((1, 1, 1, 1), 1/math.sqrt(num_heads)))

    pybuda.config.set_configuration_options(dram_placement_algorithm=dram_allocator)

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    verify_module(mod, [(microbatch_size, seq_len, hidden_dim), (microbatch_size, 1, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1,
                waive_gradient_errors={"ff.bert.encoder.layer.0.attention.self.key.bias"},
                relative_atol=relative_atol, pcc=pcc),
            input_params=[{}, {"requires_grad": False}],
            scale_params=200,
            params_centered_on_zero=True,
    )

# Big config runs out of memory - to be debugged
#@pytest.mark.parametrize("cfg", [(128, 4, 128), (384, 6, 384), (768, 12, 512), (4096, 16, 2048)])
@pytest.mark.parametrize("cfg", [(128, 4, 128), (384, 6, 384), (768, 12, 512)])
@pytest.mark.parametrize("encoder_count", [2, 4], ids=["enc2", "enc4"])
#@pytest.mark.skip(reason="Still developing")
def test_multi_encoder(test_kind, cfg, test_device, encoder_count):
    if test_kind.is_training():
        pytest.skip() # not working

    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    microbatch_size = 1

    config =  { "num_heads": num_heads, "encoder_index": 0 }

    if test_device.devtype == BackendType.Silicon:
        if hidden_dim > 128:
            pytest.skip() # failing on silicon, need to be tweaked in precision, or debugged

    modules = []
    for encoder_index in range(encoder_count):
        enc_params = get_bert_parameters("encoder", hidden_dim=hidden_dim, encoder_index=encoder_index)
        config = copy.copy(config)
        config["encoder_index"] = encoder_index
        config["passthrough_attn_mask"] = bool(encoder_index != (encoder_count - 1))

        mod = PyBudaBertEncoder(f"encoder{encoder_index}", enc_params, config)

        enc_params[f"reciprocal_of_sqrt_of_head_size_{encoder_index}"].set_value(torch.full((1, 1, 1, 1), 1/math.sqrt(num_heads)))

        modules.append(mod)

    verify_module_pipeline(modules, [(microbatch_size, 1, seq_len, hidden_dim), (microbatch_size, 1, 1, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1,
                waive_gradient_errors={"ff.bert.encoder.layer.0.attention.self.key.bias"}),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
    )


@pytest.mark.parametrize("cfg", [(128, 4, 128), (384, 6, 384), (768, 12, 512)])
@pytest.mark.parametrize("encoder_count", [2, 12], ids=["enc2", "enc12"])
def test_multichip_wormhole_multi_encoder(test_kind, cfg, test_device, encoder_count):
    if test_kind.is_training():
        pytest.skip() # still working on it.

    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    microbatch_size = 1

    config =  { "num_heads": num_heads, "encoder_index": 0 }

    modules = []
    for encoder_index in range(encoder_count):
        enc_params = get_bert_parameters("encoder", hidden_dim=hidden_dim, encoder_index=encoder_index)
        config = copy.copy(config)
        config["encoder_index"] = encoder_index
        config["passthrough_attn_mask"] = bool(encoder_index != (encoder_count - 1))

        mod = PyBudaBertEncoder(f"encoder{encoder_index}", enc_params, config)

        enc_params[f"reciprocal_of_sqrt_of_head_size_{encoder_index}"].set_value(torch.full((1, 1, 1, 1), 1/math.sqrt(num_heads)))

        modules.append(mod)

    compiler_cfg = _get_global_compiler_config()
    # tenstorrent/pybuda#480
    compiler_cfg.use_interactive_placer = False if test_device.arch is BackendDevice.Grayskull else True

    verify_module_pipeline(modules, [(microbatch_size, 1, seq_len, hidden_dim), (microbatch_size, 1, 1, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1, chip_ids=[0, 1],
                waive_gradient_errors={"ff.bert.encoder.layer.0.attention.self.key.bias"}),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
    )


@pytest.mark.skip(reason="working on this")
@pytest.mark.parametrize("cfg", [(128, 4, 128), (384, 6, 384), (768, 12, 512)])
@pytest.mark.parametrize("encoder_count", [2,],  ids=["enc2",])
@pytest.mark.parametrize("num_chips", [2,], ids=["chip2",])
def test_multichip_wormhole_split(test_kind, cfg, test_device, encoder_count, num_chips):
    if test_kind.is_training():
        pytest.skip() # still working on it.
    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    microbatch_size = 1

    config =  { "num_heads": num_heads, "encoder_index": 0 }

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.num_chips = num_chips

    modules = []
    for encoder_index in range(encoder_count):
        enc_params = get_bert_parameters("encoder", hidden_dim=hidden_dim, encoder_index=encoder_index)
        config = copy.copy(config)
        config["encoder_index"] = encoder_index
        config["passthrough_attn_mask"] = bool(encoder_index != (encoder_count - 1))

        mod = PyBudaBertEncoder(f"encoder{encoder_index}", enc_params, config)

        enc_params[f"reciprocal_of_sqrt_of_head_size_{encoder_index}"].set_value(torch.full((1, 1, 1, 1), 1/math.sqrt(num_heads)))

        if encoder_index > 0:
            compiler_cfg.place_on_new_chip(f"mha_{encoder_index}_query")

        modules.append(mod)

    verify_module_pipeline(modules, [(microbatch_size, 1, seq_len, hidden_dim), (microbatch_size, 1, 1, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1,
                waive_gradient_errors={"ff.bert.encoder.layer.0.attention.self.key.bias"}),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
    )

@pytest.mark.parametrize("cfg", [(128, 2, 128), (768, 12, 128), (1024, 16, 384)], ids=["tiny", "base", "large"])
@pytest.mark.parametrize("encoder_count", [1, 2, 4, 12, 24,], ids=["enc1", "enc2", "enc4", "enc12", "enc24"])
@pytest.mark.parametrize("num_chips", [2, 4, 8, 12,], ids=["chip2", "chip4", "chip8", "chip12"])
def test_multichip_wormhole_multi_encoder_split_concurrent(test_kind, cfg, test_device, encoder_count, num_chips):
    # Set pybuda config
    pybuda.config.set_configuration_options(default_df_override=DataFormat.Float16_b)
    
    # Skip all golden tests
    if not test_device.is_silicon():
        pytest.skip()

    if test_kind.is_training():
        pytest.skip() # still working on it.

    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    microbatch_size = 1

    config =  { "num_heads": num_heads, "encoder_index": 0 }

    compiler_cfg = _get_global_compiler_config()

    # Disable t-streaming in this particular test case
    compiler_cfg.enable_t_streaming = False

    modules = []
    for encoder_index in range(encoder_count):
        enc_params = get_bert_parameters("encoder", hidden_dim=hidden_dim, encoder_index=encoder_index)
        config = copy.copy(config)
        config["encoder_index"] = encoder_index
        config["passthrough_attn_mask"] = bool(encoder_index != (encoder_count - 1))

        mod = PyBudaBertEncoder(f"encoder{encoder_index}", enc_params, config)

        enc_params[f"reciprocal_of_sqrt_of_head_size_{encoder_index}"].set_value(torch.full((1, 1, 1, 1), 1/math.sqrt(num_heads)))
        if encoder_index > 0:
            compiler_cfg.place_on_new_epoch(f"mha_{encoder_index}_query")

        modules.append(mod)

    verify_module_pipeline(modules, [(microbatch_size, 1, seq_len, hidden_dim), (microbatch_size, 1, 1, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1,
                chip_ids=list(range(num_chips)),
                waive_gradient_errors={"ff.bert.encoder.layer.0.attention.self.key.bias"}),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
            inputs_centered_on_zero=True,
            params_centered_on_zero=True,
    )

from pybuda import PyTorchModule
from transformers import BertModel, BertConfig, BertForPreTraining, BertTokenizer, BertForQuestionAnswering
def test_pt_bert(test_kind, test_device):
    seq_len = 128
    microbatch_size = 1

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    input_sentence = "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives: Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence."
    input_tokens = tokenizer.encode(input_sentence, max_length=128, pad_to_max_length=True)

    inputs = [(Tensor.create_from_torch(torch.Tensor(input_tokens).int().unsqueeze(0)),) ,]
    model = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=False, add_pooling_layer=False)

    embeddings = PyTorchModule("bert_embeddings", model.embeddings)
    encoder = PyTorchModule("bert_encoder", model.encoder)

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    verify_module_pipeline([embeddings, encoder],
            [(microbatch_size, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, accumulation_steps=1, relative_atol=relative_atol,
                waive_gradient_errors={"layer.0.attention.self.key.bias", "layer.1.attention.self.key.bias"}),
            input_params=[{"requires_grad": False, "data_format": torch.int}],
            device_types=["CPUDevice", "TTDevice"], inputs=inputs
    )



'''
@lpanos: Concurrent mode does NOT work on the verify_module flow.

This is because we end up pushing the cpueval_forward and cpueval_backwards commands
concurrently to generate ground truth. This comes with two main issues.
    1. Pytorch Autograd cannot propogate gradients between processes.
       This leaves the "parameters" we are trying to track without grads
    2. For larger models, (i.e. bert) upon popping the response queue for backwards outputs we get
       "Trying to resize storage that is not resizeable" (issue #1029). This indicates that the tensor
       data being popped from the command response queue is not the same size as what pytorch is expecting.
       I was not able to figure out why this happens.
    
It seems as though pybuda/pybuda/verify/backend.py::_verify_training was intended to be run in sequential mode. 
The solution to this problem probably involves generating the ground truth sequentially and comparing afterwards. 
'''

@pytest.mark.parametrize("size", ["tiny", "base", "large"])
@pytest.mark.parametrize("encoder_count", [1, 2, 4, 12, 24], ids=["enc1", "enc2", "enc4", "enc12", "enc24"])
@pytest.mark.parametrize("num_chips", [1, 2, 4, 8, 12, 32], ids=["chip1", "chip2", "chip4", "chip8", "chip12", "chip32"])
def test_pt_encoder(test_kind, test_device, size, encoder_count, num_chips):
    # Set pybuda config
    pybuda.config.set_configuration_options(default_df_override=DataFormat.Float16_b)

    # Skip certain tests in golden CI (redundant)
    if not test_device.is_silicon() and (num_chips > 1 or encoder_count > 2):
        pytest.skip()

    if test_kind.is_training() and test_device.arch == BackendDevice.Grayskull and size == "large":
        pytest.skip()  # see tenstorrent/pybuda#969

    optimizer = {"type": "sgd", "params": {"learning_rate": 50.0 } }
    if size == "tiny":
        model_name = "prajjwal1/bert-tiny"
        seq_len = 128
    elif size == "base":
        model_name = "bert-base-uncased"
        seq_len = 128
        if test_device.is_silicon() and test_kind.is_training():
            _get_global_compiler_config().enable_broadcast_splitting = True # fork error workaround
            pybuda.config.override_op_size("bw_in0_matmul_128_matmul_1", (1, 2)) # tenstorrent/budabackend#667
            #pytest.skip("Issue 667") # unsure why, but CI fails even with the workaround above, while it passes in interactive runs
    elif size == "large":
        model_name = "bert-large-uncased"
        seq_len = 384
        if test_device.is_silicon() and test_kind.is_training():
            _get_global_compiler_config().enable_broadcast_splitting = True # fork error workaround

    else:
        raise RuntimeError("Unknown size")

    if test_device.is_silicon() and test_kind.is_recompute():
        pytest.skip() # intermittent hang on silicon

    # Disable t-streaming in this particular test case
    _get_global_compiler_config().enable_t_streaming = False
    
    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = encoder_count # no need to run a lot
    model = BertModel(config=config)
    encoder = PyTorchModule("bert_encoder", model.encoder)
    microbatch = 2

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device, size, microbatch)

    if test_device.is_silicon() and test_kind.is_training() and size == "base":
        # pcc quite bad for bert-base... to be debugged
        # set pcc values to highest seen on each type of architecture
        if test_device.is_grayskull():
            pcc = 0.68
        elif test_device.is_wormhole_b0():
            pcc = 0.9

    import os
    if test_device.is_silicon() and test_kind.is_training() and size == "large":
        # Revert when issue is closed: tenstorrent/pybuda#207
        os.environ["PYBUDA_NO_FUSE_MATMUL_BIAS"] = "1"
        os.environ["PYBUDA_ENABLE_BROADCAST_SPLITTING"] = "1"

    if test_kind.is_training() and size == "large":
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{77*1024}"
        os.environ["PYBUDA_DISABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"

    waive_gradient_errors = {"attention.self.key.bias"}
    verify_module(encoder, [(microbatch, seq_len, config.hidden_size), (microbatch, 1, seq_len, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc,
                optimizer=optimizer,
                accumulation_steps=1,
                microbatch_count=1,
                epochs=1,
                num_chips=num_chips,
                waive_gradient_errors=waive_gradient_errors),
            # Input gradient is really hard to match, so setting requires_grad to false.
            # Will need another way to say what's "correct"
            input_params=[{"requires_grad": False}, {"requires_grad": False}],
            uniform_inputs=True,
    )

@pytest.mark.parametrize("size", ["tiny", "base", "large"])
@pytest.mark.parametrize("encoder_count", [1, 2, 4, 12, 24], ids=["enc1", "enc2", "enc4", "enc12", "enc24"])
@pytest.mark.parametrize("num_chips", [1, 2, 4, 8, 12,], ids=["chip1", "chip2", "chip4", "chip8", "chip12"])
def test_pt_encoder_ethernet_datacopy_serialization(test_kind, test_device, size, encoder_count, num_chips):
    pytest.skip() # ethernet datacopy support needs to be picked up in BBE first
    import os
    os.environ["PYBUDA_ENABLE_ETH_DATACOPY_SERIALIZATION"] = "1"
    os.environ["PYBUDA_DISABLE_INTERACTIVE_PLACER"] = "1"

    _compiler_config = _get_global_compiler_config()
    _compiler_config.enable_t_streaming = False
    test_pt_encoder(test_kind, test_device, size, encoder_count, num_chips)

@pytest.mark.parametrize("size", ["base"])
def test_pt_encoder_gs_2x_multichip(test_kind, test_device, size):

    if size == "tiny":
        model_name = "prajjwal1/bert-tiny"
        seq_len = 128
    elif size == "base":
        model_name = "bert-base-uncased"
        seq_len = 128
        if test_device.is_silicon() and test_kind.is_training():
            pytest.skip() # fork error
    elif size == "large":
        model_name = "bert-large-uncased"
        seq_len = 384
        if test_device.is_silicon() and test_kind.is_training():
            pytest.skip() # fork error
    else:
        raise RuntimeError("Unknown size")

    if not test_kind.is_training():
        pytest.skip() 

    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2 # no need to run a lot
    model = BertModel(config=config)
    encoder = PyTorchModule("bert_encoder", model.encoder)

    relative_atol, pcc = get_relaxed_atol_pcc(test_kind, test_device, size)

    waive_gradient_errors = {"attention.self.key.bias"}
    microbatch = 1
    verify_module(encoder, [(microbatch, seq_len, config.hidden_size), (microbatch, 1, seq_len, seq_len)],
            VerifyConfig(test_kind=test_kind, devtype=test_device.devtype, arch=test_device.arch, relative_atol=relative_atol, pcc=pcc,
                accumulation_steps=1,
                microbatch_count=1,
                epochs=1,
                waive_gradient_errors=waive_gradient_errors,
                chip_ids=list(range(2))
            ),
            # Input gradient is really hard to match, so setting requires_grad to false.
            # Will need another way to say what's "correct"
            input_params=[{"requires_grad": False}, {"requires_grad": False}],
            uniform_inputs=True,
    )

def test_pt_pooler(test_kind, test_device):
    config = BertConfig.from_pretrained("prajjwal1/bert-tiny", torchscript=True)
    bert = BertForPreTraining(config)
    pretrain_heads = PyTorchModule("pooler", bert.bert.pooler)

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    verify_module(pretrain_heads, [(1, 128, 128)],
            VerifyConfig(test_kind=test_kind,
                devtype=test_device.devtype,
                arch=test_device.arch,
                relative_atol=relative_atol)
    )


def test_pt_pretrain_heads(test_kind, test_device):
    pytest.skip() # TODO: Temp skip due the CI Pipeline issues.

    if test_kind.is_training():
        pytest.skip()

    config = BertConfig.from_pretrained("prajjwal1/bert-tiny", torchscript=True)
    bert = BertForPreTraining(config)
    pretrain_heads = PyTorchModule("ptheads", bert.cls)

    relative_atol = 0.3 if test_kind.is_training() and test_device.devtype == BackendType.Silicon else 0.1
    verify_module(pretrain_heads, [(1, 128, 128), (1, 128)],
            VerifyConfig(test_kind=test_kind,
                devtype=test_device.devtype,
                arch=test_device.arch,
                relative_atol=relative_atol)
    )

from transformers.pipelines import pipeline
import pybuda
from loguru import logger

class ModelWrapper(torch.nn.Module):
    def __init__(self, device, model):
        super().__init__()
        self.device = device
        self.config = model.config
        self.kwargs = inspect.getfullargspec(device.modules[0].original_forward)[0][1:]

    def forward(self, *args, **kwargs):
        inputs = list(args)
        kwinputs = [kwargs[kwname] for kwname in self.kwargs if kwname in kwargs]
        inputs.extend(kwinputs)

        self.device.push_to_inputs(inputs)
        output_q = pybuda.run_inference()
        outputs = output_q.get()
        return [o.value() for o in outputs]


from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
@pytest.mark.parametrize("variant", ["mrm8488/bert-tiny-finetuned-squadv2", "phiyodr/bert-base-finetuned-squad2"])
def test_pt_bert_qa_fallback(test_device, variant):
    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask = {"qa_outputs"}

    # Load model and tokenizer
    model = BertForQuestionAnswering.from_pretrained(variant)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(variant)
    
    # Inputs
    context = "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. \"The prophet and founding hero of modern archaeology\", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. "
    question = "What discipline did Winkelmann create?"

    # Sanity (PyTorch) run
    question_answerer_pt = pipeline('question-answering', model=model, tokenizer=tokenizer)
    answer_pt = question_answerer_pt(question=question, context=context)

    # TT run
    question_answerer_tt = pybuda_pipeline('question-answering', model=model, tokenizer=tokenizer)
    answer_tt = question_answerer_tt(question=question, context=context)

    logger.info(f"Context: {context}")
    logger.info(f"Question: {question}")

    logger.info(f"PT answer: {answer_pt['answer']}, score: {answer_pt['score']:.2f}")
    logger.info(f"TT answer: {answer_tt['answer']}, score: {answer_tt['score']:.2f}")

    assert answer_tt['start'] == answer_pt['start'], f"Start mismatch: TT:{answer_tt} PT:{answer_pt}"
    assert answer_tt['end'] == answer_pt['end'], f"End mismatch: TT:{answer_tt} PT:{answer_pt}"


# Embedding wrapper that extends and passes attention mask through - to run on host
class EmbWrapper(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
    def forward(self, input_ids, attention_mask, token_type_ids):
        attention_mask = attention_mask * 1.0
        emb_output = self.bert.embeddings(input_ids, token_type_ids)
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        return emb_output, extended_attention_mask

# Wrapper for encoders + QA output - to run on TT
class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, emb_output, extended_attention_mask):
        out = self.model.bert.encoder(emb_output, extended_attention_mask)
        out = self.model.qa_outputs(out.last_hidden_state)
        return out

@pytest.mark.parametrize("size", ["tiny", "base"])
def test_pt_bert_qa(test_device, size):
    pytest.skip()


    if size == "tiny":
        model_name = "mrm8488/bert-tiny-finetuned-squadv2"
        context = "Manuel Romero has been working hardly in the repository hugginface/transformers lately"
        input_q = {"context": context, "question": "For which company has worked Manuel Romero?"}
    elif size == "base":
        # https://huggingface.co/phiyodr/bert-base-finetuned-squad2
        model_name = "phiyodr/bert-base-finetuned-squad2"
        context = "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."

        input_q = {"context": context, "question": "What discipline did Winkelmann create?"}
    else:
        raise RuntimeError("Unknown size")

    model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tokenizer = BertTokenizer.from_pretrained(model_name, pad_to_max_length=True)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    examples = nlp._args_parser(input_q)
    preprocess_params, _, postprocess_params = nlp._sanitize_parameters()

    inputs = []
    for model_inputs in nlp.preprocess(examples[0], **preprocess_params):
        inputs.append( {
            "data": (model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["token_type_ids"]),
            "example": model_inputs["example"],
            "inputs": model_inputs})


    # Create pipeline, with encoders on TT
    cpu0 = pybuda.CPUDevice("cpu0", module=PyTorchModule("bert_embeddings", EmbWrapper(model.bert)))
    tt1 = pybuda.TTDevice("tt1",
            devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("encoder", EncoderWrapper(model)))

    for input in inputs:

        logger.info("Running on TT")
        cpu0.push_to_inputs(input["data"])
        output_q = pybuda.run_inference(_verify_cfg=VerifyConfig(relative_atol=0.3), _sequential=True)

        outputs = output_q.get()
        logits = outputs[0].value().detach()
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous().type(torch.float32)
        end_logits = end_logits.squeeze(-1).contiguous().type(torch.float32)

        res = {"start": start_logits, "end": end_logits, "example": input["example"], **input["inputs"]}

        tt_answer = nlp.postprocess([res], **postprocess_params)

        logger.info("Running on pytorch directly")
        out = model(*input["data"])

        res = {"start": out.start_logits.detach(), "end": out.end_logits.detach(), "example": input["example"], **input["inputs"]}

        pt_answer = nlp.postprocess([res], **postprocess_params)

        logger.info(input_q)
        logger.info(f"TT answer: {tt_answer}")
        logger.info(f"PT answer: {pt_answer}")

        assert tt_answer['start'] == pt_answer['start'], f"Start mismatch: TT:{tt_answer} PT:{pt_answer}"
        assert tt_answer['end'] == pt_answer['end'], f"Start mismatch: TT:{tt_answer} PT:{pt_answer}"
