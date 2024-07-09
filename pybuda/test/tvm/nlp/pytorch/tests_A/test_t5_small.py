# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch
from transformers import T5Config, T5Model, T5ForConditionalGeneration, T5Tokenizer, T5EncoderModel
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
import pybuda
from pybuda import (
    PyTorchModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
    CPUDevice,
    TTDevice,
)
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda._C.backend_api import BackendType, BackendDevice

from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind

from loguru import logger
from pybuda.pybudaglobal import TILE_DIM

from pybuda.op.eval.common import compare_tensor_to_golden
from test.utils import download_model

@pytest.mark.skip(reason="Tested with fallback")
def test_t5_small_decoder(test_kind, test_device):
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 1, 128, 512)

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    pretrained_name = "t5-small"
    config = T5Config.from_pretrained(pretrained_name, torchscript=True)
    model = T5Model(config)
    pretrained_model = download_model(T5Model.from_pretrained, pretrained_name)
    model.load_state_dict(pretrained_model.state_dict())

    mod = PyTorchModule("t5_small_block", model.decoder.block[0])

    hidden_states = torch.rand(*input_shape)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

@pytest.mark.skip(reason="Tested with fallback")
def test_t5_small_encoder(test_kind, test_device):        
    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 1, 128, 512)

    compiler_cfg = _get_global_compiler_config()
    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    pretrained_name = "t5-small"
    config = T5Config.from_pretrained(pretrained_name, torchscript=True)
    model = T5Model(config)
    pretrained_model = download_model(T5Model.from_pretrained, pretrained_name)
    model.load_state_dict(pretrained_model.state_dict())
    mod = PyTorchModule("t5_small_block", model.encoder.block[0])

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

@pytest.mark.skip(reason="Tested with fallback")
def test_t5_encoder_stack(test_kind, test_device):
    class ListWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states):
            for module in self.model.block:
                hidden_states = module(hidden_states)[0]
            hidden_states = self.model.final_layer_norm(hidden_states)
            return hidden_states

    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    input_shape = (1, 128, 512)

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"SelfAttention.relative_attention_bias"}
    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    pretrained_name = "t5-small"
    config = download_model(T5Config.from_pretrained, pretrained_name, torchscript=True)
    t5_model = T5Model(config)
    pretrained_model = T5Model.from_pretrained(pretrained_name)
    t5_model.load_state_dict(pretrained_model.state_dict())

    torch_mod = ListWrapper(t5_model.encoder)

    mod = PyTorchModule("t5_small_encoder", torch_mod)

    hidden_states = torch.rand(*input_shape)

    verify_module(
        mod,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        )
    )

class EmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, extended_attention_mask):
        inputs_embeds = self.model.embed_tokens(input_ids)
        return inputs_embeds, extended_attention_mask

class BlocksWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.encoder_decoder = module

    def forward(self, hidden_states, extended_attention_mask):
        for block in self.encoder_decoder.block:
            hidden_states = block(
                hidden_states,
                attention_mask=extended_attention_mask
            )[0]
        hidden_states = self.encoder_decoder.final_layer_norm(hidden_states)
        return hidden_states

class LMHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)

@pytest.mark.skip(reason="Tested with fallback")
def test_t5_encoder_pipeline(test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"SelfAttention.relative_attention_bias"}

    pretrained_name = "t5-small"
    config = T5Config.from_pretrained(pretrained_name, torchscript=True)
    t5_model = T5EncoderModel(config)
    t5_model.eval()
    # pretrained_model = T5EncoderModel.from_pretrained(pretrained_name)
    # t5_model.load_state_dict(pretrained_model.state_dict())

    encoder_embeddings = EmbWrapper(t5_model.encoder)
    encoder_stack = BlocksWrapper(t5_model.encoder)

    cpu0 = CPUDevice("cpu0", module=PyTorchModule("encoder_embeddings", encoder_embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("encoder_stack", encoder_stack))


    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # inputs = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt", max_length=64, pad_to_max_length=True, truncation=True)
    # input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]
    seq_len = 128
    input_ids = torch.randint(config.vocab_size, (1, seq_len))
    attention_mask = torch.ones((1, seq_len))
    extended_attention_mask = t5_model.get_extended_attention_mask(attention_mask, input_ids.size())
    cpu0.push_to_inputs(input_ids, extended_attention_mask)
    output_q = pybuda.run_inference()
    outputs = output_q.get()

    torch_outputs = t5_model(input_ids, attention_mask=attention_mask)
    assert compare_tensor_to_golden("t5_encoder", torch_outputs[0], outputs[0].value(), is_buda=True)


@pytest.mark.skip(reason="Tested with fallback")
def test_t5_pipeline(test_kind, test_device):
    class EmbWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = self.model.embed_tokens(input_ids)
            extended_attention_mask = self.model.get_extended_attention_mask(attention_mask, input_shape)
            return inputs_embeds, extended_attention_mask

    class BlocksWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.encoder_decoder = module

        def forward(self, hidden_states, extended_attention_mask):
            for block in self.encoder_decoder.block:
                hidden_states = block(
                    hidden_states,
                    attention_mask=extended_attention_mask
                )[0]
            hidden_states = self.encoder_decoder.final_layer_norm(hidden_states)
            return hidden_states

    class LMHeadWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.lm_head = model.lm_head

        def forward(self, hidden_states):
            return self.lm_head(hidden_states)

    if test_kind == TestKind.TRAINING: # only run recompute test in post-commit
        pytest.skip()

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"SelfAttention.relative_attention_bias"}
    if test_kind.is_training():
        test_device.devtype = BackendType.NoBackend

    pretrained_name = "t5-small"
    config = T5Config.from_pretrained(pretrained_name, torchscript=True)
    t5_model = T5ForConditionalGeneration(config)
    pretrained_model = T5ForConditionalGeneration.from_pretrained(pretrained_name)
    t5_model.load_state_dict(pretrained_model.state_dict())
    tokenizer = T5Tokenizer.from_pretrained("t5-small")


    encoder_embeddings = EmbWrapper(t5_model.encoder)
    encoder_stack = BlocksWrapper(t5_model.encoder)
    decoder_embeddings = EmbWrapper(t5_model.decoder)
    decoder_stack = BlocksWrapper(t5_model.decoder)
    lm_head = LMHeadWrapper(t5_model)

    cpu0 = CPUDevice("cpu0", module=PyTorchModule("encoder_embeddings", encoder_embeddings))
    tt1 = TTDevice("tt1", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("encoder_stack", encoder_stack))
    cpu2 = CPUDevice("cpu2", module=PyTorchModule("decoder_embeddings", decoder_embeddings))
    tt3 = TTDevice("tt3", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("decoder_stack", decoder_stack))
    cpu4 = CPUDevice("cpu4", module=PyTorchModule("lm_head", lm_head))
    # inputs = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt")
    # input_ids = inputs["input_ids"]
    # outputs = t5_model.generate(input_ids)
    # logger.info(f"Input text: The house is wonderful.\nTranslated text: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")


    inputs = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt", max_length=64, pad_to_max_length=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    cpu0.push_to_inputs(input_ids, attention_mask)
    output_q = pybuda.run_inference()
    outputs = output_q.get()




class BlocksWrapper(torch.nn.Module):
    def __init__(self, model, num_blocks):
        super().__init__()
        self.t5 = model
        self.num_blocks = num_blocks


    def shape(self, states):
        """projection"""
        return states.view(1, -1, self.t5.decoder.block[0].layer[0].SelfAttention.n_heads, self.t5.decoder.block[0].layer[0].SelfAttention.key_value_proj_dim).transpose(1, 2)

    def unshape(self, states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(1, -1, self.t5.decoder.block[0].layer[0].SelfAttention.inner_dim)
    
    def forward(self, encoder_last_hidden_state, decoder_input_ids, decoder_attention_mask, encoder_attention_mask, *past_key_values):
        presents = []
        hidden_states = self.t5.decoder.embed_tokens(decoder_input_ids)
        encoder_attention_mask = self.t5.invert_attention_mask(encoder_attention_mask)
        decoder_attention_mask = self.t5.get_extended_attention_mask(decoder_attention_mask, decoder_input_ids.size())

        
        for i, block in enumerate(self.t5.decoder.block):
            past_key_value = tuple([self.shape(past_key_values[(i * 4) + j]) for j in range(4)])

            outputs = block(
                hidden_states,
                past_key_value=past_key_value,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=True,
            )
            hidden_states = outputs[0]
            for j in range(2):
                present = outputs[1][j][:, :, -32:, :]
                present = self.unshape(present)
                presents.append(present)
            for j in range(2, 4):
                present = outputs[1][j]
                presents.append(present)

        hidden_states = self.t5.decoder.final_layer_norm(hidden_states)
        sequence_output = hidden_states

        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)

        lm_logits = self.t5.lm_head(sequence_output)

        return lm_logits, *presents, encoder_last_hidden_state

variants = ["t5-small", "t5-base", "t5-large", "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_t5_past_cache(variant, test_device):
    # Too slow for post-commit ci

    if test_device.devtype != BackendType.Silicon and variant != "t5-small":
        pytest.skip()

    import os
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
    os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "169536"
    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "30000"
    os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.enable_amp_light()

    model_name = variant
    #TODO: Try cherry-picking 5993b7f8
    # Load model""
    config = T5Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    max_length = config_dict["n_positions"]

    if "t5-small" in model_name:
        compiler_cfg.loopback_outputs = {"states_1": 1, "states_3": 2, "states_5": 3, "states_7": 4, 
                                        "states_25": 5, "states_27": 6, "states_29": 7, "states_31": 8,
                                        "states_49": 9, "states_51": 10, "states_53": 11, "states_55": 12,
                                        "states_73": 13, "states_75": 14, "states_77": 15, "states_79": 16,
                                        "states_97": 17, "states_99": 18, "states_101": 19,"states_103" :20,
                                        "states_121": 21, "states_123": 22, "states_125": 23, "states_127" :24,
                                        }
    elif "t5-base" in model_name:
        compiler_cfg.loopback_outputs = {"states_1": 1, "states_3": 2, "states_5": 3, "states_7": 4,
                                        "states_25": 5, "states_27": 6, "states_29": 7, "states_31": 8,
                                        "states_49": 9, "states_51": 10, "states_53": 11, "states_55": 12,
                                        "states_73": 13, "states_75": 14, "states_77": 15, "states_79": 16,
                                        "states_97": 17, "states_99": 18, "states_101": 19,"states_103" :20,
                                        "states_121": 21, "states_123": 22, "states_125": 23, "states_127" :24,
                                        "states_145": 25, "states_147": 26, "states_149": 27, "states_151": 28, 
                                        "states_169": 29, "states_171": 30, "states_173": 31, "states_175": 32, 
                                        "states_193": 33, "states_195": 34, "states_197": 35, "states_199": 36, 
                                        "states_217": 37, "states_219": 38, "states_221": 39, "states_223": 40, 
                                        "states_241": 41, "states_243": 42, "states_245": 43, "states_247": 44, 
                                        "states_265": 45, "states_267": 46, "states_269": 47, "states_271": 48,
                                        }
    elif "t5-large" in model_name:
        compiler_cfg.loopback_outputs = {"states_1": 1, "states_3": 2, "states_5": 3, "states_7": 4,
                                        "states_25": 5, "states_27": 6, "states_29": 7, "states_31": 8,
                                        "states_49": 9, "states_51": 10, "states_53": 11, "states_55": 12,
                                        "states_73": 13, "states_75": 14, "states_77": 15, "states_79": 16,
                                        "states_97": 17, "states_99": 18, "states_101": 19,"states_103" :20,
                                        "states_121": 21, "states_123": 22, "states_125": 23, "states_127" :24,
                                        "states_145": 25, "states_147": 26, "states_149": 27, "states_151": 28, 
                                        "states_169": 29, "states_171": 30, "states_173": 31, "states_175": 32, 
                                        "states_193": 33, "states_195": 34, "states_197": 35, "states_199": 36, 
                                        "states_217": 37, "states_219": 38, "states_221": 39, "states_223": 40, 
                                        "states_241": 41, "states_243": 42, "states_245": 43, "states_247": 44, 
                                        "states_265": 45, "states_267": 46, "states_269": 47, "states_271": 48,
                                        "states_289": 49, "states_291": 50, "states_293": 51, "states_295": 52, 
                                        "states_313": 53, "states_315": 54, "states_317": 55, "states_319": 56,
                                        "states_337": 57, "states_339": 58, "states_341": 59, "states_343": 60, 
                                        "states_361": 61, "states_363": 62, "states_365": 63, "states_367": 64, 
                                        "states_385": 65, "states_387": 66, "states_389": 67, "states_391": 68, 
                                        "states_409": 69, "states_411": 70, "states_413": 71, "states_415": 72, 
                                        "states_433": 73, "states_435": 74, "states_437": 75, "states_439": 76, 
                                        "states_457": 77, "states_459": 78, "states_461": 79, "states_463": 80,
                                        "states_481": 81, "states_483": 82, "states_485": 83, "states_487": 84, 
                                        "states_505": 85, "states_507": 86, "states_509": 87, "states_511": 88, 
                                        "states_529": 89, "states_531": 90, "states_533": 91, "states_535": 92, 
                                        "states_553": 93, "states_555": 94, "states_557": 95, "states_559": 96,
                                        }
    config = T5Config(**config_dict)
    model = download_model(T5ForConditionalGeneration.from_pretrained, model_name, config=config)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    input_length = 64
    input_text = "translate English to German: The house is wonderful. We have really enjoyed living here for the past eight years. The only problem that I have with it is that it is too small."
    encoder_inptus = tokenizer(input_text, return_tensors="pt", max_length=input_length, pad_to_max_length=True, truncation=True)
    input_ids = encoder_inptus["input_ids"]
    encoder_attention_mask = encoder_inptus["attention_mask"].float()

    encoder_out = model.encoder(input_ids=input_ids, attention_mask=encoder_attention_mask, return_dict=False)
    encoder_last_hidden_state = encoder_out[0].detach()
    inputs = (encoder_last_hidden_state, )
    decoder_input_ids = torch.zeros((1, 32), dtype=torch.int)
    decoder_attention_mask = torch.zeros((1, max_length))
    inputs += (decoder_input_ids, decoder_attention_mask, encoder_attention_mask)
    enc_past_cache_shape = (1, 1, max_length-32, config.d_model)   # (batch_size, 1, seq_len, head_dim)
    num_blocks = len(model.decoder.block)

    #we need to prepopulate the past cache with encoder outpts, otherwise we'd need two compilations. 
    model_out = model(decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_out)
    
    blocks = BlocksWrapper(model, num_blocks=num_blocks)
    pad_shape = (0, 0, 0, input_length-model_out[1][0][2].shape[2])
    for i in range(num_blocks):
        inputs += (torch.zeros(enc_past_cache_shape), torch.zeros(enc_past_cache_shape),
                   torch.unsqueeze(blocks.unshape(torch.nn.functional.pad(model_out[1][i][2], pad_shape)), 0),
                   torch.unsqueeze(blocks.unshape(torch.nn.functional.pad(model_out[1][i][3], pad_shape)), 0))
    tt0 = pybuda.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("t5", blocks))
    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=inputs)
    
    import time
    abs_index = 480
    decoder_attention_mask[0, abs_index] = 1
    generated_tokens = []
    current_token_index = 0
    input_length = 64 if test_device.devtype == BackendType.Silicon else 3
    for _ in range(input_length):
        if current_token_index == 1:
            start = time.time()

        generate_inputs = (encoder_last_hidden_state, decoder_input_ids, decoder_attention_mask, encoder_attention_mask )
        tt0.push_to_inputs(generate_inputs)
        pybuda.run_generate(input_count=1, write_index=current_token_index // TILE_DIM)
        ans = output_q.get()
        lm_head_out = ans[0].value().detach()
        next_token = torch.argmax(lm_head_out[0, current_token_index % TILE_DIM])
        generated_tokens.append(next_token)

        current_token_index += 1
        if current_token_index % TILE_DIM == 0:
            decoder_attention_mask[0, :current_token_index] = 1
            decoder_attention_mask[0, abs_index:] = 1
            decoder_input_ids[0, :] = tokenizer.pad_token_id
        decoder_input_ids[0, current_token_index % TILE_DIM] = next_token

        decoder_attention_mask[0, abs_index + (current_token_index % TILE_DIM)] = 1
    print(f"{current_token_index - 1} iterations took {time.time() - start} seconds, speed: {(current_token_index - 1) / (time.time() - start)} iters/sec")
    print(f"generated tokens: {tokenizer.decode(generated_tokens)}")

variants = ["t5-small", "t5-base", "t5-large", "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_t5_past_cache_pybuda_pipeline(variant, test_device):
    # Too slow for post-commit ci
    if test_device.devtype != BackendType.Silicon and variant != "t5-small":
        pytest.skip()

    import os
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
    os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "169536"
    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "30000"
    os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.enable_amp_light()

    model_name = variant

    # Load model""
    config = T5Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    max_length = config_dict["n_positions"]

    if "t5-small" in model_name:
        compiler_cfg.loopback_outputs = {"states_1": 1, "states_3": 2, "states_5": 3, "states_7": 4, 
                                        "states_25": 5, "states_27": 6, "states_29": 7, "states_31": 8,
                                        "states_49": 9, "states_51": 10, "states_53": 11, "states_55": 12,
                                        "states_73": 13, "states_75": 14, "states_77": 15, "states_79": 16,
                                        "states_97": 17, "states_99": 18, "states_101": 19,"states_103" :20,
                                        "states_121": 21, "states_123": 22, "states_125": 23, "states_127" :24,
                                        }
    elif "t5-base" in model_name:
        compiler_cfg.loopback_outputs = {"states_1": 1, "states_3": 2, "states_5": 3, "states_7": 4,
                                        "states_25": 5, "states_27": 6, "states_29": 7, "states_31": 8,
                                        "states_49": 9, "states_51": 10, "states_53": 11, "states_55": 12,
                                        "states_73": 13, "states_75": 14, "states_77": 15, "states_79": 16,
                                        "states_97": 17, "states_99": 18, "states_101": 19,"states_103" :20,
                                        "states_121": 21, "states_123": 22, "states_125": 23, "states_127" :24,
                                        "states_145": 25, "states_147": 26, "states_149": 27, "states_151": 28, 
                                        "states_169": 29, "states_171": 30, "states_173": 31, "states_175": 32, 
                                        "states_193": 33, "states_195": 34, "states_197": 35, "states_199": 36, 
                                        "states_217": 37, "states_219": 38, "states_221": 39, "states_223": 40, 
                                        "states_241": 41, "states_243": 42, "states_245": 43, "states_247": 44, 
                                        "states_265": 45, "states_267": 46, "states_269": 47, "states_271": 48,
                                        }
    elif "t5-large" in model_name:
        compiler_cfg.loopback_outputs = {"states_1": 1, "states_3": 2, "states_5": 3, "states_7": 4,
                                        "states_25": 5, "states_27": 6, "states_29": 7, "states_31": 8,
                                        "states_49": 9, "states_51": 10, "states_53": 11, "states_55": 12,
                                        "states_73": 13, "states_75": 14, "states_77": 15, "states_79": 16,
                                        "states_97": 17, "states_99": 18, "states_101": 19,"states_103" :20,
                                        "states_121": 21, "states_123": 22, "states_125": 23, "states_127" :24,
                                        "states_145": 25, "states_147": 26, "states_149": 27, "states_151": 28, 
                                        "states_169": 29, "states_171": 30, "states_173": 31, "states_175": 32, 
                                        "states_193": 33, "states_195": 34, "states_197": 35, "states_199": 36, 
                                        "states_217": 37, "states_219": 38, "states_221": 39, "states_223": 40, 
                                        "states_241": 41, "states_243": 42, "states_245": 43, "states_247": 44, 
                                        "states_265": 45, "states_267": 46, "states_269": 47, "states_271": 48,
                                        "states_289": 49, "states_291": 50, "states_293": 51, "states_295": 52, 
                                        "states_313": 53, "states_315": 54, "states_317": 55, "states_319": 56,
                                        "states_337": 57, "states_339": 58, "states_341": 59, "states_343": 60, 
                                        "states_361": 61, "states_363": 62, "states_365": 63, "states_367": 64, 
                                        "states_385": 65, "states_387": 66, "states_389": 67, "states_391": 68, 
                                        "states_409": 69, "states_411": 70, "states_413": 71, "states_415": 72, 
                                        "states_433": 73, "states_435": 74, "states_437": 75, "states_439": 76, 
                                        "states_457": 77, "states_459": 78, "states_461": 79, "states_463": 80,
                                        "states_481": 81, "states_483": 82, "states_485": 83, "states_487": 84, 
                                        "states_505": 85, "states_507": 86, "states_509": 87, "states_511": 88, 
                                        "states_529": 89, "states_531": 90, "states_533": 91, "states_535": 92, 
                                        "states_553": 93, "states_555": 94, "states_557": 95, "states_559": 96,
                                        }
    config = T5Config(**config_dict)
    model = download_model(T5ForConditionalGeneration.from_pretrained, model_name, config=config)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    input_length = 64
    input_text = "translate English to German: The house is wonderful. We have really enjoyed living here for the past eight years. The only problem that I have with it is that it is too small."
    encoder_inptus = tokenizer(input_text, return_tensors="pt", max_length=input_length, pad_to_max_length=True, truncation=True)
    input_ids = encoder_inptus["input_ids"]
    encoder_attention_mask = encoder_inptus["attention_mask"].float()

    encoder_out = model.encoder(input_ids=input_ids, attention_mask=encoder_attention_mask, return_dict=False)
    encoder_last_hidden_state = encoder_out[0].detach()
    inputs = (encoder_last_hidden_state, )
    decoder_input_ids = torch.zeros((1, 32), dtype=torch.int)
    decoder_attention_mask = torch.zeros((1, max_length))
    inputs += (decoder_input_ids, decoder_attention_mask, encoder_attention_mask)
    enc_past_cache_shape = (1, 1, max_length-32, config.d_model)   # (batch_size, 1, seq_len, head_dim)
    num_blocks = len(model.decoder.block)

    #we need to prepopulate the past cache with encoder outpts, otherwise we'd need two compilations. 
    model_out = model(decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_out)
    blocks = BlocksWrapper(model, num_blocks=num_blocks)
    pad_shape = (0, 0, 0, input_length-model_out[1][0][2].shape[2])
    for i in range(num_blocks):
        inputs += (torch.zeros(enc_past_cache_shape), torch.zeros(enc_past_cache_shape),
                   torch.unsqueeze(blocks.unshape(torch.nn.functional.pad(model_out[1][i][2], pad_shape)), 0),
                   torch.unsqueeze(blocks.unshape(torch.nn.functional.pad(model_out[1][i][3], pad_shape)), 0))
    tt0 = pybuda.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=PyTorchModule("t5", blocks))
    output_q = pybuda.initialize_pipeline(training=False, sample_inputs=inputs)

    abs_index = 480
    current_token_index = 0

    def wrap_generate(inputs):
        nonlocal current_token_index
        decoder_input_ids[:, current_token_index % TILE_DIM] = inputs[0][:,current_token_index]
        decoder_attention_mask[0, abs_index + (current_token_index % TILE_DIM)] = 1
        generate_inputs = (encoder_last_hidden_state, decoder_input_ids, decoder_attention_mask, encoder_attention_mask)
        tt0.push_to_inputs(generate_inputs)
        pybuda.run_generate(input_count=1, write_index=current_token_index // TILE_DIM)
        ans = output_q.get()
        lm_head_out = ans[0].value().detach()
        lm_head_out = lm_head_out[:, :(current_token_index % TILE_DIM) + 1, :]

        current_token_index += 1
        if current_token_index % TILE_DIM == 0:
            decoder_attention_mask[0, :current_token_index] = 1
            decoder_attention_mask[0, abs_index:] = 1
            decoder_input_ids[0, :] = tokenizer.pad_token_id
        return lm_head_out

    text_generator = pybuda_pipeline("text2text-generation", model=model, tokenizer=tokenizer, forward_fn=wrap_generate)

    import time
    start = time.time()
    answer = text_generator(
        input_text,
        num_beams=1,
        num_return_sequences=1,
        max_length=input_length,
        no_repeat_ngram_size=2,
    )
    end = time.time()
    print(f"Time taken: {end-start}, tokens/sec = {current_token_index/(end-start)}")
    print(answer)


class T5_encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t5 = model

    def forward(self, input_ids, attention_mask):
        return self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
    
class T5_decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t5 = model

    def forward(self, decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, decoder_encoder_attention_mask, *past_key_values):
        presents = []
        pkv = []
        for i, _ in enumerate(self.t5.decoder.block):
            pkv.append(tuple([past_key_values[(i * 4) + j] for j in range(4)]))

        outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=decoder_encoder_attention_mask,
            past_key_values = pkv,
        )
        sequence_output = outputs[0]
        presents = outputs[1]
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)

        lm_logits = self.t5.lm_head(sequence_output)    
        return lm_logits, *presents

variants = ["t5-small", "t5-base"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_t5_past_cache_enc_dec(variant, test_device):
    # Too slow for post-commit ci

    if test_device.devtype != BackendType.Silicon and variant != "t5-small":
        pytest.skip()

    import os
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
    os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "120000"
    # os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "169536"
    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "30000"
    os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
    os.environ["TT_BACKEND_PROFILER"] = "1"
    os.environ["TT_BACKEND_EPOCH_BIN_NUM_SLOTS"] = "64"
    os.environ["PYBUDA_ROTATE_PAST_CACHE_PARAMS"] = "1"
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.default_df_override = pybuda._C.Float16_b
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.enable_amp_light()
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.enable_link_past_cache_ios = True

    # pybuda.set_configuration_options(performance_trace=pybuda.PerfTraceLevel.VERBOSE)
    model_name = variant
    config = T5Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    # n_layers = 1
    # config_dict["num_layers"] = n_layers
    # config_dict["num_decoder_layers"] = n_layers
    max_length = config_dict["n_positions"]

    config = T5Config(**config_dict)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    num_blocks = len(model.decoder.block)
    # num_blocks = n_layers
    for i in range(num_blocks):
        pybuda.config.override_op_size(f"t5.decoder.block.{i}.layer.0.SelfAttention.k.weight_cache_nop", [1, 1])
        pybuda.config.override_op_size(f"t5.decoder.block.{i}.layer.0.SelfAttention.v.weight_cache_nop", [1, 1])
        pybuda.config.override_t_stream_shape(f"t5.decoder.block.{i}.layer.0.SelfAttention.k.weight_cache_nop", [15, 1])
        pybuda.config.override_t_stream_shape(f"t5.decoder.block.{i}.layer.0.SelfAttention.v.weight_cache_nop", [15, 1])

    input_length = 64
    input_text = "translate English to German: The house is wonderful. We have really enjoyed living here for the past eight years. The only problem that I have with it is that it is too small and the parks are not very close."
    encoder_inptus = tokenizer(input_text, return_tensors="pt", max_length=input_length, pad_to_max_length=True, truncation=True)
    input_ids = encoder_inptus["input_ids"].int()
    encoder_attention_mask = encoder_inptus["attention_mask"].float()

    encoder_last_hidden_state_shape = (1, input_length, config.d_model)
    encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)
    decoder_input_ids = torch.zeros((1, TILE_DIM), dtype=torch.int)
    decoder_attention_mask = torch.zeros((1, max_length))
    
    enc_past_cache_self_shape = (1, config.num_heads, max_length-32, config.d_kv)
    enc_past_cache_cross_shape = (1, 1, 1, 1)
    decoder_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, encoder_attention_mask]
    for _ in range(num_blocks):
        decoder_ca_inputs +=  [torch.zeros(enc_past_cache_self_shape), torch.zeros(enc_past_cache_self_shape),
                   torch.zeros(enc_past_cache_cross_shape), torch.zeros(enc_past_cache_cross_shape)]

    enc_past_cache_cross_shape = (1, config.num_heads, input_length, config.d_kv)
    decoder_no_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, encoder_attention_mask]
    for _ in range(num_blocks):
        decoder_no_ca_inputs += [torch.zeros(enc_past_cache_self_shape), torch.zeros(enc_past_cache_self_shape),
                   torch.zeros(enc_past_cache_cross_shape), torch.zeros(enc_past_cache_cross_shape)]

    encoder_module = PyTorchModule("T5_encoder", T5_encoder(model))
    decoder_module_cross_attention = PyTorchModule("T5_decoder_with_ca", T5_decoder(model))
    decoder_module_no_cross_attention = PyTorchModule("T5_decoder_no_ca", T5_decoder(model))
    tt0 = pybuda.TTDevice(
        "tt0", 
        devtype=test_device.devtype, 
        arch=test_device.arch, 
        module=[encoder_module, decoder_module_cross_attention, decoder_module_no_cross_attention])

    output_q = pybuda.initialize_pipeline(
        training=False,
        sample_inputs=(
            (input_ids, encoder_attention_mask),
            (decoder_ca_inputs),
            (decoder_no_ca_inputs),
        ))


    import time
    start = time.time()
    tt0.set_active_subgraph(0)
    tt0.push_to_inputs((input_ids, encoder_attention_mask))
    pybuda.run_forward()
    ans = output_q.get()
    encoder_last_hidden_state = ans[0].value().detach()
    first_current_index = max_length - TILE_DIM
    decoder_attention_mask[0, first_current_index] = 1
    generated_tokens = []
    current_token_index = 0

    tokens_to_generate = input_length if test_device.devtype == BackendType.Silicon else 3
    for _ in range(tokens_to_generate):
        if current_token_index == 1:
            start_1 = time.time()
        if current_token_index == 0:
            tt0.set_active_subgraph(1)
            generate_inputs = (decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, encoder_attention_mask)
            tt0.push_to_inputs(generate_inputs)
            pybuda.run_generate(input_count=1, write_index=0)
            ans = output_q.get()
        else:
            tt0.set_active_subgraph(2)
            generate_inputs = (decoder_input_ids, decoder_attention_mask, encoder_attention_mask)
            tt0.push_to_inputs(generate_inputs)
            pybuda.run_generate(input_count=1, write_index=0)
            ans = output_q.get()

        lm_head_out = ans[0].value().detach()
        next_token = torch.argmax(lm_head_out[0, current_token_index % TILE_DIM])
        generated_tokens.append(next_token)

        current_token_index += 1
        if current_token_index % TILE_DIM == 0:
            past_cache_pages = current_token_index // TILE_DIM
            # after one page of past cache, we have to rotate. 
            tt0.set_active_subgraph(3)
            pybuda.run_generate(input_count=0, write_index=0)

            pages_current = 1
            decoder_attention_mask[0, -(past_cache_pages + pages_current) * TILE_DIM:] = 1
            decoder_attention_mask[0, first_current_index:] = 0
            decoder_input_ids[0, :] = tokenizer.pad_token_id

        decoder_input_ids[0, current_token_index % TILE_DIM] = next_token
        decoder_attention_mask[0, first_current_index + (current_token_index % TILE_DIM)] = 1
    end = time.time()
    print(f"{current_token_index} iterations took {end - start} seconds, speed: {(current_token_index) / (end - start)} iters/sec")
    print(f"Discounting initial token, {current_token_index - 1} iterations took {end - start_1} seconds, speed: {(current_token_index - 1) / (end - start_1)} iters/sec")
    print(f"generated tokens: {tokenizer.decode(generated_tokens)}")
variants = ["t5-small", "t5-base"]
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_t5_past_cache_model(variant):
    model_name = variant

    config = T5Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    max_length = config_dict["n_positions"]

    config = T5Config(**config_dict)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    num_blocks = len(model.decoder.block)

    input_length = 64
    input_text = "translate English to German: The house is wonderful. We have really enjoyed living here for the past eight years. The only problem that I have with it is that it is too small."
    encoder_inptus = tokenizer(input_text, return_tensors="pt", max_length=input_length, pad_to_max_length=True, truncation=True)
    input_ids = encoder_inptus["input_ids"].int()
    encoder_attention_mask = encoder_inptus["attention_mask"].float()

    encoder_last_hidden_state_shape = (1, input_length, config.d_model)
    encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)
    decoder_input_ids = torch.zeros((1, 32), dtype=torch.int)
    decoder_attention_mask = torch.zeros((1, max_length))
    
    enc_past_cache_self_shape = (1, config.num_heads, max_length-32, config.d_kv)
    enc_past_cache_cross_shape = (1, 1, 1, 1)
    past_key_values = []
    for _ in range(num_blocks):
        past_key_values += [torch.zeros(enc_past_cache_self_shape), torch.zeros(enc_past_cache_self_shape),
                   torch.zeros(enc_past_cache_cross_shape), torch.zeros(enc_past_cache_cross_shape)]

    encoder_module = T5_encoder(model)
    decoder_module = T5_decoder(model)

    import time
    start = time.time()
    encoder_last_hidden_state = encoder_module(input_ids, encoder_attention_mask)[0]

    abs_index = 480
    decoder_attention_mask[0, abs_index] = 1
    generated_tokens = []
    current_token_index = 0

    for _ in range(64):
        if current_token_index == 1:
            start_1 = time.time()
        ans = decoder_module(decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, encoder_attention_mask, *past_key_values)

        lm_head_out = ans[0]
        next_token = torch.argmax(lm_head_out[0, current_token_index % TILE_DIM])
        generated_tokens.append(next_token)

        current_token_index += 1
        if current_token_index % TILE_DIM == 0:
            current_tile_index = current_token_index // TILE_DIM
            next_tile_index = current_tile_index + 1
            for idx, kv in enumerate(ans[1:]):
                past_key_values[(idx * 4) + 0][:, :, -next_tile_index*TILE_DIM:-current_tile_index*TILE_DIM, :] = kv[0][:, :, -TILE_DIM:, :]
                past_key_values[(idx * 4) + 1][:, :, -next_tile_index*TILE_DIM:-current_tile_index*TILE_DIM, :] = kv[1][:, :, -TILE_DIM:, :]
            decoder_attention_mask[0, -next_tile_index*TILE_DIM:] = 1
            decoder_attention_mask[0, abs_index:] = 0
            decoder_input_ids[0, :] = tokenizer.pad_token_id

        decoder_input_ids[0, current_token_index % TILE_DIM] = next_token
        decoder_attention_mask[0, abs_index + (current_token_index % TILE_DIM)] = 1
    end = time.time()
    print(f"{current_token_index} iterations took {end - start} seconds, speed: {(current_token_index) / (end - start)} iters/sec")
    print(f"Discounting initial token, {current_token_index - 1} iterations took {end - start_1} seconds, speed: {(current_token_index - 1) / (end - start_1)} iters/sec")
    print(f"generated tokens: {tokenizer.decode(generated_tokens)}")


