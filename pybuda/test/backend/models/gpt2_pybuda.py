# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
'''
Doing GPT-2 exploration in Python script instead of notebook because the "edit PyBuda -> rebuild env -> restart notebook" workflow was too painful.
'''
import os

import pybuda
from pybuda import (BackendType, BackendDevice, TTDevice, CPUDevice, PyTorchModule, DataFormat)
from pybuda.verify import verify_module, verify_module_pipeline, VerifyConfig, TestKind
from pybuda.verify.utils import CPUCombiner
from pybuda.config import CompilerConfig, _get_global_compiler_config
from pybuda.op.eval.common import compare_tensor_to_golden, calculate_pcc
import torch
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

import pytest
import math

class CPUIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, act):
        return act + 0

class EmbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gpt2 = model.transformer

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.gpt2.wte(input_ids)
        position_ids = torch.arange(len(input_ids[0])).unsqueeze(0)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        return hidden_states, extended_attention_mask

class BlocksWrapper(torch.nn.Module):
    def __init__(self, model):
        '''
        model: GPT2LMHeadModel
        '''
        super().__init__()
        self.gpt2 = model.transformer

    def forward(self, hidden_states, extended_attention_mask=None):
        for block in self.gpt2.h:
            hidden_states = block(
                hidden_states,
                attention_mask=extended_attention_mask
            )[0]
        hidden_states = self.gpt2.ln_f(hidden_states)
        return hidden_states

class BlockWrapper(torch.nn.Module):
    def __init__(self, model, num_block):
        '''
        model: GPT2Model
        num_block: how many decoders to run
        '''
        super().__init__()
        self.model = model
        self.num_block = num_block

    def forward(self, hidden_states, extended_attention_mask=None):
        for i in range(self.num_block):
            hidden_states = self.model.h[i](
                hidden_states,
                attention_mask=extended_attention_mask
            )[0]
        return hidden_states

class LMHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


@pytest.mark.parametrize('devtype', (BackendType.Golden, BackendType.Silicon), ids=('Golden', 'Silicon'))
# @pytest.mark.parametrize('test_kind', (TestKind.INFERENCE, TestKind.TRAINING), ids=('inference', 'training'))
def test_pt_gpt2_block(devtype, test_device, test_kind):
    '''
    FP32 GPT2 block in PyBuda
    '''
    model = GPT2Model.from_pretrained("gpt2")
    block = PyTorchModule("gpt2_block_backend",model.h[0])

    relative_atol = 0.3 if test_kind.is_training() else 0.1

    verify_module(block, [(1, 64, 768),],
            VerifyConfig(test_kind=test_kind, devtype=devtype, arch=test_device.arch, accumulation_steps=1, relative_atol=relative_atol,
                # fp32_fallback=DataFormat.Float32,
                         atol = {torch.float32: 1000000}, # skip atol
                         pcc=0.95,
            waive_gradient_errors={'attn.c_attn.bias_1'}),
            input_params=[{"requires_grad": False}],
            uniform_inputs=True,
    )


# @pytest.mark.parametrize('devtype', (BackendType.Golden, BackendType.Silicon), ids=('Golden', 'Silicon'))
@pytest.mark.parametrize('num_block', (1, 12), ids=('block1', 'block12'))
@pytest.mark.parametrize('sequential', (True, False), ids=('sequential', 'concurrent'))
def test_gpt2_block_train_manual_loop(test_device, num_block, sequential):
    '''
    FWD+BWD and Loss on TTDevice, OPT in PyTorch CPU
    '''

    if sequential == False:
        pytest.skip() # hangs

    opt_on_cpu = False

    torch.manual_seed(0)

    if test_device.devtype != BackendType.Silicon:
        pytest.skip() # golden hangs on get_parameter_gradients()

    model = GPT2Model.from_pretrained("gpt2")
    block = PyTorchModule("gpt2_block_backend", BlockWrapper(model, num_block))

    embedding = EmbWrapper(GPT2LMHeadModel.from_pretrained('gpt2'))

    pybuda.set_configuration_options(accumulate_df=DataFormat.Float32)

    # Get pretrained GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    max_length=64
    text = """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best perf
    """

    input_pt = tokenizer(text, return_tensors='pt', max_length=max_length,
            padding='max_length',
            truncation=True)

    embed_out = embedding(**input_pt)
    input_hidden = embed_out[0].detach()
    input_mask = embed_out[1].detach()

    # Random targets of correct size
    targets = input_hidden + torch.rand(input_hidden.shape)

    # PyTorch training loop
    blocks_pt = BlockWrapper(GPT2Model.from_pretrained('gpt2'), num_block)
    opt_pt = torch.optim.SGD(blocks_pt.parameters(), lr=5e-5)
    loss_fn = torch.nn.L1Loss()
    loss_pt = []
    for step in range(5):
        opt_pt.zero_grad()
        out_pt = blocks_pt(input_hidden, input_mask)
        loss = loss_fn(out_pt, targets)
        loss_pt.append(loss.item())
        loss.backward()
        opt_pt.step()

    opt_tt = None
    if not opt_on_cpu:
        opt_tt = pybuda.optimizers.SGD(learning_rate=5e-5, device_params=True)

    fp32_fallback = DataFormat.Float32 if test_device.arch == BackendDevice.Wormhole_B0 else DataFormat.Float16_b

    # Set up TT devices
    tt0 = TTDevice('tt0', module=block, optimizer=opt_tt, fp32_fallback=fp32_fallback, devtype=test_device.devtype, arch=test_device.arch)


    # Whether or not we do optimizer on CPU, we compute Loss on cpu
    cpu0 = CPUDevice('cpu0', module=PyTorchModule('identity', CPUIdentity()))
    cpu0.place_loss_module(pybuda.PyTorchModule('l1loss', torch.nn.L1Loss()))
 
    #_get_global_compiler_config().enable_tvm_dropout = False
    #_get_global_compiler_config().enable_tvm_unsupported_ops = True

    # Compile
    checkpoint_q = pybuda.initialize_pipeline(training=True,
                               _sequential=sequential,
                               sample_inputs=(input_hidden, input_mask),
                               sample_targets=(targets,),
                               _verify_cfg=VerifyConfig(test_kind=TestKind.TRAINING,
                                                        devtype=test_device.devtype,
                                                        arch=test_device.arch,
                                                        accumulation_steps=1,
                                                        relative_atol=.3,
                                                        fp32_fallback=fp32_fallback,
                                                        waive_gradient_errors={"attn.c_attn.bias_1",},
                                                        scale_loss=1.0,
                                                       ))

    loss_q = pybuda.run.get_loss_queue()

    # PyBuda training loop
    for step in range(5):
        tt0.push_to_inputs((input_hidden, input_mask))
        cpu0.push_to_target_inputs(targets)

        pybuda.run_forward(input_count=1, _sequential=sequential)
        pybuda.run_backward(input_count=1, zero_grad=True, _sequential=sequential)

        if opt_on_cpu:
            grads = pybuda.get_parameter_gradients(tt0, _sequential=sequential)
            params = pybuda.get_parameter_checkpoint(tt0, _sequential=sequential)

            for name in params[0].keys():
                # Set grad for each torch tensor
                grad = grads[0][name].value()
                if torch.isinf(grad).any():
                    print('*'*50)
                    print(f"INF value found in gradient {name}")
                    print('*'*50)

                params[0][name].value().grad = grad


            opt = torch.optim.SGD([p.value() for p in params[0].values()], lr=5e-5)
            opt.step()

            pybuda.update_device_parameters(tt0, params, _sequential=sequential)

        else:
            pybuda.run_optimizer(_sequential=sequential)


    losses_pb = []
    while not loss_q.empty():
        losses_pb.append(loss_q.get()[0])

    pybuda.shutdown()

    print('pybuda loss history:')
    print(losses_pb)
    print('PyTorch loss history:')
    print(loss_pt)

    rel_tol = 0
    if test_device.arch == BackendDevice.Grayskull and num_block == 1:
        rel_tol = 0.038
    elif test_device.arch == BackendDevice.Grayskull and num_block == 12:
        rel_tol = 0.058
    elif test_device.arch == BackendDevice.Wormhole_B0 and num_block == 1:
        rel_tol = 0.018
    elif test_device.arch == BackendDevice.Wormhole_B0 and num_block == 12:
        rel_tol = 0.20

    assert len(losses_pb) == len(loss_pt)
    for i, l_pb in enumerate(losses_pb):
        print(f"index={i} pybuda loss/pytorch loss={l_pb/loss_pt[i]}")
        assert math.isclose(l_pb, loss_pt[i], rel_tol=rel_tol, abs_tol=0.03)

class GPT2BlockWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gpt2 = model

    def forward(self, hidden_states, extended_attention_mask=None):
        hidden_states = hidden_states / 20
        for block in self.gpt2.h:
            hidden_states = block(
                hidden_states,
                attention_mask=extended_attention_mask
            )[0]
        hidden_states = self.gpt2.ln_f(hidden_states)
        return hidden_states

@pytest.mark.parametrize('devtype', (BackendType.Golden, BackendType.Silicon), ids=('Golden', 'Silicon'))
#@pytest.mark.parametrize('test_kind', (TestKind.INFERENCE, TestKind.TRAINING), ids=('inference', 'training'))
def test_pt_gpt2_blocks(devtype, test_kind):
    if test_kind.is_training():
        pytest.skip() # failing for a while
    '''
    FP32 GPT2 blocks in PyBuda
    '''
    #model = GPT2LMHeadModel.from_pretrained("gpt2")
    #blocks = PyTorchModule("gpt2_blocks", BlocksWrapper(model))
    model = GPT2Model.from_pretrained("gpt2")
    model_wrapped = GPT2BlockWrapper(model)
    blocks = PyTorchModule("gpt2_blocks", model_wrapped)

    torch.manual_seed(0)

    pybuda.set_configuration_options(accumulate_df=DataFormat.Float32)
    compiler_cfg = _get_global_compiler_config()

    relative_atol = 0.3 if test_kind.is_training() else 0.1

    verify_module(blocks, [(1, 64, 768),],
            VerifyConfig(test_kind=test_kind, devtype=devtype, arch=BackendDevice.Wormhole_B0, accumulation_steps=1, relative_atol=relative_atol, pcc=.95, fp32_fallback=DataFormat.Float32,
            waive_gradient_errors={'attn.c_attn.bias_1'}),
            input_params=[{"requires_grad": False}],
    )


@pytest.mark.parametrize('devtype', (BackendType.Golden, BackendType.Silicon), ids=('Golden', 'Silicon'))
@pytest.mark.parametrize('dataformat', (DataFormat.Float32, DataFormat.Float16_b), ids=('fp32', 'bf16'))
def test_gpt2_inference(devtype, dataformat):
    '''
    This test will pass a real sentence through a tokenizer, GPT2 embeddings, and GPT2 blocks in Silicon and in PyTorch and compare the outputs.
    '''
    if (dataformat == DataFormat.Float16_b):
        pytest.skip() # failing for a while

    pybuda.set_configuration_options(accumulate_df=dataformat)
    compiler_cfg = _get_global_compiler_config()

    # Get pretrained GPT2
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Split up GPT2 into embeddings, blocks, and lm head
    embedding = EmbWrapper(model)
    blocks = BlocksWrapper(model)
    lm_head = LMHeadWrapper(model)

    # GPT2 blocks ground truth
    max_length=64
    text = """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality
    while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.
    We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."""*4

    input_pt = tokenizer(text, return_tensors='pt', max_length=max_length,
            padding='max_length',
            truncation=True)
    print('input # tokens')
    print(input_pt['input_ids'].shape)

    embed_out = embedding(**input_pt)
    res_pt = blocks(*embed_out)

    # Set up TT Device with module
    buda_blocks = pybuda.PyTorchModule("gpt2_blocks", blocks)
    tt0 = pybuda.TTDevice('tt0', devtype=devtype, arch=BackendDevice.Wormhole_B0, fp32_fallback=dataformat)
    tt0.place_module(buda_blocks)

    hidden_tt = embed_out[0].clone().detach()
    mask_tt = embed_out[1].clone().detach()

    # Run on TT hardware
    tt0.push_to_inputs((hidden_tt, mask_tt))
    output_q = pybuda.run_inference()
    out_tt = output_q.get()
    res_tt = out_tt[0].value().detach()

    # Compare
    print('PyTorch blocks out')
    print(res_pt)
    print('TT blocks out')
    print(res_tt)

    res_tt = res_tt.float() # if we did BF16 math, the rest needs this as FP32

    print(f'Output has NAN? {torch.isnan(res_tt).any()}')
    print(f'Output has INF? {torch.isinf(res_tt).any()}')
    print("Max ATOL Delta: " + "{:.3e}".format(torch.max(torch.abs(res_pt - res_tt)).item()))
    print("Max RTOL Delta: " + "{:.3e}".format(torch.max(torch.abs(res_pt - res_tt)/res_tt).item()))
    print(f"PCC: {calculate_pcc(res_pt, res_tt)}")

    # Check LMHead output
    lmhead_pt = lm_head(res_pt)
    lmhead_tt = lm_head(res_tt)

    next_token_pt = torch.argmax(lmhead_pt, dim=-1)[0][max_length-1]
    next_token_tt = torch.argmax(lmhead_tt, dim=-1)[0][max_length-1]

    print('Next token PT')
    print(next_token_pt)
    print('Next token TT')
    print(next_token_tt)
    assert next_token_pt == next_token_tt



@pytest.mark.parametrize('devtype', (BackendType.Golden, BackendType.Silicon), ids=('Golden', 'Silicon'))
# @pytest.mark.parametrize('test_kind', (TestKind.INFERENCE, TestKind.TRAINING), ids=('inference', 'training'))
def test_pt_gpt2_generate(devtype, test_kind):
    compiler_cfg = pybuda.config._get_global_compiler_config()

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    embedding = EmbWrapper(model)
    blocks = BlocksWrapper(model)
    lm_head = LMHeadWrapper(model)

    for name, param in blocks.gpt2.h[0].attn.c_attn.named_parameters():
        print(param.shape)
    print(blocks.gpt2.h[0].attn.embed_dim)
    print(blocks.gpt2.h[0].attn.split_size)

    buda_blocks = pybuda.PyTorchModule("gpt2_blocks", blocks)

    cpu0 = pybuda.CPUDevice("cpu0", module=pybuda.PyTorchModule("gpt2_embeddings", embedding))
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, arch=BackendDevice.Wormhole_B0, num_chips=1,
                           module=buda_blocks)
    cpu1 = pybuda.CPUDevice("cpu1", module=pybuda.PyTorchModule("gpt2_lm_head", lm_head))

    dummy_text = "Text to generate input tensor for the compiler "
    dummy_input = tokenizer(dummy_text, return_tensors='pt', max_length=32, padding='max_length', truncation=True)


    '''
    Straight GPT2
    '''
    output_hug = model(**dummy_input)



    input_ids_tt = dummy_input['input_ids'].clone().detach().int()
    attn_mask_tt = dummy_input['attention_mask'].clone().detach().int()

    last_prefix_token =  (dummy_input['attention_mask']==0).nonzero(as_tuple=True)[1][0].item() - 1


    tokens_to_generate = 10
    for i in range(tokens_to_generate):
        cpu0.push_to_inputs((input_ids_tt, attn_mask_tt))
        output_q = pybuda.run_inference()
        outputs = output_q.get()
        lm_head_out = outputs[0].value().detach()
        next_token = torch.argmax(lm_head_out, dim=-1)[0][last_prefix_token + i]
        next_token_index = last_prefix_token + i + 1
        input_ids_tt[0][next_token_index] = next_token
        attn_mask_tt[0][next_token_index] = 1

    pybuda.shutdown()

    generated_text_tt = tokenizer.decode(input_ids_tt[0][:next_token_index].numpy().tolist())

    print(f'Input text: {dummy_text}')
    print(f'Generated text: {generated_text_tt}')


if __name__ == '__main__':
    test_pt_gpt2_block(devtype=BackendType.Golden, test_kind=TestKind.TRAINING)
