# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
import pytest

import jax
import numpy as np
from jax import numpy as jnp
from flax import linen as nn
from transformers import FlaxBertModel
from transformers.models.bert.modeling_flax_bert import FlaxBertEmbeddings, FlaxBertAttention, FlaxBertSelfAttention, FlaxBertIntermediate, FlaxBertEncoder, FlaxBertPooler
from transformers.models.bert.configuration_bert import BertConfig

from pybuda import (
    JaxModule,
    VerifyConfig,
)
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from pybuda.config import CompileDepth, _get_global_compiler_config


def test_bert(test_kind, test_device):
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:  
        pytest.skip()

    class Wrapper(nn.Module):
        model: FlaxBertModel
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            self.attention_mask = jax.random.randint(key, (1, 128), 0, 2)
            self.token_type_ids = jax.random.randint(key,  (1, 128), 0, 2)
            self.position_ids = jax.random.randint(key,  (1, 128), 0, 512 - 1)

        def __call__(self, input_ids):
            act = self.model(input_ids, self.attention_mask, self.token_type_ids, self.position_ids)

            return act

    compiler_cfg = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_cfg.compile_depth = CompileDepth.FULL
    else:
        # Fails on backend with following error:
        # "Backward error: element 0 of tensors does not require grad and does not have a grad_fn"
        compiler_cfg.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY 
    compiler_cfg.cpu_fallback_ops.add("all")
    compiler_cfg.cpu_fallback_ops.add("where")

    # Fetch pre-trained model
    framework_module = FlaxBertModel.from_pretrained("prajjwal1/bert-tiny", from_pt=True, add_pooling_layer=False)
    
    # Initialize a model
    key = jax.random.PRNGKey(0)
    input_shape = (1, 128)
    input_ids = jax.random.randint(key, input_shape, 0, 100)
    
    framework_module = Wrapper(
        model=framework_module, 
        dtype=jax.numpy.float32,
    )
    vars = framework_module.init(key, 
        input_ids=input_ids,
    )
    framework_module = framework_module.bind(vars)

    # # Run module
    # res = framework_module(
    #     input_ids=input_ids,
    # )

    pybuda_module = JaxModule("bert_jax", framework_module)
    verify_module(
        pybuda_module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_bert_embeddings(test_kind, test_device):
    # No need to instance specific module as full model tests are up and running
    pytest.skip()

    # TODO: Remove
    if test_kind.is_training():
        pytest.skip()

    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:  
        pytest.skip()

    class Embedding(nn.Module):
        config: BertConfig
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            self.token_type_ids = jax.random.randint(key,  (1, 128), 0, 2)
            self.position_ids = jax.random.randint(key,  (1, 128), 0, config.max_position_embeddings - 1)
            self.attention_mask = jax.random.randint(key, (1, 128), 0, 2)

            self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
            self.dense = nn.Dense(features=128, use_bias=True)

        def __call__(self, input_ids):
            act = self.embeddings(input_ids, self.token_type_ids, self.position_ids, self.attention_mask)
            act = nn.softmax(act)

            return act

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.FULL
    else:
        compiler_config.compile_depth = CompileDepth.FULL
    compiler_config.retain_tvm_python_files = True 
    compiler_config.cpu_fallback_ops.add("all")
    compiler_config.cpu_fallback_ops.add("where")

    # Initialize module
    key = jax.random.PRNGKey(0)

    config = BertConfig()
    input_ids_shape = (1, 128)
    input_ids = jax.random.randint(key, input_ids_shape, 0, 100)

    dtype = jax.numpy.float32
    framework_module = Embedding(
        config=config, 
        dtype=dtype,
    )
    vars = framework_module.init(key, 
        input_ids=input_ids,
    )
    framework_module = framework_module.bind(vars)

    # Run module
    res = framework_module(
        input_ids=input_ids,
    )

    pybuda_module = JaxModule("bert_embeddings_jax", framework_module)
    verify_module(
        pybuda_module,
        (input_ids_shape,),
        inputs=[(input_ids,),],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
    


def test_bert_attention(test_kind, test_device):
    # No need to instance specific module as full model tests are up and running
    pytest.skip()

    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:  
        pytest.skip()

    class Wrapper(nn.Module):
        config: BertConfig
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            self.attention_mask = jax.random.randint(key, ((1, 1)), 0, 2)
            self.layer_head_mask = jnp.ones((1,))

            self.attn = FlaxBertAttention(self.config, dtype=self.dtype)

        def __call__(self, hidden_state):
            act = self.attn(hidden_state, self.attention_mask, self.layer_head_mask)

            return act

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.FULL
    else:
        # Fails on backend with following error:
        # "Backward error: element 0 of tensors does not require grad and does not have a grad_fn"
        compiler_config.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    # Initialize module
    config = BertConfig()

    hidden_state_shape = (1, 1, 768)
    key = jax.random.PRNGKey(0)
    hidden_state = jax.random.uniform(key, hidden_state_shape)

    dtype = jax.numpy.float32
    framework_module = Wrapper(
        config=config, 
        dtype=dtype,
    )
    vars = framework_module.init(key, 
        hidden_state=hidden_state,
    )
    framework_module = framework_module.bind(vars)

    # Run module
    # res = framework_module(
    #     hidden_state=hidden_state,
    # )

    pybuda_module = JaxModule("bert_attention_jax", framework_module)
    verify_module(
        pybuda_module,
        (hidden_state_shape,),
        inputs=[(hidden_state,),],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )

    
def test_bert_intermediate(test_kind, test_device):
    # No need to instance specific module as full model tests are up and running
    pytest.skip()
    
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:  
        pytest.skip()

    class Wrapper(nn.Module):
        config: BertConfig
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            self.intermediate = FlaxBertIntermediate(self.config, dtype=self.dtype)

        def __call__(self, hidden_state):
            act = self.intermediate(hidden_state)

            return act

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.FULL
    else:
        # Fails on backend with following error:
        # "Backward error: element 0 of tensors does not require grad and does not have a grad_fn"
        compiler_config.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    # Initialize module
    config = BertConfig()

    hidden_state_shape = (1, 1, 768)
    key = jax.random.PRNGKey(0)
    hidden_state = jax.random.uniform(key, hidden_state_shape)

    dtype = jax.numpy.float32
    framework_module = Wrapper(
        config=config, 
        dtype=dtype,
    )
    vars = framework_module.init(key, 
        hidden_state=hidden_state,
    )
    framework_module = framework_module.bind(vars)

    # Run module
    # res = framework_module(
    #     hidden_state=hidden_state,
    # )

    pybuda_module = JaxModule("bert_intermediates_jax", framework_module)
    verify_module(
        pybuda_module,
        (hidden_state_shape,),
        inputs=[(hidden_state,),],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )


def test_bert_self_attention(test_kind, test_device):
    # No need to instance specific module as full model tests are up and running
    pytest.skip()
    
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:  
        pytest.skip()

    class SelfAttention(nn.Module):
        config: BertConfig
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            self.layer_head_mask = jnp.ones((2,))
            self.attention_mask = jax.random.randint(key, (1, 128), 0, 2)

            self.attn = FlaxBertSelfAttention(self.config, dtype=self.dtype)

        def __call__(self, hidden_state):
            act = self.attn(hidden_state, None, None)

            return act

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        compiler_config.compile_depth = CompileDepth.FULL
    else:
        # Fails on backend with following error:
        # "Backward error: element 0 of tensors does not require grad and does not have a grad_fn"
        compiler_config.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY

    # Initialize module
    config = BertConfig()
    config.hidden_size = 128
    config.num_attention_heads = 2
    config.num_hidden_layers = 2
    config.intermediate_size = 512

    hidden_state_shape = (1, 128, 128)
    key = jax.random.PRNGKey(0)
    hidden_state = jax.random.uniform(key, hidden_state_shape)

    dtype = jax.numpy.float32
    framework_module = SelfAttention(
        config=config, 
        dtype=dtype,
    )
    vars = framework_module.init(key, 
        hidden_state=hidden_state,
    )
    framework_module = framework_module.bind(vars)

    # Run module
    # res = framework_module(
    #     hidden_state=hidden_state,
    # )

    pybuda_module = JaxModule("bert_self_attention_jax", framework_module)
    verify_module(
        pybuda_module,
        (hidden_state_shape,),
        inputs=[(hidden_state,),],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={
                "attn.value.bias",
            }
        ),
    )


def test_bert_encoder(test_kind, test_device):
    # No need to instance specific module as full model tests are up and running
    pytest.skip()
    
    if test_kind.is_training():
        pytest.skip()

    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:  
        pytest.skip()

    class Wrapper(nn.Module):
        model: FlaxBertEncoder
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            self.attention_mask = jnp.ones(attention_mask_shape)
            self.head_mask = jnp.ones(head_mask_shape)

        def __call__(self, hidden_states):
            act = self.model(hidden_states, self.attention_mask, self.head_mask)

            return act

    compiler_config = _get_global_compiler_config()
    if not test_kind.is_training():
        # compiler_config.compile_depth = CompileDepth.FULL
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER # Remove when maximum decomposition is enabled
    else:
        # Fails on backend with following error:
        # "Backward error: element 0 of tensors does not require grad and does not have a grad_fn"
        # compiler_config.compile_depth = CompileDepth.BACKEND_GOLDEN_VERIFY
        compiler_config.compile_depth = CompileDepth.BUDA_GRAPH_PRE_PLACER # Remove when maximum decomposition is enabled
    compiler_config.retain_tvm_python_files = True

    # Initialize module
    key = jax.random.PRNGKey(0)

    hidden_states_shape = (1, 1, 768)
    attention_mask_shape = (1, 1)
    head_mask_shape = (1, 1)
    
    hidden_states = jax.random.uniform(key, hidden_states_shape)

    config = BertConfig()
    config.num_hidden_layers = 1

    dtype = jax.numpy.float32
    framework_module = FlaxBertEncoder(
        config=config, 
        dtype=dtype,
    )
    framework_module = Wrapper(
        model=framework_module, 
        dtype=jax.numpy.float32,
    )
    vars = framework_module.init(key, 
        hidden_states=hidden_states,
    )
    framework_module = framework_module.bind(vars)

    # Run module
    # res = framework_module(
    #     hidden_states=hidden_states,
    # )

    pybuda_module = JaxModule("bert_encoder_jax", framework_module)
    verify_module(
        pybuda_module,
        (hidden_states_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            waive_gradient_errors={
                "layer.0.attention.self.key.bias",
                "layer.1.attention.self.key.bias",
            },
        ),
        input_params=[{"requires_grad": False}],
    )


def test_bert_pooler(test_kind, test_device):    
    # Only run recompute test in post-commit
    if test_kind == TestKind.TRAINING:  
        pytest.skip()

    # Initialize module
    key = jax.random.PRNGKey(0)
    hidden_states_shape = (1, 1, 128)
    hidden_states = jax.random.uniform(key, hidden_states_shape)

    config = BertConfig()
    config.num_hidden_layers = 1

    dtype = jax.numpy.float32
    framework_module = FlaxBertPooler(
        config=config, 
        dtype=dtype,
    )
    vars = framework_module.init(key, 
        hidden_states=hidden_states,
    )
    framework_module = framework_module.bind(vars)

    # # Run module
    # res = framework_module(
    #     hidden_states=hidden_states,
    # )

    pybuda_module = JaxModule("bert_pooler_jax", framework_module)
    verify_module(
        pybuda_module,
        (hidden_states_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
        ),
    )
