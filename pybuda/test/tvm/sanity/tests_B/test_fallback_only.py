# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# This File is NOT intended for CI
#
import pytest
from collections import OrderedDict

import torch
from torch import nn
from loguru import logger

from transformers import BertModel, BertConfig, BertForPreTraining, TFBertMainLayer, TFBertForQuestionAnswering


import pybuda
from pybuda import (
    PyTorchModule,
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    pybuda_compile,
)
from pybuda.tensor import to_pt_tensors
from pybuda.op.eval import compare_tensor_to_golden
from test.tvm.utils import evaluate_framework_vs_pybuda
from pybuda.config import CompileDepth, _get_global_compiler_config
from pybuda.verify import verify_module
from pybuda.verify.config import TestKind
from test.backend.models.test_bert import get_relaxed_atol_pcc


################################### PYTORCH ###################################

from test.tvm.nlp.pytorch import test_bert as test_bert_pt
def test_bert_pt_fallback(test_kind, test_device):
    test_bert_pt.test_bert_pt_fallback(test_kind, test_device)

def test_bert_pt_embeddings_fallback(test_kind, test_device):
    test_bert_pt.test_bert_embeddings_fallback(test_kind, test_device)

def test_bert_pt_direct_fallback(test_kind, test_device):
    test_bert_pt.test_bert_direct_fallback(test_kind, test_device)

from test.tvm.nlp.pytorch import test_gpt2 as test_gpt2_pt

def test_gpt2_pt_tvm_fallback(test_kind, test_device):
    test_gpt2_pt.test_tvm_gpt2_fallback(test_kind, test_device)

from test.tvm.nlp.pytorch import test_t5_small as test_t5_small_pt

def test_t5_small_pt_fallback(test_kind, test_device):
    test_t5_small_pt.test_t5_small_fallback(test_kind, test_device)

################################### TENSORFLOW ##################################

from test.tvm.nlp.tensorflow import test_bert as test_bert_tf
def test_bert_tf_fallback(test_kind, test_device):
    test_bert_tf.test_bert_tf_fallback(test_kind, test_device)

def test_bert_qa_tf_fallback(test_kind, test_device):
    test_bert_tf.test_bert_qa_tf_fallback(test_kind, test_device)

from test.tvm.nlp.tensorflow import test_gpt2 as test_gpt2_tf
def test_gpt2_tf_fallback(test_kind, test_device):
    test_gpt2_tf.test_tvm_gpt2_fallback(test_kind, test_device)

from transformers import T5Config, TFT5Model, TFT5ForConditionalGeneration, TFT5EncoderModel
from test.tvm.nlp.tensorflow import test_t5_small_tf

def test_t5_small_tf_fallback(test_kind, test_device):
    test_t5_small_tf.test_t5_small_fallback(test_kind, test_device)

from test.tvm.nlp.tensorflow import test_opt as test_opt_tf
def test_opt_tf_fallback(test_kind, test_device):
    test_opt_tf.test_opt_fallback(test_kind, test_device)

################################### ONNX ##################################

from test.tvm.sanity import test_sanity_onnx
def test_emb_linear_onnx(test_kind, test_device):
    test_sanity_onnx.test_tvm_emb_linear_onnx_fallback(test_kind, test_device)

from test.tvm.nlp.onnx import test_t5 as test_t5_onnx
def test_t5_onnx_fallback(test_kind, test_device):
    test_t5_onnx.test_t5_small_fallback(test_kind, test_device)

from test.tvm.nlp.onnx import test_gpt2 as test_gpt2_onnx
def test_gpt2_onnx_fallback(test_kind, test_device):
    test_gpt2_onnx.test_tvm_gpt2_layer_onnx_fallback(test_kind, test_device)


from test.tvm.nlp.onnx import test_xglm as test_xglm_onnx
def test_xlgm_onnx_fallback(test_kind, test_device):
    test_xglm_onnx.test_tvm_xglm_fallback(test_kind, test_device)

from test.tvm.nlp.onnx import test_opt as test_opt_onnx
def test_opt_onnx_fallback(test_kind, test_device):
    test_opt_onnx.test_tvm_opt_fallback(test_kind, test_device)