# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from pybuda.module import PyBudaModule
import pytest

import math
import torch

import pybuda
import pybuda.op
from pybuda import (
    Tensor,
    pybuda_compile,
    CompilerConfig,
    VerifyConfig,
    TTDevice,
    AdamW,
    LARS,
    LAMB,
    SGD
)
from pybuda.verify import verify_module, VerifyConfig, TestKind
from pybuda._C.backend_api import BackendDevice, BackendType, get_output
from .common import compile, device, ModuleBuilder
from pybuda.config import _get_global_compiler_config
import numpy as np
from loguru import logger



from test.bert.modules import (
    PyBudaBertMHA,
    PyBudaBertEncoder,
    PyBudaFeedForward,
    PyBudaPredictionHeadDecoder,
    PyBudaPredictionHeadTransform,
    PyBudaFFNorm,
    get_bert_parameters
)

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

@pytest.mark.parametrize("bias_correction", (False, True), ids=['no_bias', 'bias'])
@pytest.mark.parametrize("weight_decay", (0.0, 0.99), ids=["no_weight_decay", "weight_decay"])
def test_mm_adam_optimizer(bias_correction, weight_decay):
    
    shape = (1, 1, 64, 64)
    def single_matmul(act, weights=None):
        assert weights
        return pybuda.op.Matmul("matmul1", act, weights)

    torch_weights = torch.rand(*shape, requires_grad=True)
    weights = pybuda.Parameter.create_from_torch(torch_weights)
    module = ModuleBuilder(single_matmul, weights=weights)
    module.set_parameter("weights", torch_weights)

    fp32_fallback = pybuda.DataFormat.Float16_b
    if bias_correction:
        # Requires higher precision to pass
        # pybuda.config.set_configuration_options(accumulate_df=pybuda.DataFormat.Float32)
        pytest.skip("Data mismatch issue.")

    verify_module(module,
            [shape],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                devtype=BackendType.Golden,
                accumulation_steps=1,
                steps=2,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "adam",
                    "params": {
                        "learning_rate": 0.5,
                        "weight_decay": weight_decay,
                        "bias_correction": bias_correction
                    }
                },
            ),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
    )

@pytest.mark.parametrize("bias_correction", (False, True), ids=['no_bias', 'bias'])
def test_mm_adam_consteval_optimizer(bias_correction):
    
    shape = (128, 512)
    def single_matmul(act, weights=None):
        assert weights
        transposed_weights = pybuda.op.Transpose("transposed_weights", weights, dim0=-2, dim1=-1)
        return pybuda.op.Matmul("matmul1", act, transposed_weights)

    torch_weights = torch.rand(*shape, requires_grad=True)
    weights = pybuda.Parameter.create_from_torch(torch_weights)
    module = ModuleBuilder(single_matmul, weights=weights)
    module.set_parameter("weights", torch_weights)

    fp32_fallback = pybuda.DataFormat.Float16_b
    if bias_correction:
        # Requires higher precision to pass
        # pybuda.config.set_configuration_options(accumulate_df=pybuda.DataFormat.Float32)
        pytest.skip("Data mismatch issue.")

    verify_module(module,
            [(1, *shape)],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                devtype=BackendType.Golden,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "adam",
                    "params": {
                        "learning_rate": 0.5,
                        "bias_correction": bias_correction,
                    }
                },
            ),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
    )


@pytest.mark.parametrize("bias_correction", (False, True), ids=['no_bias', 'bias'])
@pytest.mark.parametrize("weight_decay", (0.0, 0.99), ids=["no_weight_decay", "weight_decay"])
def test_mm_double_adam_optimizer(bias_correction, weight_decay):

    shape = (1, 1, 64, 64)
    def double_matmul(act, *, weights1=None, weights2=None):
        m1 = pybuda.op.Matmul("matmul1", act, weights1)
        m2 = pybuda.op.Matmul("matmul2", act, weights2)
        return pybuda.op.Add("add", m1, m2)

    torch_weights1 = torch.rand(*shape, requires_grad=True)
    torch_weights2 = torch.rand(*shape, requires_grad=True)
    weights1 = pybuda.Parameter.create_from_torch(torch_weights1)
    weights2 = pybuda.Parameter.create_from_torch(torch_weights2)
    module = ModuleBuilder(double_matmul, weights1=weights1, weights2=weights2)

    module.set_parameter("weights1", torch_weights1)
    module.set_parameter("weights2", torch_weights2)

    fp32_fallback = pybuda.DataFormat.Float16_b
    if bias_correction:
        # Requires higher precision to pass
        # pybuda.config.set_configuration_options(accumulate_df=pybuda.DataFormat.Float32)
        pytest.skip("Data mismatch issue.")

    verify_module(module,
            [shape],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                devtype=BackendType.Golden,
                accumulation_steps=1,
                steps=3,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "adam",
                    "params": {
                        "learning_rate": 0.5,
                        "weight_decay": weight_decay,
                        "bias_correction": bias_correction
                    }
                },
            ),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
    )


@pytest.mark.parametrize("cfg", [(128, 4, 128),])
@pytest.mark.parametrize("weight_decay", (0.0, 0.99), ids=["no_weight_decay", "weight_decay"])
@pytest.mark.parametrize("bias_correction", (False, True), ids=['no_bias', 'bias'])
def test_mha(cfg, weight_decay, bias_correction):

    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    microbatch_size = 1


    params = get_bert_parameters("mha", hidden_dim=hidden_dim)
    config =  { "num_heads": num_heads, "encoder_index": 0 }
    mod = PyBudaBertMHA("mha", params, config)

    fp32_fallback = pybuda.DataFormat.Float16_b
    if bias_correction:
        # Requires higher precision to pass
        # pybuda.config.set_configuration_options(accumulate_df=pybuda.DataFormat.Float32)
        pytest.skip("Data mismatch issue.")

    verify_module(mod, [(microbatch_size, seq_len, hidden_dim), (microbatch_size, 1, seq_len)],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                devtype=BackendType.Golden,
                accumulation_steps=1,
                steps=1,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "adam",
                    "params": {
                        "learning_rate": 0.5,
                        "weight_decay": weight_decay,
                        "bias_correction": bias_correction
                    }
                },
                waive_gradient_errors={"mha.bert.encoder.layer.0.attention.self.key.bias"},
            ),
            input_params=[{}, {"requires_grad": False}],
            scale_params=100,
    )

@pytest.mark.parametrize("cfg", [(20, 0.1, .5, .1),])
@pytest.mark.parametrize("bias_correction", (False, True), ids=['no_bias', 'bias'])
def test_learning_rate_scheduler_with_linear_warmup_and_decay(cfg, bias_correction):
    pytest.skip("Data mismatch issue")

    # MHA configs
    num_steps = cfg[0]
    learning_rate_warmup = cfg[1]
    learning_rate_decay = 1.0 - learning_rate_warmup
    max_learning_rate = cfg[2]
    min_learning_rate = cfg[3]

    shape = (1, 1, 64, 64)

    fp32_fallback = pybuda.DataFormat.Float16_b

    def single_matmul(act, weights=None):
        assert weights
        return pybuda.op.Matmul("matmul1", act, weights)

    torch_weights = torch.rand(*shape, requires_grad=True)
    weights = pybuda.Parameter.create_from_torch(torch_weights)
    module = ModuleBuilder(single_matmul, weights=weights)
    module.set_parameter("weights", torch_weights)

    # Requires higher precision to pass
    pybuda.config.set_configuration_options(accumulate_df=pybuda.DataFormat.Float32)

    def scheduler_iterable():
        warmup_slope = (max_learning_rate - min_learning_rate) / (learning_rate_warmup * num_steps)
        decay_slope = (min_learning_rate - max_learning_rate) / (learning_rate_decay * num_steps)

        cur_lr = min_learning_rate

        while cur_lr < max_learning_rate:
            yield cur_lr

            cur_lr += warmup_slope

        while cur_lr > min_learning_rate:
            yield cur_lr

            cur_lr += decay_slope

    class TorchSchedulerWithWarmupAndDecay(pybuda.torch_schedulers.TorchLearningRateScheduler):
        def __init__(self, optimizer):
            super().__init__(optimizer)
            self.get_lr_iterable = scheduler_iterable()

        def get_lr(self):
            return [next(self.get_lr_iterable)]

    class LearningRateSchedulerWithWarmupAndDecay(pybuda.schedulers.LearningRateScheduler):
        def __init__(self, optimizer):
            super().__init__(optimizer)
            self.get_lr_iterable = scheduler_iterable()

        def get_lr(self):
            return next(self.get_lr_iterable)
        
        def get_scheduler_params(self, name, is_buda):
            opt_params = self.optimizer.get_optimizer_params(name, is_buda=is_buda)
            return {'lr': opt_params['lr']}

        def get_pytorch_scheduler(self, optimizer: torch.optim.Optimizer):
            if self.torch_scheduler is None:
                self.torch_scheduler = TorchSchedulerWithWarmupAndDecay(
                    optimizer=optimizer
                )

            return self.torch_scheduler

    verify_module(module, [shape],
        VerifyConfig(
            test_kind=TestKind.TRAINING,
            devtype=BackendType.Golden,
            accumulation_steps=1,
            steps=2,
            epochs=20,
            fp32_fallback=fp32_fallback,
            optimizer={
                "type": "adam",
                "params": {
                    "learning_rate": min_learning_rate,
                    "weight_decay": 0.0,
                }
            },
            scheduler={
                "type": LearningRateSchedulerWithWarmupAndDecay,
                "params": {

                }
            },
            waive_gradient_errors={"mha.bert.encoder.layer.0.attention.self.key.bias"},
        ),
        input_params=[{}, {"requires_grad": False}],
    )


# ==========================================================================
# SIMPLE TEST MODELS, TEST FUNCTIONS
# ==========================================================================

from torch.distributions import Normal, Uniform

DISTRIBUTION = Uniform
DISTRIBUTION_MIN = 0.0
DISTRIBUTION_MAX = 1.0


def matmul(shape):

    def matmul_inner(activations, weights):
        return pybuda.op.Matmul("mm", activations, weights)

    torch_weights = DISTRIBUTION(DISTRIBUTION_MIN, DISTRIBUTION_MAX).sample(shape)
    torch_weights.requires_grad = True
    weights = pybuda.Parameter.create_from_torch(torch_weights)
    module = ModuleBuilder(matmul_inner, weights=weights)
    module.set_parameter("weights", torch_weights)

    return module


def ff_2(shape):

    def ff_2_inner(activations, weights1, weights2):
        """ Feed-Forward Neural Net, 2 Layers"""
        l1 = pybuda.op.Matmul("mm1", activations, weights1)
        act1 = pybuda.op.Gelu("gelu1", l1)
        l2 = pybuda.op.Matmul("mm2", act1, weights2)
        return l2
        
    torch_weights1 = DISTRIBUTION(DISTRIBUTION_MIN, DISTRIBUTION_MAX).sample(shape)
    torch_weights1.requires_grad = True
    torch_weights2 = DISTRIBUTION(DISTRIBUTION_MIN, DISTRIBUTION_MAX).sample(shape)
    torch_weights2.requires_grad = True
    
    weights1 = pybuda.Parameter.create_from_torch(torch_weights1)
    weights2 = pybuda.Parameter.create_from_torch(torch_weights2)
    
    module = ModuleBuilder(ff_2_inner, weights1=weights1, weights2=weights2)
    module.set_parameter("weights1", torch_weights1)
    module.set_parameter("weights2", torch_weights2)

    return module

def ff_5(shape):

    def ff_5_inner(activations, weights1, weights2, weights3, weights4, weights5):
        """ Feed-Forward Neural Net, 5 Layers"""

        l1 = pybuda.op.Matmul("mm1", activations, weights1)
        act1 = pybuda.op.Gelu("gelu1", l1)
        
        l2 = pybuda.op.Matmul("mm2", act1, weights2)
        act2 = pybuda.op.Gelu("gelu2", l2)
        
        l3 = pybuda.op.Matmul("mm3", act2, weights3)
        act3 = pybuda.op.Relu("gelu3", l3)

        l4 = pybuda.op.Matmul("mm4", act3, weights4)
        act4 = pybuda.op.Gelu("gelu4", l4)

        l5 = pybuda.op.Matmul("mm5", act4, weights5)
        
        return l5
        
    torch_weights1 = DISTRIBUTION(DISTRIBUTION_MIN, DISTRIBUTION_MAX).sample(shape)
    torch_weights1.requires_grad = True
    torch_weights2 = DISTRIBUTION(DISTRIBUTION_MIN, DISTRIBUTION_MAX).sample(shape)
    torch_weights2.requires_grad = True
    torch_weights3 = DISTRIBUTION(DISTRIBUTION_MIN, DISTRIBUTION_MAX).sample(shape)
    torch_weights3.requires_grad = True
    torch_weights4 = DISTRIBUTION(DISTRIBUTION_MIN, DISTRIBUTION_MAX).sample(shape)
    torch_weights4.requires_grad = True
    torch_weights5 = DISTRIBUTION(DISTRIBUTION_MIN, DISTRIBUTION_MAX).sample(shape)
    torch_weights5.requires_grad = True
    
    weights1 = pybuda.Parameter.create_from_torch(torch_weights1)
    weights2 = pybuda.Parameter.create_from_torch(torch_weights2)
    weights3 = pybuda.Parameter.create_from_torch(torch_weights3)
    weights4 = pybuda.Parameter.create_from_torch(torch_weights4)
    weights5 = pybuda.Parameter.create_from_torch(torch_weights5)
    
    module = ModuleBuilder(
        ff_5_inner, 
        weights1=weights1, 
        weights2=weights2,
        weights3=weights3,
        weights4=weights4,
        weights5=weights5
    )
    module.set_parameter("weights1", torch_weights1)
    module.set_parameter("weights2", torch_weights2)
    module.set_parameter("weights3", torch_weights3)
    module.set_parameter("weights4", torch_weights4)
    module.set_parameter("weights5", torch_weights5)

    return module

# ==========================================================================
# TEST PARAMETERS/CONSTANTS FOR AdamW OPTIMIZER
# ==========================================================================

ADAMW_EPOCHS = [1]
ADAMW_ACCUMULATION_STEPS = [1]
ADAMW_STEPS = [2]
ADAMW_ROUND = 2

ADAMW_LR_NO = 1
ADAMW_LR_MIN = 1e-4
ADAMW_LR_MAX = 1e-2
ADAMW_LR_ROUND = 4

ADAMW_BETA_1_NO = 1
ADAMW_BETA_1_MIN = 0.8
ADAMW_BETA_1_MAX = 0.95
ADAMW_BETA_2_NO = 1
ADAMW_BETA_2_MIN = 0.99
ADAMW_BETA_2_MAX = 0.999

ADAMW_EPSILON_NO = 1
ADAMW_EPSILON_MIN = 1e-9
ADAMW_EPSILON_MAX = 1e-7
ADAMW_EPSILON_ROUND = 8

ADAMW_WEIGHT_DECAY_NO = 1
ADAMW_WEIGHT_DECAY_MIN = 1e-3  
ADAMW_WEIGHT_DECAY_MAX = 1e-1
ADAMW_WEIGHT_DECAY_ROUND = 3

np.random.seed(1)

adamw_learning_rate = [np.round(np.random.rand() * (ADAMW_LR_MAX - ADAMW_LR_MIN) + ADAMW_LR_MIN, ADAMW_LR_ROUND) for _ in range(ADAMW_LR_NO)] 
adamw_beta1 = [np.round(np.random.rand() * (ADAMW_BETA_1_MAX - ADAMW_BETA_1_MIN) + ADAMW_BETA_1_MIN, ADAMW_ROUND) for _ in range(ADAMW_BETA_1_NO)]
adamw_beta2 = [np.round(np.random.rand() * (ADAMW_BETA_2_MAX - ADAMW_BETA_2_MIN) + ADAMW_BETA_2_MIN, ADAMW_ROUND) for _ in range(ADAMW_BETA_2_NO)]
adamw_epsilon = [np.round(np.random.rand() * (ADAMW_EPSILON_MAX - ADAMW_EPSILON_MIN) + ADAMW_EPSILON_MIN, ADAMW_EPSILON_ROUND) for _ in range(ADAMW_EPSILON_NO)]
adamw_weight_decay = [np.round(np.random.rand() * (ADAMW_WEIGHT_DECAY_MAX - ADAMW_WEIGHT_DECAY_MIN) + ADAMW_WEIGHT_DECAY_MIN, ADAMW_WEIGHT_DECAY_ROUND) for _ in range(ADAMW_WEIGHT_DECAY_NO)]


@pytest.mark.parametrize("epochs", ADAMW_EPOCHS, ids=[f"epochs={item}" for item in ADAMW_EPOCHS])
@pytest.mark.parametrize("steps", ADAMW_STEPS, ids=[f"steps={item}" for item in ADAMW_STEPS])
@pytest.mark.parametrize("accumulation_steps", ADAMW_ACCUMULATION_STEPS, ids=[f"acc_steps={item}" for item in ADAMW_ACCUMULATION_STEPS])
@pytest.mark.parametrize("weight_decay", adamw_weight_decay, ids=[f"weight_decay={item}" for item in adamw_weight_decay])
@pytest.mark.parametrize("epsilon", adamw_epsilon, ids=[f"epsilon={item}" for item in adamw_epsilon])
@pytest.mark.parametrize("beta2", adamw_beta2, ids=[f"beta2={item}" for item in adamw_beta2])
@pytest.mark.parametrize("beta1", adamw_beta1, ids=[f"beta1={item}" for item in adamw_beta1])
@pytest.mark.parametrize("learning_rate", adamw_learning_rate, ids=[f"lr={item}" for item in adamw_learning_rate])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 64, 64)], ids=["shape=1x1x32x32", "shape=1x1x64x64"])
@pytest.mark.parametrize("model", [matmul, ff_2, ff_5], ids=[])
def test_adamw_optimizer(
    test_device,
    model,
    shape,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    weight_decay,
    epochs,
    accumulation_steps,
    steps
):

    module = model(shape)

    fp32_fallback = pybuda.DataFormat.Float16_b

    verify_module(module,
            [shape],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                # devtype=BackendType.Golden,
                devtype=test_device.devtype,
                arch=test_device.arch,
                epochs=epochs,
                accumulation_steps=accumulation_steps,
                steps=steps,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "adamw",
                    "params": {
                        "learning_rate": learning_rate,
                        "beta1": beta1,
                        "beta2": beta2,
                        "eps": epsilon,
                        "weight_decay": weight_decay
                    }
                },
            ),
            input_params=[{}, {"requires_grad": True}],
            scale_params=1,
    )


# ==========================================================================
# TEST PARAMETERS/CONSTANTS FOR LAMB OPTIMIZER
# ==========================================================================

LAMB_EPOCHS = [1]
LAMB_ACCUMULATION_STEPS = [1]
LAMB_STEPS = [2]
LAMB_ROUND = 2

LAMB_LR_NO = 1
LAMB_LR_ROUND = 8
LAMB_LR_MIN = 1e-2
LAMB_LR_MAX = 1e-1

LAMB_BETA_1_NO = 1
LAMB_BETA_1_MIN = 0.8
LAMB_BETA_1_MAX = 0.95
LAMB_BETA_2_NO = 1
LAMB_BETA_2_MIN = 0.99
LAMB_BETA_2_MAX = 0.999

LAMB_EPSILON_NO = 1
LAMB_EPSILON_ROUND = 8
LAMB_EPSILON_MIN = 1e-9
LAMB_EPSILON_MAX = 1e-7

LAMB_WEIGHT_DECAY_NO = 1
LAMB_WEIGHT_DECAY_MIN = 0.01
LAMB_WEIGHT_DECAY_MAX = 0.015

LAMB_CLIP_NO = 1
LAMB_CLIP_MIN = 0.0
LAMB_CLIP_MAX = 10.0


lamb_weight_decay = [np.round(np.random.rand() * (LAMB_WEIGHT_DECAY_MAX - LAMB_WEIGHT_DECAY_MIN) + LAMB_WEIGHT_DECAY_MIN, LAMB_ROUND) for _ in range(LAMB_WEIGHT_DECAY_NO)]
lamb_epsilon = [np.round(np.random.rand() * (LAMB_EPSILON_MAX - LAMB_EPSILON_MIN) + LAMB_EPSILON_MIN, LAMB_EPSILON_ROUND) for _ in range(LAMB_EPSILON_NO)]
lamb_beta2 = [np.round(np.random.rand() * (LAMB_BETA_2_MAX - LAMB_BETA_2_MIN) + LAMB_BETA_2_MIN, LAMB_ROUND) for _ in range(LAMB_BETA_2_NO)]
# lamb_beta1 = [np.round(np.random.rand() * (LAMB_BETA_1_MAX - LAMB_BETA_1_MIN) + LAMB_BETA_1_MIN, LAMB_ROUND) for _ in range(LAMB_BETA_1_NO)]
lamb_beta1 = [0.999]
# lamb_learning_rate = [np.round(np.random.rand() * (LAMB_LR_MAX - LAMB_LR_MIN) + LAMB_LR_MIN, LAMB_LR_ROUND) for _ in range(LAMB_LR_NO)]
lamb_learning_rate = [0.001]
# lamb_clip_value = [np.round(np.sort(np.random.rand(2, ) * (LAMB_CLIP_MAX - LAMB_CLIP_MIN) + LAMB_CLIP_MIN), LAMB_ROUND) for _ in range(LAMB_CLIP_NO)]
lamb_clip_value = [(0.0, 10.0)]


@pytest.mark.parametrize("epochs", LAMB_EPOCHS, ids=[f"epochs={item}" for item in LAMB_EPOCHS])
@pytest.mark.parametrize("accumulation_steps", LAMB_ACCUMULATION_STEPS, ids=[f"acc_steps={item}" for item in LAMB_ACCUMULATION_STEPS])
@pytest.mark.parametrize("steps", LAMB_STEPS, ids=[f"steps={item}" for item in LAMB_STEPS])
@pytest.mark.parametrize("clip_value", lamb_clip_value, ids=[f"clip={item[0]}_{item[1]}" for item in lamb_clip_value])
@pytest.mark.parametrize("correction", [True, False], ids=["Correction", "NoCorrection"])
@pytest.mark.parametrize("weight_decay", lamb_weight_decay, ids=[f"weight_decay={item}" for item in lamb_weight_decay])
@pytest.mark.parametrize("epsilon", lamb_epsilon, ids=[f"epsilon={item}" for item in lamb_epsilon])
@pytest.mark.parametrize("beta2", lamb_beta2, ids=[f"beta1={item}" for item in lamb_beta1])
@pytest.mark.parametrize("beta1", lamb_beta1, ids=[f"beta2={item}" for item in lamb_beta2])
@pytest.mark.parametrize("learning_rate", lamb_learning_rate, ids=[f"lr={item}" for item in lamb_learning_rate])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 64, 64)], ids=["shape=1x1x32x32", "shape=1x1x64x64"])
@pytest.mark.parametrize("model", [matmul, ff_2, ff_5], ids=[])
def test_lamb_optimizer(
    test_device,
    model,
    shape,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    weight_decay,
    correction,
    clip_value,
    epochs,
    accumulation_steps,
    steps
):
    #Fusing disabled due to tenstorrent/pybuda#548
    pybuda.set_configuration_options(enable_auto_fusing=False)

    module = model(shape)

    fp32_fallback = pybuda.DataFormat.Float16_b

    verify_module(module,
            [shape],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                # devtype=BackendType.Golden,
                devtype=test_device.devtype,
                arch=test_device.arch,
                epochs=epochs,
                accumulation_steps=accumulation_steps,
                steps=steps,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "lamb",
                    "params": {
                        "learning_rate": learning_rate,
                        "beta1": beta1,
                        "beta2": beta2,
                        "eps": epsilon,
                        "weight_decay": weight_decay,
                        "correction": correction,
                        "clip_value": clip_value
                    }
                },
            ),
            input_params=[{}, {"requires_grad": True}],
            scale_params=1,
    )



# ==========================================================================
# TEST PARAMETERS/CONSTANTS FOR LARS OPTIMIZER
# ==========================================================================

LARS_EPOCHS = [1]
LARS_ACCUMULATION_STEPS = [1]
LARS_STEPS = [2]
LARS_ROUND = 2

LARS_LR_NO = 1
LARS_LR_ROUND = 5
LARS_LR_MIN = 1e-4
LARS_LR_MAX = 1e-2

LARS_MOMENUTM_NO = 1
LARS_MOMENTUM_MIN = 0.8
LARS_MOMENUTM_MAX = 0.95

LARS_WEIGHT_DECAY_NO = 1
LARS_WEIGHT_DECAY_MIN = 1e-3
LARS_WEIGHT_DECAY_MAX = 1e-2

LARS_EPSILON_NO = 1
LARS_EPSILON_ROUND = 16
LARS_EPSILON_MIN = 1e-16
LARS_EPSILON_MAX = 1e-15

LARS_COEFF_NO = 1
LARS_COEFF_ROUND = 5
LARS_COEFF_MIN = 0.001  #1e-4
LARS_COEFF_MAX = 0.0015 #1e-2

lars_coeff = [np.round(np.random.rand() * (LARS_COEFF_MAX - LARS_COEFF_MIN) + LARS_COEFF_MIN, LARS_COEFF_ROUND) for _ in range(LARS_COEFF_NO)]
lars_momentum = [np.round(np.random.rand() * (LARS_MOMENUTM_MAX - LARS_MOMENTUM_MIN) + LARS_MOMENTUM_MIN, LARS_ROUND) for _ in range(LARS_MOMENUTM_NO)]
lars_learning_rate = [np.round(np.random.rand() * (LARS_LR_MAX - LARS_LR_MIN) + LARS_LR_MIN, LARS_LR_ROUND) for _ in range(LARS_LR_NO)]
lars_weight_decay = [np.round(np.random.rand() * (LARS_WEIGHT_DECAY_MAX - LARS_WEIGHT_DECAY_MIN) + LARS_WEIGHT_DECAY_MIN, LARS_ROUND) for _ in range(LARS_WEIGHT_DECAY_NO)]
lars_epsilon = [np.round(np.random.rand() * (LARS_EPSILON_MAX - LARS_EPSILON_MIN) + LARS_EPSILON_MIN, LARS_EPSILON_ROUND) for _ in range(LARS_EPSILON_NO)]


@pytest.mark.parametrize("epochs", LARS_EPOCHS, ids=[f"epochs={item}" for item in LARS_EPOCHS])
@pytest.mark.parametrize("accumulation_steps", LARS_ACCUMULATION_STEPS, ids=[f"acc_steps={item}" for item in LARS_ACCUMULATION_STEPS])
@pytest.mark.parametrize("steps", LARS_STEPS, ids=[f"steps={item}" for item in LARS_STEPS])
@pytest.mark.parametrize("lars_coeff", lars_coeff, ids=[f"lars_coeff={item}" for item in lars_coeff])
@pytest.mark.parametrize("weight_decay", lars_weight_decay, ids=[f"weight_decay={item}" for item in lars_weight_decay])
@pytest.mark.parametrize("momentum", lars_momentum, ids=[f"momentum={item}" for item in lars_momentum])
@pytest.mark.parametrize("epsilon", lars_epsilon, ids=[f"epsilon={item}" for item in lars_epsilon])
@pytest.mark.parametrize("learning_rate", lars_learning_rate, ids=[f"lr={item}" for item in lars_learning_rate])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 64, 64)], ids=["shape=1x1x32x32", "shape=1x1x64x64"])
@pytest.mark.parametrize("model", [matmul, ff_2, ff_5], ids=[])
def test_lars_optimizer(
    test_device,
    model,
    shape,
    learning_rate,
    momentum,
    weight_decay,
    lars_coeff,
    epsilon,
    epochs,
    accumulation_steps,
    steps
):

    #Fusing disabled due to tenstorrent/pybuda#548
    pybuda.set_configuration_options(enable_auto_fusing=False)

    module = model(shape)

    fp32_fallback = pybuda.DataFormat.Float16_b

    verify_module(module,
            [shape],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                # devtype=BackendType.Golden,
                devtype=test_device.devtype,
                arch=test_device.arch,
                epochs=epochs,
                fp32_fallback=fp32_fallback,
                accumulation_steps=accumulation_steps,
                steps=steps,
                optimizer={
                    "type": "lars",
                    "params": {
                        "learning_rate": learning_rate,
                        "momentum": momentum,
                        "lars_coeff": lars_coeff,
                        "weight_decay": weight_decay,
                        "epsilon": epsilon
                    }
                },
                waive_gradient_errors={},
            ),
            input_params=[{}, {"requires_grad": True}],
            scale_params=1,
    )


# ==========================================================================
# MORE COMPLEX TEST MODELS/MODULES
# ==========================================================================

@pytest.mark.parametrize("weight_decay", adamw_weight_decay, ids=[f"weight_decay={item}" for item in adamw_weight_decay])
@pytest.mark.parametrize("epsilon", adamw_epsilon, ids=[f"epsilon={item}" for item in adamw_epsilon])
@pytest.mark.parametrize("beta2", adamw_beta2, ids=[f"beta2={item}" for item in adamw_beta2])
@pytest.mark.parametrize("beta1", adamw_beta1, ids=[f"beta1={item}" for item in adamw_beta1])
@pytest.mark.parametrize("learning_rate", [0.01])
@pytest.mark.parametrize("cfg", [(64, 1, 64)])
def test_mha_adamw(
    test_device,
    cfg, 
    weight_decay, 
    learning_rate,
    beta1,
    beta2,
    epsilon
):

    #Fusing disabled due to tenstorrent/pybuda#548
    pybuda.set_configuration_options(enable_auto_fusing=False)

    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    microbatch_size = 1


    params = get_bert_parameters("mha", hidden_dim=hidden_dim)
    config =  { "num_heads": num_heads, "encoder_index": 0 }
    mod = PyBudaBertMHA("mha", params, config)

    fp32_fallback = pybuda.DataFormat.Float16_b

    verify_module(mod, [(microbatch_size, seq_len, hidden_dim), (microbatch_size, 1, seq_len)],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                # devtype=BackendType.Golden,
                devtype=test_device.devtype,
                arch=test_device.arch,
                accumulation_steps=1,
                steps=1,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "adamw",
                    "params": {
                        "learning_rate": learning_rate,
                        "beta1": beta1,
                        "beta2": beta2,
                        "eps": epsilon,
                        "weight_decay": weight_decay
                    }
                },
                waive_gradient_errors={"mha.bert.encoder.layer.0.attention.self.key.bias"},
            ),
            input_params=[{}, {"requires_grad": False}],
            # scale_params=100,
    )


@pytest.mark.parametrize("epsilon", lamb_epsilon, ids=[f"epsilon={item}" for item in lamb_epsilon])
@pytest.mark.parametrize("beta2", [0.999])
@pytest.mark.parametrize("beta1", [0.9])
@pytest.mark.parametrize("learning_rate", [0.001])
@pytest.mark.parametrize("cfg", [(64, 1, 64)])
@pytest.mark.parametrize("weight_decay", (0.0, 0.99), ids=["no_weight_decay", "weight_decay"])
def test_mha_lamb(
    test_device,
    cfg, 
    weight_decay, 
    learning_rate,
    beta1,
    beta2,
    epsilon
):

    #Fusing disabled due to tenstorrent/pybuda#548
    pybuda.set_configuration_options(enable_auto_fusing=False)

    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    microbatch_size = 1


    params = get_bert_parameters("mha", hidden_dim=hidden_dim)
    config =  { "num_heads": num_heads, "encoder_index": 0 }
    mod = PyBudaBertMHA("mha", params, config)

    fp32_fallback = pybuda.DataFormat.Float16_b

    verify_module(mod, [(microbatch_size, seq_len, hidden_dim), (microbatch_size, 1, seq_len)],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                # devtype=BackendType.Golden,
                devtype=test_device.devtype,
                arch=test_device.arch,
                accumulation_steps=1,
                steps=1,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "lamb",
                    "params": {
                        "learning_rate": learning_rate,
                        "beta1": beta1,
                        "beta2": beta2,
                        "eps": epsilon,
                        "weight_decay": weight_decay,
                        # "correction": correction,
                        # "clip_value": clip_value
                    }
                },
                waive_gradient_errors={"mha.bert.encoder.layer.0.attention.self.key.bias"},
            ),
            input_params=[{}, {"requires_grad": False}],
            # scale_params=100,
    )

@pytest.mark.parametrize("lars_coeff", lars_coeff, ids=[f"lars_coeff={item}" for item in lars_coeff])
@pytest.mark.parametrize("momentum", lars_momentum, ids=[f"momentum={item}" for item in lars_momentum])
@pytest.mark.parametrize("epsilon", lars_epsilon, ids=[f"epsilon={item}" for item in lamb_epsilon])
@pytest.mark.parametrize("learning_rate", [0.001])
@pytest.mark.parametrize("cfg", [(64, 1, 64)])
@pytest.mark.parametrize("weight_decay", [0.0], ids=["no_weight_decay"])
def test_mha_lars(
    test_device,
    cfg, 
    weight_decay, 
    learning_rate,
    epsilon,
    lars_coeff,
    momentum
):

    #Fusing disabled due to tenstorrent/pybuda#548
    pybuda.set_configuration_options(enable_auto_fusing=False)

    hidden_dim = cfg[0]
    num_heads = cfg[1]
    seq_len = cfg[2]

    microbatch_size = 1


    params = get_bert_parameters("mha", hidden_dim=hidden_dim)
    config =  { "num_heads": num_heads, "encoder_index": 0 }
    mod = PyBudaBertMHA("mha", params, config)

    fp32_fallback = pybuda.DataFormat.Float16_b

    verify_module(mod, [(microbatch_size, seq_len, hidden_dim), (microbatch_size, 1, seq_len)],
            VerifyConfig(
                test_kind=TestKind.TRAINING,
                # devtype=BackendType.Golden,
                devtype=test_device.devtype,
                arch=test_device.arch,
                accumulation_steps=1,
                steps=1,
                fp32_fallback=fp32_fallback,
                optimizer={
                    "type": "lars",
                    "params": {
                        "learning_rate": learning_rate,
                        "momentum": momentum,
                        "lars_coeff": lars_coeff,
                        "weight_decay": weight_decay,
                        "epsilon": epsilon
                    }
                },
                waive_gradient_errors={"mha.bert.encoder.layer.0.attention.self.key.bias"},
            ),
            input_params=[{}, {"requires_grad": False}],
            # scale_params=100,
    )
