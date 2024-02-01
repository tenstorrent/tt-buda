# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Optional, Tuple, Iterable

import torch
import numpy as np


# SPDX-FileCopyrightText: Copyright (c) 2016 Facebook, Inc
#
# SPDX-License-Identifier: Caffe2
# https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py
def adam_no_bias_correction(params: List[torch.Tensor],
         grads: List[torch.Tensor],
         exp_avgs: List[torch.Tensor],
         exp_avg_sqs: List[torch.Tensor],
         max_exp_avg_sqs: List[torch.Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         enable_adam_w: bool):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.

    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        if weight_decay != 0 and enable_adam_w:
            param.mul_(1 - lr * weight_decay)
        elif weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1)
        exp_avg.add_(grad, alpha=1 - beta1)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt()).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt()).add_(eps)

        step_size = lr

        param.addcdiv_(exp_avg, denom, value=-step_size)


class AdamNoBiasCorrection(torch.optim.Optimizer):
    """
    Implements Adam algorithm without bias correction.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, enable_adam_w=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamNoBiasCorrection, self).__init__(params, defaults)
        self.enable_adam_w = enable_adam_w

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam_no_bias_correction(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'],
                   enable_adam_w=self.enable_adam_w)
        return loss

# SPDX-FileCopyrightText: Copyright (c) 2019 cybertronai
#
# SPDX-License-Identifier: MIT
# https://github.com/cybertronai/pytorch-lamb/
class LAMB(torch.optim.Optimizer):
    """
    Implements LAMB optimization algorithm.
    This is a new layerwise adaptive large batch optimization technique.

    Args:
        params (iterable): Parameters to optimize
        lr (float): Learning rate
        betas (Tuple[float, float]): coefficients used for computing 
            mean and variance for gradient
        eps (float): coefficient used to stabilize "trust ratio"
        weight_decay (float): weight decay
        corection(bool): Correct mean and variance or not.
        clip_value(tuple[int]): Min and max value of parameters.
    """
    
    def __init__(
        self,
        params,
        lr,
        betas,
        eps,
        weight_decay,
        correction = False,
        clip_value = (0.0, 10.0)
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta 1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta 2 parameter: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if not clip_value[0] < clip_value[1]:
            raise ValueError(f"Invalid clip values, it must be clip_value < high: {clip_value}")
        if not 0.0 <= clip_value[0] < 20.0:
            raise ValueError(f"Invalid lower clip value: {clip_value[0]}")
        if not 0.0 < clip_value[1] <= 20.0:
            raise ValueError(f"Invalid higher clip value: {clip_value[1]}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

        super(LAMB, self).__init__(params, defaults)
        self.correction = correction
        self.clip_value = clip_value
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instead.')

                state = self.state[param]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mean'] = torch.zeros_like(param.data, memory_format=torch.preserve_format)
                    state['var'] = torch.zeros_like(param.data, memory_format=torch.preserve_format)

                mean, var = state['mean'], state['var']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                # Decay mean and variance
                # m(t) = m(t - 1) * beta1 + grad * (1 - beta2)
                mean *= beta1
                mean += grad * (1 - beta1)
                
                # v(t) = v(t - 1) * beta2 + grad * grad * (1 - beta2)
                var *= beta2
                var += grad * grad * (1 - beta2)
                
                learning_rate = group['lr']

                if self.correction:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    learning_rate *= np.sqrt(bias_correction2) / bias_correction1

                # L2 normaliztion of weights
                phi = param.data ** 2
                phi = torch.sum(phi)
                phi = torch.sqrt(phi)
                    
                epsilon = group['eps']
                weight_decay = group['weight_decay']

                # adam ratio, ratio of corrected mean and corrected variance stabilized with epsilon, or
                adam_ratio = mean / (torch.sqrt(var) + epsilon)
                
                if weight_decay != 0:
                    adam_ratio += param.data * weight_decay
                adam_ratio_norm = adam_ratio ** 2
                adam_ratio_norm = torch.sum(adam_ratio_norm)
                adam_ratio_norm = torch.sqrt(adam_ratio_norm)
                
                if phi != 0 and adam_ratio_norm != 0:
                    trust_ratio = phi / adam_ratio_norm
                else:
                    trust_ratio = torch.tensor([1.0])
                trust_ratio = torch.clamp(trust_ratio, self.clip_value[0], self.clip_value[1])
                    
                state['phi'] = phi
                state['adam_ratio_norm'] = adam_ratio_norm
                state['trust_ratio'] = trust_ratio
                
                param.data += adam_ratio * (-learning_rate * trust_ratio)

        return loss

# SPDX-FileCopyrightText: Copyright (c) 2019 Kakao Brain
#
# SPDX-License-Identifier: MIT
# https://github.com/kakaobrain/torchlars
class LARS(torch.optim.Optimizer):
    """
    Implements LARS optimization algorithm.
    """

    def __init__(
        self,
        params,
        lr,
        momentum,
        nesterov = False,
        dampening = 0,
        lars_coeff = 1e-3,
        weight_decay = 0,
        eps = 1e-8
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= dampening and dampening < 1.0:
            raise ValueError(f"Invalid dampening: {dampening}")
        if not 0.0 <= lars_coeff < 1.0:
            raise ValueError(f"Invalid LARS coefficient: {lars_coeff}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            lars_coeff=lars_coeff,
            eps=eps,
            weight_decay=weight_decay,
            nesterov=nesterov,
            dampening=dampening
        )

        super(LARS, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lars_coeff = group['lars_coeff']
            eps = group['eps']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for param in group['params']:
                if param.grad is None:
                    continue

                weight_norm = torch.sqrt(torch.sum(param.data ** 2))
                gradient_norm = torch.sqrt(torch.sum(param.grad.data ** 2))

                if weight_norm == 0 or gradient_norm == 0:
                    local_lr = 1
                else:
                    local_lr = lars_coeff * weight_norm / (gradient_norm + weight_decay * weight_norm + eps)

                gradient = param.grad.data
                if weight_decay != 0:
                    gradient += weight_decay * param.data

                updated_momentum = local_lr * lr * gradient

                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum' in param_state:
                        buf = param_state['momentum']
                        buf *= momentum
                        updated_momentum += buf

                param.data -= updated_momentum

        return loss
