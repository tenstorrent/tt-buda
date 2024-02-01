# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Golden verification on CPU
"""

from typing import Tuple, Dict, List

from loguru import logger
import torch
import tensorflow as tf

from ..tensor import to_pt_tensors
from ..pybudaglobal import get_devices
from ..utils import detach_tensors
import pybuda
from pybuda.tvm_utils import map_tf_dtype_to_pt, map_pt_dtype_to_tf

def cpueval_inference(
        inputs: List[Tuple[torch.Tensor, ...]], 
        parameters: List[Dict[str, torch.Tensor]],
        sequential: bool) -> List[Tuple[torch.Tensor, ...]]:
    """
    Use CPU/Pytorch to run inference of the full pipeline of devices, equivalen to what run_inference would do.
    This uses the initial graph for pybuda models, or pytorch for pytorch models. 

    Parameters
    ----------
    inputs: List[Tuple[torch.Tensor, ...]]
        Full batch of ordered inputs into the first device

    parameters: List[Dict[str, torch.Tensor]]
        Parameters, for each device in pipeline

    Returns
    -------
    List[Tuple[Tensor, ...]]
        Output of the last device in pipeline, one tuple per input
    """
    devices = get_devices()
    assert len(devices) == len(parameters), "Mismatched number of devices and parameters"

    from pybuda.run.impl import _run_command
    from pybuda.run.commands import Command

    ret = []
    for input in inputs:
        data = to_pt_tensors(input)
        for d, params in zip(devices, parameters):
            data = _run_command(d, sequential, Command.cpueval_forward(data, params, save_for_backward=False, targets=[]), response=True)["result"]
        ret.append(data)

    return ret

class TrainingEvalData:
    """
    Structure holding all golden data calculate from runnig training eval, to be compared with 
    data running on Tenstorrent device / model for verification
    """
    class GradData:
        inputs: List[torch.Tensor] # Ordered input gradients
        parameters: Dict[str, torch.Tensor] # Gradients for each parameter
        def __init__(self, inputs: List[torch.Tensor], parameters: Dict[str, torch.Tensor] = {}):
            self.inputs = inputs
            self.parameters = parameters

    class DeviceData:
        grad: List["TrainingEvalData.GradData"] # Gradients, one for each input
        final_parameters: Dict[str, torch.Tensor] # final parameters, after step of training
        def __init__(self):
            self.grad = []
            self.final_parameters = {}

    devices: List["TrainingEvalData.DeviceData"] # One set of data for each device in pipeline


def cpueval_training(
        inputs: List[Tuple[torch.Tensor, ...]], 
        parameters: List[Dict[str, torch.Tensor]], 
        targets: List[Tuple[torch.Tensor, ...]],
        sequential: bool,
        scale_loss: float,
        lr=None) -> TrainingEvalData:
    """
    Use CPU/Pytorch to run training on the full pipeline of devices, similar to what run_training would do. 
    Inputs should contain a full batch of inputs, and optimizer will run once at the end. Return values include
    all input gradients (if any), gradients for each parameter, both for each input, as well as the final
    updated weights after optimizer.

    Parameters
    ----------
    inputs: List[Tuple[Tensor, ...]]
        A full batch of ordered inputs into the first device

    parameters: List[Dict[str, Tensor]]
        Initial values for all parameters in the model, one per device. This needs to be a separate copy 
        of parameter tensors, to keep it separate from the real run which will adjust parameters through training.

    targets: List[Tuple[Tensor, ...]]
        Training targets for each input

    Returns
    -------
    TrainingEvalData
    """
    devices = get_devices()

    from pybuda.run.impl import _run_command
    from pybuda.run.commands import Command

    ret = TrainingEvalData()
    ret.devices = []
    for d in devices:
        ret.devices.append(TrainingEvalData.DeviceData())

    for i, d in enumerate(devices):
        optim = d.get_pytorch_optimizer(parameters[i], lr=lr) # Reinitialize learning rate if given
        if optim:
            optim.zero_grad()

    for input, target in zip(inputs, targets):
        data = to_pt_tensors(input, convert_format=False) # TODO: convert for silicon runs to get more accurate comparison
        fw_outputs = []
        for i, d in enumerate(devices):
            logger.trace("Running forward on {}", d)
            detached_data = detach_tensors(data)
            device_targets = target if (d == devices[-1]) else []
            data = _run_command(d, sequential, Command.cpueval_forward(detached_data, parameters[i], save_for_backward=True, targets=device_targets), response=True)["result"]
            logger.trace("Forward out on {} is {}", d, data)
            fw_outputs.append(data)

        bw_input_grads = []
        for i in reversed(range(len(devices))):
            d = devices[i]
            logger.trace("Running backward on {}", d)
            grads = _run_command(d, sequential, Command.cpueval_backward(bw_input_grads, parameters[i]), response=True)
            bw_input_grads = grads["input_grads"]
            bw_parameter_grads = grads["params_grads"]
            ret.devices[i].grad.append(TrainingEvalData.GradData(inputs=bw_input_grads, parameters=bw_parameter_grads))

    for i, d in enumerate(devices):
        if isinstance(d, pybuda.cpudevice.CPUDevice) and d.framework == "tensorflow":
            assert all([x.name in parameters[i] for x in d.modules[0].module.trainable_variables])

            cpu_grads = [tf.Variable(grad.detach().numpy(), dtype=map_pt_dtype_to_tf(grad.dtype)) for grad in ret.devices[i].grad[0].parameters.values()]
            params = [tf.Variable(param.detach().numpy(), dtype=map_pt_dtype_to_tf(param.dtype), name=name.split(":")[0]) for name,param in parameters[i].items()]
            d.optimizer.apply_gradients(zip(cpu_grads, params))

            parameters[i] = {
                param.name : torch.Tensor(param.numpy())
                for param in params
            }
        else:
            optim = d.get_pytorch_optimizer(parameters[i])
            if optim:
                optim.step()

        ret.devices[i].final_parameters = parameters[i]

    return ret

