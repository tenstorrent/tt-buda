# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Optional, Union, Dict
import queue

import torch
import torch.multiprocessing as mp

from ..module import PyBudaModule
from ..tensor import Tensor
from ..verify import VerifyConfig

from pybuda._C import DataFormat
from pybuda._C.backend_api import DeviceMode

from ..pybudaglobal import get_devices
from .impl import (
    _run_forward,
    _initialize_pipeline,
    _run_inference,
    _run_devices_training,
    _run_generate,
    _shutdown,
    _run_backward,
    _run_optimizer,
    _run_schedulers,
    _save_parameter_checkpoint,
    _get_parameter_checkpoint,
    _get_parameter_gradients,
    _update_device_parameters,
    _error_raised,
    _get_checkpoint_queue,
    _get_loss_queue,
    _get_intermediates_queue,
    _sync,
    _detect_available_devices,
)


def run_inference(
        module: Optional[PyBudaModule] = None,
        inputs: List[Union[Tuple[Union[torch.Tensor, Tensor], ...], Dict[str, Union[torch.Tensor, Tensor]]]] = [],
        input_count: int = 1,
        output_queue: queue.Queue = None,
        _sequential: bool = False,
        _perf_trace: bool = False,
        _verify_cfg: Optional[VerifyConfig] = None) -> queue.Queue:
    """
    Main "run" function for inference. After all modules have been defined and placed on devices, this will 
    execute the workload. Unless 'sequential' is set, the function will return as soon as the devices are set up
    to run, and inference will run as long as new inputs are pushed into the device(s). If sequential mode is on,
    the function will run through inputs that are already in the input buffer and return when done.

    Parameters
    ----------
    module: PyBudaModule, optional
        If provided, place given module on a TT Device and run inference. Alternatively, manually create device(s) and
        placed module(s) on them.

    inputs: List[Union[Tuple[Union[torch.Tensor, Tensor], ...], Dict[str, Union[torch.Tensor, Tensor]]]], optional
        An optional list of input tensor tuples or dictionaries (passed as args or kwargs to module), to feed into the inference pipeline. 
        Alternatively, use `device.push_to_inputs` to manually provide inputs outside of this call.

    input_count: int, default=1
        The number of inputs to run inference on. If 0, inference will run "forever", until `shutdown` or `run_inference`
        is called again.

    output_queue: queue.Queue, optional
        If provided, outputs will be pushed into the queue as they are calculated. Otherwise, one will be created
        and returned.

    _sequential: bool, Internal
        Don't use.

    _perf_trace: bool, Internal
        Don't use.

    _verify_cfg: Internal
        Don't use.

    Returns
    -------
    queue.Queue
        Queue holding the output results. Either the output_queue provided, or one that's created.

    """

    return _run_inference(module, inputs, input_count, output_queue, _sequential, _perf_trace, _verify_cfg)

def run_training(
        epochs: int = 1,
        steps: int = 1,
        accumulation_steps: int = 1,
        microbatch_count: int = 1,
        checkpoint_queue: queue.Queue = None,
        loss_queue: queue.Queue = None,
        checkpoint_interval: int = 0,
        _sequential: bool = False,
        _perf_trace: bool = False,
        _verify_cfg: Optional[VerifyConfig] = None) -> queue.Queue:
        
    """
    Main "run" function for training. After all modules have been defined and placed on devices, this will 
    execute the workload.

    Parameters
    ----------
    epochs: int
        The number of epoch to run. Scheduler, if provided, will be stepped after each one.

    steps: int
        The number of batches to run. After every step, the optimizer will be stepped.

    accumulation_steps: int
        The number of mini-batches in a batch. Each mini-batch is limited in size by how much of the
        intermediate data can fit in device memory. 

    microbatch_count: int
        Each mini-batch is optionally further broken into micro-batches. This is necessary to fill a 
        multi-device pipeline, and should be roughly 4-6x the number of devices in the pipeline for ideal
        performance.

    checkpoint_queue: Queue, optional
        If provided, weight checkpoints will be pushed into this queue, along with the final set of weights.
        If one is not provided, one will be created and returned.

    loss_queue: Queue, optional
        If provided, loss values will be pushed into this queeu.

    checkpoint_interval: int, optional
        The weights will be checkpointed into checkpoint queues on host every `checkpoint_interval` optimizer
        steps, if set to non-zero. Zero by default.

    _sequential: Internal
        Don't use

    _perf_trace: Internal
        Don't use

    _verify_cfg: Internal
        Don't use.

    Returns
    -------
    queue.Queue
         Checkpoint queue, holding weight checkpoints, and final trained weights.

    """

    if epochs == 0 or steps == 0 or accumulation_steps == 0 or microbatch_count == 0:
        raise RuntimeError("Calling run_training with one of the loop indices at 0. Nothing to do.")

    return _run_devices_training(sequential=_sequential, epochs=epochs, steps=steps, accumulation_steps=accumulation_steps, microbatch_count=microbatch_count, checkpoint_interval=checkpoint_interval, perf_trace=_perf_trace, checkpoint_queue=checkpoint_queue, loss_queue=loss_queue, verify_cfg=_verify_cfg)

def run_generative_inference(
        module: Optional[PyBudaModule] = None,
        inputs: List[Union[Tuple[Union[torch.Tensor, Tensor], ...], Dict[str, Union[torch.Tensor, Tensor]]]] = [],
        input_count: int = 1,
        output_queue: queue.Queue = None,
        _sequential: bool = False,
        _perf_trace: bool = False,
        _verify_cfg: Optional[VerifyConfig] = None) -> queue.Queue:
    """
    Main "run" function for generative inference. After all modules have been defined and placed on devices, this will 
    execute the workload. Unless 'sequential' is set, the function will return as soon as the devices are set up
    to run, and inference will run as long as new inputs are pushed into the device(s). If sequential mode is on,
    the function will run through inputs that are already in the input buffer and return when done.

    Parameters
    ----------
    module: PyBudaModule, optional
        If provided, place given module on a TT Device and run inference. Alternatively, manually create device(s) and
        placed module(s) on them.

    inputs: List[Union[Tuple[Union[torch.Tensor, Tensor], ...], Dict[str, Union[torch.Tensor, Tensor]]]], optional
        An optional list of input tensor tuples or dictionaries (passed as args or kwargs to module), to feed into the inference pipeline. 
        Alternatively, use `device.push_to_inputs` to manually provide inputs outside of this call.

    input_count: int, default=1
        The number of inputs to run inference on. If 0, inference will run "forever", until `shutdown` or `run_inference`
        is called again.

    output_queue: queue.Queue, optional
        If provided, outputs will be pushed into the queue as they are calculated. Otherwise, one will be created
        and returned.

    _sequential: bool, Internal
        Don't use.

    _perf_trace: bool, Internal
        Don't use.

    _verify_cfg: Internal
        Don't use.

    Returns
    -------
    queue.Queue
        Queue holding the output results. Either the output_queue provided, or one that's created.

    """

    return _run_generative_inference(module, inputs, input_count, output_queue, _sequential, _perf_trace, _verify_cfg)

def run_forward(input_count: int = 1, _sequential: bool = False):
    """
    Run forward passes on the pre-compiled and initialized pipeline of devices. This API should be
    called from custom implementations of inference and training loops, in lieue of calling 
    `run_inference` and `run_training` APIs.

    If this is a part of an inference run, the results will be placed in the outptut queues which 
    should have already been setup through `initialize_pipeline` call. If this is called as a part
    of the training pass, then loss will be pushed to the output queue, if one was set up.

    Parameters
    ----------
    input_count: int, default=1
        The number of inputs to run inference on. If 0, inference will run "forever", until `shutdown` or `run_inference`
        is called again.

    _sequential: Internal
        Don't use
    """
    return _run_forward(input_count, _sequential)

def run_generate(input_count: int = 1, write_index: int = -1, tokens_per_iter: int = -1, token_id: int = -1, _sequential: bool = False):
    """
    Run forward passes on the pre-compiled and initialized pipeline of devices and maintain past cache
    write and read pointers. This API should be called from custom implementations of inference and 
    training loops, in lieue of calling `run_inference` and `run_training` APIs.

    If this is a part of an inference run, the results will be placed in the outptut queues which 
    should have already been setup through `initialize_pipeline` call. If this is called as a part
    of the training pass, then loss will be pushed to the output queue, if one was set up.

    Parameters
    ----------
    input_count: int, default=1
        The number of inputs to run inference on. If 0, inference will run "forever", until `shutdown` or `run_inference`
        is called again.

    _sequential: Internal
        Don't use
    """
    assert write_index >= 0 or (tokens_per_iter > 0 and token_id >= 0), "Either write_index or tokens_per_iter and token_id should be set."
    return _run_generate(input_count, write_index, tokens_per_iter, token_id, _sequential)

def run_backward(input_count: int = 1, zero_grad: bool = False, _sequential: bool = False):
    """
    Run backward passes on the pre-compiled and initialized pipeline of devices. This API should be 
    called from custom implementations of inference and training loops, in lieue of calling 
    `run_inference` and `run_training` APIs.

    `zero_grad` should be set for the first backward call of a batch, to zero out accumulated gradients.

    No results will be returned. get_parameter_gradients() can be used to get a snapshot of
    gradients after the backward pass has completed.

    Parameters
    ----------
    input_count: int, default=1
        The number of inputs to run inference on. If 0, inference will run "forever", until `shutdown` or `run_inference`
        is called again.

    zero_grad: bool, optional
        If set, acccumulated gradients on device will be zeroed out before the backward pass begins.

    _sequential: Internal
        Don't use
    """
    return _run_backward(input_count, zero_grad, _sequential)

def run_optimizer(checkpoint: bool = False, _sequential: bool = False):
    """
    Run optimizer on all devices. If `checkpoint` is set, a checkpoint of parameters will be taken and 
    placed into the checkpoint queue that has been set up during `initialize_pipeline` call.

    Parameters
    ----------
    checkpoint: bool, optional
        If set, checkpoint of parameters will be placed into checkpoint queue.

    _sequential: Internal
        Don't use
    """
    _run_optimizer(_sequential)

    if checkpoint:
        _save_parameter_checkpoint(_sequential)

def run_schedulers(_sequential: bool = False):
    """
    Run learning rate schedulers on all devices. 

    Parameters
    ----------
    _sequential: Internal
        Don't use
    """
    _run_schedulers(_sequential)

def get_parameter_gradients(device: Optional[Union["CPUDevice", "TTDevice"]] = None, _sequential: bool = False) -> List[Dict[str, Tensor]]:
    """
    Return currently accumulated parameter gradients. If a device is specified, only gradients for that device
    will be returned, otherwise a list of gradients for all devices will come back.

    Parameters
    ----------
    device: Union[CPUDevice, TTDevice], Optional
        Device to read parameter gradients from. If None, all devices will be read from.

    _sequential: Internal
        Don't use

    Returns
    -------
    List[Dict[str, Tensor]]
        List of parameter checkpoints for devices in the pipeline, or the given device
    """
    if device is None:
        return [_get_parameter_gradients(d, _sequential) for d in get_devices()]

    return [_get_parameter_gradients(device, _sequential)]

def get_parameter_checkpoint(device: Optional[Union["CPUDevice", "TTDevice"]] = None, _sequential: bool = False) -> List[Dict[str, Tensor]]:
    """
    Return current parameter values. If a device is specified, only parameters for that device will 
    be returned, otherwise a list of parameters for all devices will come back.

    Parameters
    ----------
    device: Union[CPUDevice, TTDevice], Optional
        Device to read parameter values from. If None, all devices will be read from.

    _sequential: Internal
        Don't use

    Returns
    -------
    List[Dict[str, Tensor]]
        List of parameter checkpoints for devices in the pipeline, or the given device
    """
    if device is None:
        return [_get_parameter_checkpoint(d, _sequential) for d in get_devices()]

    return [_get_parameter_checkpoint(device, _sequential)]

def update_device_parameters(device: Optional[Union["CPUDevice", "TTDevice"]] = None, parameters: List[Dict[str, Tensor]] = [], _sequential: bool = False):
    """
    Push new parameters onto given device, or if none is provided, then all devices in the pipeline.

    Parameters
    ----------
    device: Union[CPUDevice, TTDevice], Optional
        Device to read parameter values from. If None, all devices will be read from.

    parameters: List[Dict[str, torch.Tensor]]
        List of dictionaries of parameters to update

    _sequential: Internal
        Don't use
    """
    devices = [device] if device is not None else get_devices()
    return _update_device_parameters(devices, parameters, _sequential)


def initialize_pipeline(
        training: bool, 
        output_queue: Optional[queue.Queue] = None, 
        checkpoint_queue: Optional[queue.Queue] = None, 
        sample_inputs: Union[Tuple[Union[torch.Tensor, Tensor], ...], Dict[str, Union[torch.Tensor, Tensor]]] = tuple(),
        sample_targets: Tuple[Union[torch.Tensor, Tensor], ...] = tuple(),
        microbatch_count: int = 1,
        d2d_fwd_queues: List[queue.Queue] = [],
        d2d_bwd_queues: List[queue.Queue] = [],
        _sequential: bool = False, 
        _verify_cfg: Optional[VerifyConfig] = None,
        _device_mode: DeviceMode = DeviceMode.CompileAndRun) -> queue.Queue:
    """
    Initialize the pipeline to run inference and training through manual `run_forward`, `run_backward`, `run_optimizer`, etc. calls. This should be not used with 
    "all-in-one" APIs like `run_inference` and `run_training`, which will initialize the pipeline themselves.

    Parameters
    ----------
    training: bool
        Set to true to prepare the pipeline for training.

    output_queue: queue.Queue, optional
        If provided, inference outputs will be pushed into the queue as they are calculated. Otherwise, one will be created
        and returned (in inference mode)

    checkpoint_queue: Queue, optional
        If provided, weight checkpoints will be pushed into this queue, along with the final set of weights.
        If one is not provided, one will be created and returned (in training mode)

    sample_inputs: Tuple[Union[torch.Tensor, Tensor], ...], optional
        If calling initialize_pipeline directly to compile models and initialize devices, then a representative sample
        of inputs must be provided to accuractely compile the design. Typically, this would be the first input that 
        will be sent through the model post-compile. The tensors must be of the correct shape and data type.

    sample_targets: Tuple[Union[torch.Tensor, Tensor], ...], optional
        If calling initialize_pipeline directly to compile models and initialize devices for training, then a 
        representative sample of training tagets must be provided to accuractely compile the design. 
        Typically, this would be the first target that will be sent to the last device post-compile. 
        The tensors must be of the correct shape and data type.

    microbatch_count: int
        Only relevant for training. This represents the number of microbatches that are pushed through
        fwd path before bwd path runs. The device will ensure that buffering is large enough to contain
        microbatch_count number of microbatch intermediate data.

    d2d_fwd_queues: List[queue.Queue], optional
        If provided, device-to-device intermediate data that passes through host will also be stored in the provided
        queues. The queues are assigned in order from the first device in the pipeline. The last device will not 
        be assigned a queue.

    d2d_bwd_queues: List[queue.Queue], optional
        If provided, device-to-device intermediate data in the training backward pass, that passes through 
        host will also be stored in the provided queues. The queues are assigned in order from the 
        second device in the pipeline. The first device will not be assigned a queue.

    _sequential: Internal
        Don't use

    _verify_cfg: Internal
        Don't use.

    Returns
    -------
    queue.Queue
        Output queue for inference, or checkpoint queue for training


    """

    if not training:
        assert len(sample_targets) == 0, "Sample targets should not be provided unless the training mode is on"

    return _initialize_pipeline(training, output_queue, checkpoint_queue, sample_inputs, sample_targets, microbatch_count,
            d2d_fwd_queues, d2d_bwd_queues, _sequential, _verify_cfg, _device_mode)


def get_loss_queue() -> queue.Queue:
    """
    If a loss queue was not provided for training, one will be automatically created. This call can
    be used to retrieve that queue.
    """
    return _get_loss_queue()

def get_checkpoint_queue() -> queue.Queue:
    """
    If a checkpoint queue was not provided for training, one will be automatically created. This call can
    be used to retrieve that queue.
    """
    return _get_checkpoint_queue()

def get_intermediates_queue() -> queue.Queue:
    """
    If intermediates were tagged for saving, they will be pushed into a queue. This call can be used to retrieve that queue.
    """
    return _get_intermediates_queue()

def sync():
    """
    Block until all devices have gone idle.
    """
    _sync()

def shutdown():
    """ 
    Shutdown running processes and clean up pybuda
    """
    return _shutdown()

def error_raised() -> bool:
    """
    Returns True if an unrecoverable error has been raised. A full shutdown / reset is needed to restart.
    """
    return _error_raised()

def detect_available_devices():
    """
    Returns a list of available devices on the system.
    """
    return _detect_available_devices()

import atexit
atexit.register(shutdown)

