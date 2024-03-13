# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple, Optional, Union, Dict
import queue
import os
import threading
import copy

import torch
import torch.multiprocessing as mp
from loguru import logger

from .commands import Command
from .context import RunContext, get_current_context, clear_current_context
from ..pybudaglobal import get_devices, profiler, state_changed, clear_state_changed, set_device_pipeline, create_queue
from ..device import Device
from ..ttdevice import TTDevice
from ..cpudevice import CPUDevice
from ..gpudevice import GPUDevice
from ..module import PyBudaModule
from ..tensor import Tensor, remove_microbatch, to_buda_tensors, to_pt_tensors
from ..config import CompilerConfig
from ..verify import VerifyConfig, TestKind
from ..config import _get_global_compiler_config
from ..utils import detach_tensors

from pybuda.tvm_to_python import generate_pybuda_module, cleanup_temporary_files
from pybuda.tvm_utils import flatten_inputs
from pybuda._C.backend_api import BackendDevice, BackendType, initialize_child_process, finish_child_process, DeviceMode, clear_backend_param_cache, detect_available_silicon_devices

def _detect_available_devices():
    if "PYBUDA_EMULATE_SILICON_DEVICE" in os.environ:
        if "GOLDEN_WORMHOLE_B0" in os.environ:
            return [BackendDevice.from_string("wormhole_b0")]
        else:
            return [BackendDevice.from_string("grayskull")]
    else:
        return detect_available_silicon_devices()


def _translate_framework_modules_on_devices(
        sample_inputs: List[Tuple[Union[torch.Tensor, Tensor], ...]],
        sample_targets: List[Tuple[Union[torch.Tensor, Tensor], ...]],
        sample_input_names: List[str],
        compiler_cfg: CompilerConfig,
        verify_cfg: VerifyConfig):

    def _wrap_inputs(inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs, )
        return inputs

    prev_state = state_changed()
    devices = get_devices()
    updated_devices = copy.copy(devices)
    _, inputs = _get_device_zero_inputs(sample_inputs, peek=True)
    device_index = 0
    while device_index < len(updated_devices):
        device = updated_devices[device_index]
        if isinstance(device, (CPUDevice,)):
            for module in device.modules:
                if device.loss_module is None:
                    if isinstance(device, GPUDevice):
                        inputs = [input.cuda() for input in to_pt_tensors(inputs)]
                        inputs = module.forward(*inputs)
                        inputs = [input.cpu() for input in inputs]
                        inputs = _wrap_inputs(inputs)
                    else:
                        module.compilation = True
                        inputs = _wrap_inputs(module.forward(*to_pt_tensors(inputs)))

        elif isinstance(device, TTDevice):
            # Modes of operation:
            # 1. compiler_cfg.compile_subgraphs = True:
            #    - Ensure number of input groups match with number of modules
            #    - For each module, generate a PyBudaModule, mark subgraph ID
            #    - ASSERT NO CPU fallback
            # 2. compiler_cfg.compile_subgraphs = False:
            #    - feed output of previous module to next module
            #    - For each module, generate a PyBudaModule, mark with the same subgraph ID
            #    - ASSERT NO CPU fallback

            multiple_module_on_one_device = len(device.modules) > 1
            
            # Multiple modules on one device
            if len(device.modules) > 1 and device.loss_module is None:

                # Compile multiple subgraphs
                if (compiler_cfg.compile_subgraphs):
                    num_modules = len(device.modules)
                    num_input_groups = len(inputs)
                    assert num_modules == num_input_groups, "Number of modules on a single TTDevice must match the number of input groups"
                    assert  device.loss_module is None, "Compile subgraph currently does not support loss module on the same device"
                    assert len(devices) == 1, "Compile subgraph currently does not support multiple devices"

                    for module_index, (module, input_group) in enumerate(zip(device.modules, inputs)):
                        if not isinstance(module, PyBudaModule):
                            # Generate PybudaModule through TVM
                            (
                                translated_modules,
                                translated_device_types,
                                inputs
                            ) = generate_pybuda_module(module, input_group, verify_cfg=verify_cfg, clean_later=True)

                            tt_device = updated_devices[device_index]

                            assert (len(translated_device_types) == 1 
                                    and translated_device_types[0] == "TTDevice"), "Compile subgraph currently does not support CPU fallback"

                            translated_pybuda_module = translated_modules[0]
                            inputs = _wrap_inputs(translated_pybuda_module.forward(*to_buda_tensors(inputs)))
                            translated_pybuda_module.subgraph_idx = module_index
                            tt_device.modules[module_index] = translated_pybuda_module
                        else:
                            module.subgraph_idx = module_index

                # Merge multiple subgraphs into one graph
                else:
                    for module_index, module in enumerate(device.modules):
                        if not isinstance(module, PyBudaModule):
                            # Generate PybudaModule through TVM
                            (
                                translated_modules,
                                translated_device_types,
                                inputs
                            ) = generate_pybuda_module(module, inputs, verify_cfg=verify_cfg, clean_later=True)

                            tt_device = updated_devices[device_index]

                            assert (len(translated_device_types) == 1 
                                    and translated_device_types[0] == "TTDevice"), "Multiple module on 1 device currently does not support CPU fallback"

                            translated_pybuda_module = translated_modules[0]
                            inputs = _wrap_inputs(translated_pybuda_module.forward(*to_buda_tensors(inputs)))
                            translated_pybuda_module.subgraph_idx = 0 # Multiple modules on 1 device, merge into 1 graph
                            tt_device.modules[module_index] = translated_pybuda_module
                        else:
                            inputs = _wrap_inputs(module.forward(*to_buda_tensors(inputs)))
                            module.subgraph_idx = 0 # Multiple modules on 1 device, merge into 1 graph
            
            else:
                assert compiler_cfg.compile_subgraphs == False, "Found only 1 module on a TTDevice, but compiler_cfg.compile_subgraphs is set to True"
                module_index = 0
                while module_index < len(device.modules):
                    module = updated_devices[device_index].modules[module_index]
                    is_last_device = device_index == len(updated_devices) - 1

                    if not isinstance(module, PyBudaModule):
                        is_loss_module = False
                        if module is device.loss_module:
                            is_loss_module = True
                            inputs = tuple(list(inputs + sample_targets))
                        translated_modules, translated_device_types, inputs = generate_pybuda_module(module, inputs, verify_cfg=verify_cfg, clean_later=True, input_names=sample_input_names)
                        tt_device = updated_devices[device_index]

                        added_modules = 0
                        assert len(translated_device_types) <= 3
                        assert any([device_type == "TTDevice" for device_type in translated_device_types])
                        for index, (module, device_type) in enumerate(zip(translated_modules, translated_device_types)):
                            if device_type == "CPUDevice":
                                inputs = to_pt_tensors(inputs)
                                input_dtypes = [inp.dtype for inp in inputs]
                                inputs = _wrap_inputs(module.forward(*inputs))
                                cpu_device = CPUDevice(name=f"cpu{index}_fallback", module=module, input_dtypes=input_dtypes)
                                logger.warning("Unsupported ops found {} main graph, will be executed on {}", 'before' if index == 0 else 'after', cpu_device)
                                if index == 0:
                                    # if the first device is a fallback device, we want any subsequent inputs pushed to the 
                                    # original device to go to cpu_device
                                    while not tt_device._input_buffer.empty():
                                        logger.debug("Copied input buffer from tt to cpu device")
                                        cpu_device.push_to_inputs(tt_device._input_buffer.get())
                                    tt_device.cpu_fallback_device_pre = cpu_device
                                else:
                                    tt_device.cpu_fallback_device_post = cpu_device
                                    if tt_device.loss_module is not None:
                                        logger.warning("Due to CPU fallback, loss module moved to {}", cpu_device)
                                        cpu_device.place_loss_module(tt_device.loss_module)
                                        tt_device.remove_loss_module()
                                        while not tt_device.target_input_queue.empty():
                                            logger.debug("Copied target buffer from tt to cpu device")
                                            cpu_device.push_to_target_inputs(tt_device.target_input_queue.get())

                                updated_devices.insert(device_index, cpu_device)
                                device_index += 1
                            else:
                                inputs = _wrap_inputs(module.forward(*to_buda_tensors(inputs)))
                                tt_device.modules[module_index] = module
                                if is_loss_module:
                                    tt_device.loss_module = module
                                added_modules += 1
                                device_index += 1
                        # if the original device had an optimizer, and we have fallback device(s) we need to create one for the fallback device(s)
                        if tt_device.optimizer and (tt_device.cpu_fallback_device_pre or tt_device.cpu_fallback_device_post):
                            fallback_params = {}
                            if tt_device.cpu_fallback_device_pre:
                                fallback_params.update({param.get_name() : param.value() for param in tt_device.cpu_fallback_device_pre.modules[0].get_parameters()})
                            if tt_device.cpu_fallback_device_post:
                                fallback_params.update({param.get_name() : param.value() for param in tt_device.cpu_fallback_device_post.modules[0].get_parameters() if not any([param.value() is existing_param for existing_param in fallback_params.values()])})

                            if len(fallback_params):
                                cpu_optim = tt_device.optimizer.get_pytorch_optimizer(fallback_params)
                                # optimizer goes on last device in the pipeline
                                if tt_device.cpu_fallback_device_post:
                                    tt_device.cpu_fallback_device_post.optimizer = cpu_optim
                                else:
                                    tt_device.cpu_fallback_device_pre.optimizer = cpu_optim

                        # incremented for each added device, and will be incremented once again below
                        device_index -= 1
                        module_index += added_modules - 1
                    elif (not is_last_device or len(device.modules) > 1) and device.loss_module is None:
                        inputs = _wrap_inputs(module.forward(*to_buda_tensors(inputs)))
                    module_index += 1

        device_index += 1

    set_device_pipeline(updated_devices)
    if not prev_state:
        clear_state_changed()

def _cleanup_temporary_files():
    cleanup_temporary_files()

def _run_inference(
        module: Optional[PyBudaModule] = None,
        inputs: List[Union[Tuple[Union[torch.Tensor, Tensor], ...], Dict[str, Union[torch.Tensor, Tensor]]]] = [],
        input_count: int = 1,
        output_queue: queue.Queue = None,
        sequential: bool = False,
        perf_trace: bool = False,
        verify_cfg: Optional[VerifyConfig] = None) -> queue.Queue:
    """
    Main "run" function for inference. After all modules have been defined and placed on devices, this will 
    execute the workload. Unless 'sequential' is set, the function will return as soon as the devices are set up
    to run, and inference will run as long as new inputs are pushed into the device(s). If sequential mode is on,
    the function will run through inputs that are already in the input buffer and return when done.
    """

    resume = False
    if module is not None:
        # Create a device if one hasn't been created yet
        devices = get_devices()
        if len(devices) == 0:
            _ = TTDevice("auto_tt0")
            devices = get_devices()

        # Check if we'r resuming or starting a new run
        if (any(len(d.modules) > 0 for d in devices[1:])
                or len(devices[0].modules) != 1
                or (len(devices[0].modules) == 1 and devices[0].modules[0] != module)):
            for d in devices:
                d.remove_modules()
            devices[0].place_module(module)
        else:
            logger.debug("Resuming previous inference")
            resume = True # called with the same module

    elif not state_changed():
        resume = True


    if len(inputs) > 0:
        devices = get_devices()
        if len(devices) == 0:
            raise RuntimeError("No devices have been created, and no modules provided. There's nothing to run inference on.")
        for input in inputs:
            devices[0].push_to_inputs(input)
        if input_count != 1 and input_count != len(inputs):
            raise RuntimeError("Input count should not be provided when a list of inputs exists")
        input_count = len(inputs)

    if input_count == 0 and sequential:
        raise RuntimeError("In sequential mode, inputs must be pushed ahead of time. Therefore, 'run forever' mode is invalid.")

    clear_state_changed()
    return _run_devices_inference(
            input_count=input_count,
            sequential=sequential, 
            output_queue=output_queue, 
            perf_trace=perf_trace, 
            verify_cfg=verify_cfg, 
            resume=resume)

def _run_command(device: Union[CPUDevice, TTDevice], sequential: bool, command: Command, response: bool = False
        ) -> Optional[Dict]:
    if sequential:
        logger.trace("{}: Got command from queue: {}", device, command)
        device.run_next_command(command)
    else:
        device.push_to_command_queue(command)

    if response:
        return device.get_command_queue_response()

    return None

def _sequential_override(sequential: bool) -> bool:
    """
    Force sequential on if any of the devices are Golden model
    """
    if sequential:
        return True
    if "PYBUDA_FORCE_SEQUENTIAL" in os.environ:
        return True
    for d in get_devices():
        if d.devtype == BackendType.Golden or d.devtype == BackendType.Model:
            return True
    return False
    

def _initialize_pipeline(
        training: bool, 
        output_queue: Optional[queue.Queue] = None,
        checkpoint_queue: Optional[queue.Queue] = None, 
        sample_inputs: Union[Tuple[Union[torch.Tensor, Tensor], ...], Dict[str, Union[torch.Tensor, Tensor]]] = tuple(),
        sample_targets: Tuple[Union[torch.Tensor, Tensor], ...] = tuple(),
        microbatch_count: int = 1,
        d2d_fwd_queues: List[queue.Queue] = [],
        d2d_bwd_queues: List[queue.Queue] = [],
        sequential: bool = False, 
        verify_cfg: Optional[VerifyConfig] = None,
        device_mode: DeviceMode = DeviceMode.CompileAndRun) -> queue.Queue:
    """
    Initialize the pipeline to run inference and training through manual `run_forward`, `run_backward`, 
    `run_optimizer`, etc. calls. This should be not used with "all-in-one" APIs like `run_inference` 
    and `run_training`, which will initialize the pipeline themselves.
    """

    # If sample_inputs is a dictionary, extract names from its keys
    sample_input_names = []
    if isinstance(sample_inputs, dict):
        sample_input_names = list(sample_inputs.keys())
        sample_inputs = list(sample_inputs.values())

    devices = get_devices()

    if len(devices) == 0:
        logger.warning("Nothing to do")
        return None

    sequential = _sequential_override(sequential)
    if not sequential:
        initialize_child_process(_get_global_compiler_config().backend_output_dir)

    if training:
        for d in devices[:-1]:
            if d.loss_module is not None:
                raise RuntimeError("Only the last device in the pipieline should have a loss module.")

        if devices[-1].loss_module is None:
            raise RuntimeError("The last device in pipeline must have a loss module to be able to train.")
    else:
        if checkpoint_queue is not None:
            raise RuntimeError("Checkpoint queue should only be provided in training mode")

    # Translate framework modules. May increase number of devices due to CPU fallback
    # sample_inputs, _, _ = flatten_inputs(sample_inputs) # NESTED INPUT ASSERT NUM GROUP == NUM MODULES ON THAT DEVICE
    _translate_framework_modules_on_devices(sample_inputs, sample_targets, sample_input_names, _get_global_compiler_config(), verify_cfg)
    devices = get_devices()

    # Initialize & connect devices
    shutdown_event, final_barrier = _initialize_devices(devices, sequential, training=training, verify_cfg=verify_cfg)

    # Create a new context
    ctx = get_current_context()
    if ctx is None:
        ctx = RunContext.create_new(training, shutdown_event, final_barrier)

    mp_context = mp.get_context('spawn')
    if output_queue is None:
        output_queue = create_queue(mp_context)
    
    if _get_global_compiler_config().save_intermediates:
        ctx.intermediates_queue = create_queue(mp_context)

    microbatch, _ = _get_device_zero_inputs(sample_inputs, peek=True)

    # It's possible that no inputs have been provided at this point, just default to 1
    if microbatch is None:
        microbatch = 1

    if training:
        input_gradient_queue = create_queue(mp_context) # create a sink so that we can drain it here
        _connect_devices(
                devices, 
                sequential=sequential,
                training=True, 
                microbatch=microbatch,
                input_gradient_queue=input_gradient_queue, 
                output_queue=output_queue,
                intermediates_queue=ctx.intermediates_queue,
                d2d_fwd_queues=d2d_fwd_queues,
                d2d_bwd_queues=d2d_bwd_queues)
        ctx.input_gradient_queue = input_gradient_queue
    else:
        _connect_devices(
                devices, 
                sequential=sequential,
                training=False,
                microbatch=microbatch,
                input_gradient_queue=None, 
                output_queue=output_queue,
                intermediates_queue=ctx.intermediates_queue,
                d2d_fwd_queues=d2d_fwd_queues)

    ctx.output_queue = output_queue

    # Start device processes
    if not sequential:
        ctx.processes = _start_device_processes(devices, _get_global_compiler_config().backend_output_dir)

    # Compile all devices
    _compile_devices(sequential, training=training, sample_inputs=sample_inputs, sample_targets=sample_targets, microbatch_count=microbatch_count, verify_cfg=verify_cfg)

    if device_mode == DeviceMode.CompileOnly:
        return output_queue

    # Pass DRAM queue information between compiled devices
    _pass_dram_io_descriptors(devices, sequential, training=training, save_intermediates=_get_global_compiler_config().save_intermediates)

    if training:
        # Create queues for input/parameter gradients, for verification (if enabled)
        if verify_cfg and verify_cfg.enable_input_gradient_checking:
            verify_cfg._input_gradient_queue = input_gradient_queue
        if verify_cfg and verify_cfg.enable_parameter_gradient_checking:
            verify_cfg._parameter_gradient_queue = create_queue(mp_context)

        # Create checkpoint_queue if one is not provided
        if checkpoint_queue is None:
            checkpoint_queue = queue.Queue()

        ctx.checkpoint_queue = checkpoint_queue

        return checkpoint_queue

    return output_queue

def _is_active() -> bool:
    """ 
    Return true if a run is active
    """
    ctx = get_current_context()
    return ctx is not None and ctx.active

def _run_devices_inference(input_count: int, sequential: bool, output_queue: Optional[queue.Queue], resume: bool, perf_trace: bool, verify_cfg: Optional[VerifyConfig]):

    devices = get_devices()

    if len(devices) == 0:
        logger.warning("Nothing to do")
        return output_queue

    if resume and not _is_active():
        # can't really resume
        resume = False
        
    if resume:
        ctx = get_current_context()
        assert ctx is not None

    if resume and ctx.output_queue is None:
        logger.warning("Output queue not saved from previous run")
        resume = False

    if not resume:
        output_queue = _initialize_pipeline(False, output_queue, sequential=sequential, verify_cfg=verify_cfg)
    else:
        output_queue = ctx.output_queue

    sequential = _sequential_override(sequential)
    if sequential:
        _run_forward(input_count, sequential)
    else:
        loop_thread = threading.Thread(target=_run_forward, args=(input_count, sequential))
        ctx = get_current_context()
        assert ctx is not None
        ctx.loop_thread = loop_thread
        loop_thread.start()

    return output_queue

def _error_shutdown():
    """ 
    Cleanup on error
    """
    if "PYBUDA_TRACE_SHUTDOWN" in os.environ:
        import traceback
        logger.debug(traceback.format_exc())
    ctx = get_current_context()
    if ctx is None:
        # There's not context, something went really wrong...
        logger.warning("No context available for error shutdown.")
        return

    ctx.error = True
    if ctx.final_barrier:
        ctx.final_barrier.abort()
    if ctx.shutdown_event:
        ctx.shutdown_event.set()

    _shutdown(clear_context=False)

def _error_raised() -> bool:
    ctx = get_current_context()
    if ctx is not None and ctx.shutdown_event and ctx.shutdown_event.is_set():
        return True

    return ctx is not None and ctx.error

def _run_forward_with_fw_looping(ctx: RunContext, microbatch_looping: bool, devices: List[Device], input_count: int, sequential: bool):
    num_pushes_per_fwd = int(os.environ["NUM_EXEC_LOOP_ITERATIONS"])

    logger.info(f"impl.py::_run_forward num_pushes_per_fwd = {num_pushes_per_fwd}")
    logger.info(f"impl.py::_run_forward input_count = {input_count}")

    try:
        i = ctx.global_input_index
        logger.info(f"ctx.global_input_index={ctx.global_input_index}")
        if microbatch_looping:
            if fw_epoch_looping_enabled:
                assert(num_pushes_per_fwd % input_count == 0)
            
            for d in devices:
                invoke_fwd = (i % num_pushes_per_fwd) == 0 or not isinstance(d, TTDevice)
                if invoke_fwd:
                    logger.debug("Running {} device forward: {}", 'sequential' if sequential else 'concurrent', d)
                    _run_command(d, sequential, Command.run_forward(loop_count=input_count))
                    if _error_raised():
                        return 

            ctx.global_input_index += input_count    

            for _ in range(input_count):
                if ctx.training:
                    _run_command(devices[-1], sequential, Command.dc_transfer("target"))
                    if _error_raised():
                        return

                for d in devices:
                    _run_command(d, sequential, Command.dc_transfer("forward_input"))
                    if _error_raised():
                        return

                # Read out the output
                _run_command(devices[-1], sequential, Command.dc_transfer("forward"))
                if _error_raised():
                    return
        else:
            for _ in range(input_count):
                if ctx.training:
                    _run_command(devices[-1], sequential, Command.dc_transfer("target"))
                    if _error_raised():
                        return

                for d in devices:
                    _run_command(d, sequential, Command.dc_transfer("forward_input"))
                    invoke_fwd = (i % num_pushes_per_fwd) == 0 or not isinstance(d, TTDevice)
                    if invoke_fwd:
                        logger.debug("Running {} device forward: {}", 'sequential' if sequential else 'concurrent', d)
                        _run_command(d, sequential, Command.run_forward(loop_count=1))

                        if _error_raised():
                            return 

                # Read out the output
                _run_command(devices[-1], sequential, Command.dc_transfer("forward"))
                ctx.global_input_index += 1

    except Exception as e:
        logger.error("Forward loop error: {}", e)
        _error_shutdown()

def _run_forward(input_count: int = 1, sequential: bool = False):
    """
    Run forward passes on the pre-compiled and initialized pipeline of devices. This API should be 
    called from custom implementations of inference and training loops, in lieue of 
    calling `run_inference` and `run_training` APIs.

    The result (inference output, or loss if running training) will be placed in the output 
    queue which should have already been setup through `initialize_pipeline` call.
    """

    if _error_raised():
        return 

    devices = get_devices()
    if len(devices) == 0:
        logger.warning("Nothing to do")
        return 

    sequential = _sequential_override(sequential)
    microbatch_looping = False if sequential else "PYBUDA_MICROBATCH_LOOPING" in os.environ

    ctx = get_current_context()
    if ctx is None:
        raise RuntimeError("Trying to run forward without initializing the pipeline.")

    fw_epoch_looping_enabled = bool("NUM_EXEC_LOOP_ITERATIONS" in os.environ and int(os.environ["NUM_EXEC_LOOP_ITERATIONS"]) > 0)
    if fw_epoch_looping_enabled:
        _run_forward_with_fw_looping(ctx, microbatch_looping, devices, input_count, sequential)
        ## return instead of if-else just to minimize the diff for when we push (and later revert) this change
        return

    try:
        if microbatch_looping:
            for d in devices:
                logger.debug("Running {} device forward: {}", 'sequential' if sequential else 'concurrent', d)
                _run_command(d, sequential, Command.run_forward(loop_count=input_count))
                if _error_raised():
                    return 

            for _ in range(input_count):
                if ctx.training:
                    _run_command(devices[-1], sequential, Command.dc_transfer("target"))
                    if _error_raised():
                        return

                for d in devices:
                    _run_command(d, sequential, Command.dc_transfer("forward_input"))
                    if _error_raised():
                        return

                # Read out the output
                _run_command(devices[-1], sequential, Command.dc_transfer("forward"))
                if _error_raised():
                    return

                # Read out intermediates output
                if not ctx.training:
                    for d in devices:
                        _run_command(d, sequential, Command.dc_transfer("intermediates"))
                        if _error_raised():
                            return

        else:
            for _ in range(input_count):
                if ctx.training:
                    _run_command(devices[-1], sequential, Command.dc_transfer("target"))
                    if _error_raised():
                        return

                for d in devices:
                    logger.debug("Running {} device forward: {}", 'sequential' if sequential else 'concurrent', d)
                    _run_command(d, sequential, Command.dc_transfer("forward_input"))
                    _run_command(d, sequential, Command.run_forward(loop_count=1))
                    if _error_raised():
                        return

                # Read out the output
                _run_command(devices[-1], sequential, Command.dc_transfer("forward"))
                if _error_raised():
                    return

                # Read out intermediates output
                if not ctx.training:
                    for d in devices:
                        _run_command(d, sequential, Command.dc_transfer("intermediates"))
                        if _error_raised():
                            return

    except Exception as e:
        logger.error("Forward loop error: {}", e)
        _error_shutdown()

def _run_backward(input_count: int, zero_grad: bool, sequential: bool):

    if _error_raised():
        return 

    devices = get_devices()
    if len(devices) == 0:
        logger.warning("Nothing to do")
        return 

    sequential = _sequential_override(sequential)
    microbatch_looping = False if sequential else True

    try:
        if microbatch_looping:
            for d in reversed(devices):
                logger.debug("Running {} device backward: {}", 'sequential' if sequential else 'concurrent', d)
                _run_command(d, sequential, Command.run_backward(loop_count=input_count, zero_grad=zero_grad))
                if _error_raised():
                    return 

                for _ in range(input_count):
                    _run_command(d, sequential, Command.dc_transfer("backward"))
                    if _error_raised():
                        return

                    _run_command(d, sequential, Command.dc_transfer("intermediates"))
                    if _error_raised():
                        return 
        else:
            for i in range(input_count):
                for d in reversed(devices):
                    logger.debug("Running {} device backward: {}", 'sequential' if sequential else 'concurrent', d)
                    _run_command(d, sequential, Command.run_backward(loop_count=1, zero_grad=zero_grad))
                    if _error_raised():
                        return 

                    _run_command(d, sequential, Command.dc_transfer("backward"))
                    if _error_raised():
                        return 

                    _run_command(d, sequential, Command.dc_transfer("intermediates"))
                    if _error_raised():
                        return
                zero_grad = False

    except Exception as e:
        logger.error("Backward loop error: {}", e)
        _error_shutdown()

def _run_generate(input_count: int, write_index: int, tokens_per_iter: int, token_id: int, sequential: bool):
    if _error_raised():
        return 

    devices = get_devices()
    if len(devices) == 0:
        logger.warning("Nothing to do")
        return 

    sequential = _sequential_override(sequential)
    microbatch_looping = False if sequential else "PYBUDA_MICROBATCH_LOOPING" in os.environ

    try:
        if microbatch_looping:
            for d in devices:
                logger.debug("Running {} device forward: {}", 'sequential' if sequential else 'concurrent', d)
                _run_command(d, sequential, Command.run_generate(loop_count=input_count, write_index=write_index, tokens_per_iter=tokens_per_iter, token_id=token_id))
                if _error_raised():
                    return 

            for _ in range(input_count):
                for d in devices:
                    _run_command(d, sequential, Command.dc_transfer("forward_input"))
                    if _error_raised():
                        return

                # Read out the output
                _run_command(devices[-1], sequential, Command.dc_transfer("forward"))
                if _error_raised():
                    return
        else:
            for _ in range(input_count):
                for d in devices:
                    _run_command(d, sequential, Command.dc_transfer("forward_input"))
                    logger.debug("Running {} device forward: {}", 'sequential' if sequential else 'concurrent', d)
                    _run_command(d, sequential, Command.run_generate(loop_count=1, write_index=write_index, tokens_per_iter=tokens_per_iter, token_id=token_id))
                    token_id += tokens_per_iter
                    if _error_raised():
                        return

                # Read out the output
                _run_command(devices[-1], sequential, Command.dc_transfer("forward"))
                if _error_raised():
                    return
            # if there are no inputs to push, just execute the program once
            if input_count == 0:
                for d in devices:
                    _run_command(d, sequential, Command.run_generate(loop_count=1, write_index=write_index, tokens_per_iter=tokens_per_iter, token_id=token_id))
                    if _error_raised():
                        return
               

    except Exception as e:
        logger.error("Generate loop error: {}", e)
        _error_shutdown()

def _run_optimizer(sequential: bool):

    if _error_raised():
        return 

    devices = get_devices()
    if len(devices) == 0:
        logger.warning("Nothing to do")
        return 

    sequential = _sequential_override(sequential)

    try:
        for d in devices:
            logger.debug("Running {} device optimizer: {}", 'sequential' if sequential else 'concurrent', d)
            _run_command(d, sequential, Command.run_optimizer())
            if _error_raised():
                return 
    except Exception as e:
        logger.error("Optimizer loop error: {}", e)
        _error_shutdown()

def _run_schedulers(sequential: bool):

    if _error_raised():
        return 

    devices = get_devices()
    if len(devices) == 0:
        logger.warning("Nothing to do")
        return 

    sequential = _sequential_override(sequential)

    try:
        for d in devices:
            logger.debug("Running {} device scheduler: {}", 'sequential' if sequential else 'concurrent', d)
            _run_command(d, sequential, Command.run_schedulers())
            if _error_raised():
                return 
    except Exception as e:
        logger.error("Scheduler loop error: {}", e)
        _error_shutdown()

def _get_parameter_checkpoint(device: Union[CPUDevice, TTDevice], sequential: bool) -> Dict[str, Tensor]:
    sequential = _sequential_override(sequential)

    try:
        ret = _run_command(device, sequential, Command.get_parameter_checkpoint(), response=True)
        if ret is None:
            raise RuntimeError("Error getting parameter checkpoint")

        return ret["checkpoint"]

    except Exception as e:
        logger.error("Parameter checkpoint error: {}", e)
        _error_shutdown()
        return {}

def _get_parameter_gradients(device: Union[CPUDevice, TTDevice], sequential: bool) -> Dict[str, Tensor]:
    sequential = _sequential_override(sequential)

    try:
        ret = _run_command(device, sequential, Command.get_parameter_gradients(), response=True)
        if ret is None:
            raise RuntimeError("Error getting parameter gradients")

        return ret["gradients"]

    except Exception as e:
        logger.error("Parameter gradient read error: {}", e)
        _error_shutdown()
        return {}
    
def _get_device_intermediates(device: Union[CPUDevice, TTDevice], sequential: bool) -> Dict[str, Tensor]:
    sequential = _sequential_override(sequential)

    try:
        ret = _run_command(device, sequential, Command.get_device_intermediates(), response=True)
        if ret is None:
            raise RuntimeError("Error getting intermediate activations")

        return ret["device_intermediates"]

    except Exception as e:
        logger.error("Intermediate activations read error: {}", e)
        _error_shutdown()
        return {}
    

def _run_training_loop(
        sequential: bool,
        epochs: int, 
        steps: int, 
        accumulation_steps: int, 
        microbatch_count: int, 
        checkpoint_interval: int,
        checkpoint_queue: queue.Queue,
        verify_cfg: Optional[VerifyConfig]):
    """
    Run the training loop after everything's been set up.
    """
    try:
        optimizer_step_count = 0
        checkpointed = False
        devices = get_devices()
        for epoch in range(epochs):

            if _error_raised():
                return

            logger.info("**** Starting epoch {}", epoch)

            checkpointed = False
            for batch in range(steps):
            
                logger.info("** Starting batch {} in epoch {}", batch, epoch)
                for mini_batch in range(accumulation_steps):
                    logger.info("** Starting mini-batch {}, batch {}, in epoch {}", mini_batch, batch, epoch)
                    _run_forward(input_count=microbatch_count, sequential=sequential)

                    if _error_raised():
                        return

                    _run_backward(input_count=microbatch_count, zero_grad=(mini_batch==0), sequential=sequential)

                    if _error_raised():
                        return

                    # Drain input gradients if nobody is consuming them - TODO
                    #if verify_cfg is None or not verify_cfg.enable_input_gradient_checking:
                    #    while not input_gradient_queue.empty():
                    #        input_gradient_queue.get()

                    # Record gradients for verification
                    if verify_cfg and verify_cfg._parameter_gradient_queue is not None:
                        
                        gradient_checkpoint = [_get_parameter_gradients(device, sequential) for device in devices]
                        verify_cfg._parameter_gradient_queue.put(gradient_checkpoint)


                _run_optimizer(sequential)

                if _error_raised():
                    return

                optimizer_step_count += 1
                if (checkpoint_interval > 0) and (optimizer_step_count % checkpoint_interval == 0):
                    checkpoint = [_get_parameter_checkpoint(device, sequential) for device in devices]
                    checkpoint_queue.put(checkpoint)
                    checkpointed = True


            for d in devices:
                d._step_schedulers()

            if _error_raised():
                return
    
        # Save final checkpoint
        if not checkpointed: # don't double-checkpoint on the last one
            checkpoint = [_get_parameter_checkpoint(device, sequential) for device in devices]
            checkpoint_queue.put(checkpoint)

    except Exception as e:
        logger.error("Training loop error: {}", e)
        _error_shutdown()

def _run_devices_training(
        sequential: bool,
        epochs: int, 
        steps: int, 
        accumulation_steps: int, 
        microbatch_count: int, 
        checkpoint_interval: int,
        perf_trace: bool,
        checkpoint_queue: Optional[queue.Queue],
        loss_queue: Optional[queue.Queue],
        verify_cfg: Optional[VerifyConfig]) -> queue.Queue:
    
    devices = get_devices()

    if len(devices) == 0:
        logger.warning("Nothing to do")
        return checkpoint_queue

    checkpoint_queue = _initialize_pipeline(training=True, output_queue=loss_queue, checkpoint_queue=checkpoint_queue, sequential=sequential, verify_cfg=verify_cfg, microbatch_count=microbatch_count)
    
    sequential = _sequential_override(sequential)
    
    loop_args = (sequential, epochs, steps, accumulation_steps, microbatch_count, checkpoint_interval, checkpoint_queue, verify_cfg)
    if sequential:
        _run_training_loop(*loop_args)
        if profiler is not None:
            profiler.stop()
            profiler.print()
            html = profiler.output_html()
            with open('training_sequential_profile.html', 'w') as f:
                f.write(html)
    else:
        loop_thread = threading.Thread(target=_run_training_loop, args=loop_args)
        ctx = get_current_context()
        assert ctx is not None
        ctx.loop_thread = loop_thread
        loop_thread.start()

    return checkpoint_queue

def _save_parameter_checkpoint(sequential: bool):
    """
    Read a checkpoint of parameters and push to checkpoint queue
    """
    devices = get_devices()

    if len(devices) == 0:
        logger.warning("Nothing to do")
        return

    ctx = get_current_context()
    if ctx is None:
        raise RuntimeError("No current running context")

    if not ctx.training or ctx.checkpoint_queue is None:
        raise RuntimeError("Pipeline hasn't been initialized for training")

    try:
        checkpoint_queue = ctx.checkpoint_queue
        sequential = _sequential_override(sequential)
        checkpoint = [_get_parameter_checkpoint(device, sequential) for device in devices]
        checkpoint_queue.put(checkpoint)
    except Exception as e:
        logger.error("Save parameter checkpoint error: {}", e)
        _error_shutdown()

def _initialize_devices(devices: List[Union[CPUDevice, TTDevice]], sequential: bool, training: bool, verify_cfg: Optional[VerifyConfig]):
    """
    Setup all devices
    """
    if not sequential:
        mp_context = mp.get_context('spawn')
        shutdown_event = mp_context.Event()
        final_barrier = mp_context.Barrier(len(devices) + 1) # plus 1 for this process
    else:
        final_barrier = None
        shutdown_event = mp.Event()

    scale_loss = 1.0
    if verify_cfg is not None and training:
        scale_loss = verify_cfg.scale_loss

    d: Union[CPUDevice, TTDevice]
    for i, d in enumerate(devices):
        d._initialize(
                training=training, 
                sequential=sequential,
                final_barrier=final_barrier,
                shutdown_event=shutdown_event,
                checkpoint_interval=0,
                scale_loss=scale_loss)
                #perf_dump_mode=buda.PerfDumpMode.SingleDumpPerEpoch if perf_trace else buda.PerfDumpMode.Disable)
        if i > 0:
            d._first_device = False

    return shutdown_event, final_barrier

def _connect_devices(
        devices: List[Union[CPUDevice, TTDevice]], 
        sequential: bool,
        training: bool,
        microbatch: int,
        output_queue: queue.Queue,
        input_gradient_queue: Optional[queue.Queue],
        intermediates_queue: Optional[queue.Queue],
        d2d_fwd_queues: List[queue.Queue] = [],
        d2d_bwd_queues: List[queue.Queue] = []):
    """
    Connect devices by creating device connectors between the appropriate pairs
    """
    for i, d in enumerate(devices[:-1]):
        target_device = devices[i+1]

        d2d_fwd_queue = d2d_fwd_queues[i] if len(d2d_fwd_queues) > i else None
        d._create_forward_device_connector(target_device=target_device, sequential=sequential, d2d_fwd_queue=d2d_fwd_queue, microbatch=microbatch)
        if training:
            d2d_bwd_queue = d2d_bwd_queues[i] if len(d2d_bwd_queues) > i else None
            target_device._create_backward_device_connector(d, sequential=sequential, d2d_bwd_queue=d2d_bwd_queue)

    # Connect the first device to input / output of the whole system
    devices[0]._create_input_queue_device_connector(devices[0]._input_buffer, sequential=sequential)

    if intermediates_queue is not None:
        for device in devices:
            if isinstance(device, TTDevice):
                device._create_intermediates_queue_device_connector(intermediates_queue)

    if training:
        # Input gradient queue if one is provided
        if input_gradient_queue is not None:
            devices[0]._create_backward_output_queue_device_connector(input_gradient_queue)

        # Target & Loss queues
        devices[-1]._create_forward_output_queue_device_connector(output_queue)
        devices[-1]._create_target_queue_device_connector(devices[-1].target_input_queue, sequential=sequential)
    else:
        devices[-1]._create_forward_output_queue_device_connector(output_queue)

def _pass_dram_io_descriptors(devices: List[Union[CPUDevice, TTDevice]], sequential: bool, training: bool, save_intermediates: bool):
    """
    Pass dram io descriptors from TT devices to CPU devices
    """
    for i, d in enumerate(devices[:-1]):
        target_device = devices[i+1]

        # Get queue descriptors from the target device, if it's a TTDevice

        # Forward
        if isinstance(target_device, TTDevice):
            # Pushing to TTDevice, we need to set destination queue information
            ret = _run_command(target_device, sequential, Command.get_queues("input"), response=True)
            if ret is None:
                raise RuntimeError("Failed to connect devices.")
            _run_command(d, sequential, Command.set_queues("forward_out", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))

        if isinstance(d, TTDevice):
            # Reading from TTDevice, need to set TTDevice's output queues
            ret = _run_command(d, sequential, Command.get_queues("output"), response=True)
            if ret is None:
                raise RuntimeError("Failed to connect devices.")
            _run_command(target_device, sequential, Command.set_queues("forward_in", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))

        if not training:
            continue

        # Backward, the other way around
        if isinstance(d, TTDevice):
            # Pushing backward to TTDevice, we need to set destination queue information
            ret = _run_command(d, sequential, Command.get_queues("bw_input"), response=True)
            if ret is None:
                raise RuntimeError("Failed to connect devices.")
            _run_command(target_device, sequential, Command.set_queues("backward_out_push", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))

        if isinstance(target_device, TTDevice):
            # Reading backward from TTDevice, need to set TTDevice's output queues
            ret = _run_command(target_device, sequential, Command.get_queues("bw_output"), response=True)
            if ret is None:
                raise RuntimeError("Failed to connect devices.")
            _run_command(d, sequential, Command.set_queues("backward_in", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))


    # Set it for the first device
    if isinstance(devices[0], TTDevice):
        ret = _run_command(devices[0], sequential, Command.get_queues("input"), response=True)
        if ret is None:
            raise RuntimeError("Failed to connect devices.")
        # Force "sequential" to true to set ret on local process, which will be pushing data in
        _run_command(devices[0], sequential, Command.set_queues("forward_in_push", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))

    # Set it for the last device
    if training and isinstance(devices[-1], TTDevice):
        ret = _run_command(devices[-1], sequential, Command.get_queues("target"), response=True)
        if ret is None:
            raise RuntimeError("Failed to connect devices.")
        # Force "sequential" to true to set ret on local process, which will be pushing data in
        _run_command(devices[-1], sequential, Command.set_queues("target_in_push", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))

    if training and isinstance(devices[0], TTDevice):
        ret = _run_command(devices[0], sequential, Command.get_queues("bw_output"), response=True)
        if ret is None:
            raise RuntimeError("Failed to connect devices.")
        _run_command(devices[0], sequential, Command.set_queues("backward_out", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))

    if isinstance(devices[-1], TTDevice):
        ret = _run_command(devices[-1], sequential, Command.get_queues("output"), response=True)
        if ret is None:
            raise RuntimeError("Failed to connect devices.")
        _run_command(devices[-1], sequential, Command.set_queues("forward_out_pop", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))

    for device in devices:
        if save_intermediates and isinstance(device, TTDevice):
            ret = _run_command(device, sequential, Command.get_queues("intermediates"), response=True)
            if ret is None:
                raise RuntimeError("Failed to connect devices.")
            _run_command(device, sequential, Command.set_queues("intermediates_pop", ret["queues"], ret["tile_broadcast_dims"], ret["original_shapes"], ret["requires_grad"], ret["runtime_tensor_transforms"], ret["constant_inputs"], ret["tile_dims"]))


def _start_device_processes(devices: List[Union[CPUDevice, TTDevice]], output_dir: str) -> List[mp.Process]:
    processes: List = []
    mp_context = mp.get_context('spawn')

    try:
        for i, d in enumerate(devices):
            logger.trace("Creating child process for device {}", d)

            # Only first device should still have first inputs around. Due to automatic CPU fallback,
            # TT device that's no longer first could have a stale copy of these, which pytorch will
            # try to transfer over when starting the process, causing a "bad fds_to_keep" system error.
            if i > 0:
                d._first_inputs = None 

            # Create python thread instead of another process
            if os.environ.get("PYBUDA_FORCE_THREADS", "0") != "0":
                processes.append(threading.Thread(target=d.run, args=(output_dir,)))
            else:
                processes.append(mp_context.Process(target=d.run, args=(output_dir,)))

        for p in processes:
            p.start()
    except Exception as e:
        logger.error("Process spawn error: {}", e)
        _error_shutdown()

    return processes

def _get_device_zero_inputs(sample_inputs, peek=False):
    compiler_cfg = _get_global_compiler_config()
    devices = get_devices()
    if compiler_cfg.compile_subgraphs:
        num_input_groups = len(sample_inputs)
        num_modules = len(devices[0].modules)
        assert num_input_groups == num_modules, \
                "Number of input groups ({}) must match number of modules ({})".format(num_input_groups, num_modules)
        microbatch_size = sample_inputs[0][0].shape[0]

        batch_removed_inputs = []
        for i in range(num_input_groups):
            assert microbatch_size == sample_inputs[i][0].shape[0], \
                    "Microbatch size must be the same for subgraph modules."
            batch_removed_inputs.append(remove_microbatch(sample_inputs[i]))

        inputs = batch_removed_inputs
    else:
        if len(sample_inputs) > 0:
            microbatch_size = sample_inputs[0].shape[0]
            inputs = remove_microbatch(sample_inputs)
        else:
            microbatch_size, inputs = devices[0].get_first_inputs(peek)

    return microbatch_size, inputs

def _compile_devices(
        sequential: bool, 
        training: bool, 
        sample_inputs: Tuple[Union[torch.Tensor, Tensor], ...] = tuple(),
        sample_targets: Tuple[Union[torch.Tensor, Tensor], ...] = tuple(),
        microbatch_count: int = 1,
        verify_cfg: Optional[VerifyConfig] = None):
    """
    Compile modules on TT devices, for inference or training. If input shaes / types are provided, those
    will be used... otherwise, first input from the input buffer will be peeked at (one must be there already).
    """

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_training = training
    if verify_cfg is None:
        verify_cfg = VerifyConfig.disabled()
    else:
        verify_cfg.run_golden = False

    # TODO: need to give user ability to set this outside of verify_cfg
    for epoch_break in verify_cfg.epoch_breaks:
        compiler_cfg.place_on_new_epoch(epoch_break)

    devices = get_devices()
    microbatch_size, inputs = _get_device_zero_inputs(sample_inputs)
    data_parallel = os.getenv("PYBUDA_N300_DATA_PARALLEL", 0)
    if data_parallel:
        assert microbatch_size > 1, "microbatch size is expected to be >= 1 for data parallel"
        microbatch_size = int(microbatch_size / 2)

    targets = []
    if training:
        if len(sample_targets) > 0:
            targets = remove_microbatch(sample_targets)
        else:
            targets = devices[-1].get_first_targets()

    for i, d in enumerate(devices):
        dev_targets = [] if i < len(devices) - 1 else targets
        ret = _run_command(d, sequential, Command.compile(inputs, compiler_cfg, dev_targets, microbatch_size, microbatch_count, verify_cfg), response=True)
        if isinstance(ret, Exception):
            raise ret
        if ret is None:
            raise RuntimeError(f"Compile failed for {d}")

        inputs = ret["outputs"]

def _shutdown(clear_context: bool = True):
    """ 
    Shutdown running processes and clean up
    """
    _cleanup_temporary_files()

    ctx = get_current_context()
    if ctx is None:
        clear_backend_param_cache()
        return # nothing to shutdown
        
    logger.debug("PyBuda shutdown")

    sequential = len(ctx.processes) == 0
    devices = get_devices()

    if not _error_raised():
        for d in devices:
            _run_command(d, sequential, Command.quit())

        if ctx.final_barrier:
            logger.trace("Setting final barrier on main process")
            ctx.final_barrier.wait()

        if ctx.loop_thread:
            ctx.loop_thread.join()

    logger.debug("Waiting until processes done")
    if len(ctx.processes) > 0:

        if _error_raised():
            # wait a couple of seconds, then kill processes
            import time
            start = time.time()
            while time.time() - start <= 2:
                if not any(p.is_alive() for p in ctx.processes):
                    break
                time.sleep(0.25)
        
            for p in ctx.processes:
                p.terminate()
                p.join()

        else:
            # clean join
            for p in ctx.processes:
                if p == mp.current_process():
                    continue # don't wait on yourself
                p.join()

        finish_child_process() # clean up backend

    if clear_context:
        clear_current_context()

def _update_device_parameters(devices: List[Union["CPUDevice", "TTDevice"]], parameters: List[Dict[str, torch.Tensor]], sequential: bool = False):
    """
    Push new parameters onto given device, or if none is provided, then all devices in the pipeline.
    """
    sequential = _sequential_override(sequential)
    for p in parameters:
        for name in p:
            p[name] = p[name].detach().value() if isinstance(p[name], Tensor) else detach_tensors([p[name]])[0]
    for i, d in enumerate(devices):
        _run_command(d, sequential, Command.update_device_parameters(parameters[i]))
        
def _get_loss_queue() -> Optional[queue.Queue]:
    ctx = get_current_context()
    if ctx is None or not _is_active() or not ctx.training:
        logger.warning("No active training run, no loss queue available.")
        return None
    return ctx.output_queue

def _get_checkpoint_queue() -> Optional[queue.Queue]:
    ctx = get_current_context()
    if ctx is None or not _is_active() or not ctx.training:
        logger.warning("No active training run, no checkpoint queue available.")
        return None
    return ctx.checkpoint_queue

def _get_intermediates_queue() -> Optional[queue.Queue]:
    ctx = get_current_context()
    if ctx is None or not _is_active():
        logger.warning("No active training run, no checkpoint queue available.")
        return None
    return ctx.intermediates_queue

def _sync():
    """
    Call sync on each device, block until response has been received.
    """
    ctx = get_current_context()
    if ctx is None:
        return # nothing to sync on
        
    logger.debug("PyBuda sync")

    sequential = len(ctx.processes) == 0
    devices = get_devices()
    for d in devices:
        _run_command(d, sequential, Command.sync(), response=True)
    
