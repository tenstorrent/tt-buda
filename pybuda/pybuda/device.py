# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import atexit
import os
from contextlib import contextmanager
from typing import List, Tuple, Union, Optional, Dict, Any, Iterator
import queue
import threading

import torch
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Barrier as BarrierClass
from multiprocessing.synchronize import Lock as LockClass
import torch.multiprocessing as mp
from loguru import logger
from collections import OrderedDict, UserDict

from .module import Module
from .pybudaglobal import register_device, lazy_trace_data, set_state_changed, create_queue
from .tensor import Tensor, buda_dataformat_to_pytorch_dtype, remove_microbatch, to_pt_tensors
from .device_connector import DeviceConnector
from pybuda._C.backend_api import initialize_child_process, finish_child_process
from pybuda._C.graph import RuntimeTensorTransform
from .utils import detach_tensors
from pybuda._C import DataFormat

class Device:
    """
    Device class represents a physical device which can be a Tenstorrent device, or a CPU. In a typical operation, 
    each device spawns a process on the host CPU which is either used to run commands on the CPU (if device is 
    a CPU), or feeds commands to the Tenstorrent device. 

    Each device will allocate input queues for the first module it will execute. On a CPU, these are usually
    some kind of multiprocessing queues with shared memory storage, and Tenstorrent devices have queues in 
    on-device memory.

    One or more Modules can be placed on the device to be executed.
    """

    def __init__(self, name: str, mp_context = None):
        """
        Create a device with a given name. Optionally override Python multi-procesing context.

        Parameters
        ----------
        name: str
            Device name

        mp_context: mp.context, Optional
            If provided, mp_context will be used to create multi-processing queues, instead of the default one
        """

        super().__init__()

        self.name: str = name
        self.modules: List[Module] = []
        self.loss_module: Module = None # optional loss module when last device in pipeline
        register_device(self)

        #
        # Input queues
        # These input queues are used by the CPU device, and TT model devices. On silicon devices,
        # the queues are in device memory, and these ones will not be used
        if mp_context is None:
            mp_context = mp.get_context('spawn')
        self.target_input_queue: queue.Queue = create_queue(mp_context)
        self.recompute_input_queue: queue.Queue = create_queue(mp_context)

        # First device needs to buffer inputs if they are pushed in before its been initialized and compiled.
        self._input_buffer = create_queue(mp_context)

        # Process control events, set in _initialize if needed
        self.shutdown_event : Optional[EventClass] = None
        self.final_barrier : Optional[BarrierClass] = None

        # Main process command queue
        self.command_queue: queue.Queue = create_queue(mp_context)
        self._command_queue_resp: queue.Queue = create_queue(mp_context)

        # Flag indicating that we still need to compile this device
        self._compiled = False
        self._compile_output : "CompileResults" = []

        # Device runs in same process as all others, running one at a time
        self._sequential = True

        self._first_device = True # to be cleared on non-first device when we start running

        # Save first input on the side to use for initial compilation.. this is because we can't peek
        # the first element of an mp queue
        self._first_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._first_targets: Optional[Tuple[torch.Tensor, ...]] = None

        self.forward_dc : DeviceConnector = None  # push through here
        self.backward_dc : DeviceConnector = None  # push backward through here
        self.forward_input_dc : DeviceConnector = None # read forward inputs through here
        self.backward_input_dc : DeviceConnector = None # read backward inputs through here
        self.target_input_dc: DeviceConnector = None
        self.intermediates_dc: DeviceConnector = None # read intermediate outputs through here
        self.dc_transfer_threads : Dict[str, Tuple[threading.Thread, queue.Queue]] = {}

        # cpueval forward/backward intermediates
        self._saved_fw_outputs = None
        self._saved_fw_inputs = None

        # If an automatic cpu fallback devices are generated, we may  need to forward inputs /outputs from this device 
        # to the newly created ones
        self.cpu_fallback_device_pre = None
        self.cpu_fallback_device_post = None

        # For generative inference, we want to keep track of the current token and tile indicies
        self._current_token_idx = -1
        self._current_tile_idx = -1

        # Store io queue information for multiple subgraphs
        self._io_queues = {}

    def _initialize(self,
            sequential: bool,
            final_barrier: Optional[BarrierClass],
            shutdown_event: Optional[EventClass]):
        """
        Setup steps before the workload starts.

        Parameters
        ----------
        sequential: bool
            Set sequential/concurrent mode for this device

        final_barrier: mp.Barrier, optional
            If provided, device process will wait for all other proceses to cross the Barrier, allowing 
            processes and queues to be alive until everything has completed.

        shutdown_event: mp.Event, optional
            If provided, forward will trigger the event in case of an exception, letting other processes know to
            shut down. This should always be set in concurrent mode.

        """
        self._sequential = sequential
        self.final_barrier = final_barrier
        self.shutdown_event = shutdown_event

    def place_module(self, module: Union[Module, Tuple[Module], List[Module]]):
        """
        Places a module, or list of modules, on this device for execution. Modules will be run as a sequential pipeline
        on this single device.

        Parameters
        ----------
        module: Union[Module, Tuple[Module], List[Module]]
            A single Module or a list of Modules to be placed on the device
        """

        def add(modules, module, device):
            for m in modules:
                if m.name == module.name:
                    raise RuntimeError("Module names should be unique for each module on device")
            modules.append(module)
            module._set_device(device)

        if isinstance(module, Module):
            add(self.modules, module, self)
        elif isinstance(module, (tuple, list)):
            for m in module:
                if not isinstance(m, Module):
                    raise RuntimeError("Expected a Module in the list, but got " + str(type(m)))
                add(self.modules, m, self)
        else:
            raise RuntimeError("Expected a Module or list of modules")
        set_state_changed()

    def remove_modules(self):
        self.modules = []
        set_state_changed()

    def place_loss_module(self, module: Module):
        """
        Places a module used to calculate loss on this device. This must be the last device in the pipeline.

        Parameters
        ----------
        module: Module
            A single loss module
        """

        if not isinstance(module, Module):
            raise RuntimeError("Expected a Module, but got " + str(type(module)))

        self.place_module(module)
        self.loss_module = module
        set_state_changed()

    def remove_loss_module(self):
        """
        Remove module used to calculate loss from this device
        """

        assert self.loss_module is not None

        self.modules.remove(self.loss_module)
        self.loss_module = None

    def push_to_inputs(self, *tensors: Union[Tuple[Union[torch.Tensor, Tensor], ...], Dict[str, Union[torch.Tensor, Tensor]]]):
        """
        Push tensor(s) to module inputs, either in order, or by keyword argumet if a dictionary is used. The data will be queued 
        up on the target device until it is ready to be consumed. 

        This call can block if there is no space on the target device's input queues.

        Parameters
        ----------
        *tensors: Union[torch.Tensor, Tensor]
            Ordered list of inputs to be pushed into the module's input queue. Can be pytorch or pybuda tensor.

        """
        if self.cpu_fallback_device_pre is not None:
            logger.info("push_to_inputs redirected from {} to {}", self, self.cpu_fallback_device_pre)
            return self.cpu_fallback_device_pre.push_to_inputs(*tensors)

        logger.trace("push_to_inputs on {}", self)
        if len(tensors) == 1 and isinstance(tensors[0], (tuple, list, dict, UserDict, OrderedDict)):
            # already grouped, break it up
            tensors = tensors[0]

        if isinstance(tensors, (dict, UserDict, OrderedDict)):
            [self.modules[0].input_names, tensors] = zip(*tensors.items())

        if self._first_inputs is None:
            self._first_inputs = tensors

        if ((self._first_inputs[0].shape)[0] != (tensors[0].shape)[0]):
            raise RuntimeError("Batch size mismatch between first input and current input")
        
        self._input_buffer.put(to_pt_tensors(tensors))

    def push_to_target_inputs(self, *tensors):
        """
        Push tensor(s) to module training target inputs, in order. The data will be queued up on the target 
        device until it is ready to be consumed. 

        This call can block if there is no space on the target device's input queues.

        Parameters
        ----------
        tensors
            Ordered list of inputs to be pushed into the module's target input queue
        """
        if self.cpu_fallback_device_post is not None:
            logger.info("push_to_target_inputs redirected from {} to {}", self, self.cpu_fallback_device_post)
            return self.cpu_fallback_device_post.push_to_target_inputs(*tensors)

        logger.trace("push_to_target_inputs on {}", self)
        if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
            # already grouped, break it up
            tensors = tensors[0]

        if self._first_targets is None:
            self._first_targets = tensors

        self.target_input_queue.put(tensors)

    def _get_target_inputs(self):
        """
        Get inputs from training target input queue to send to a module for processing. Blocking until data has 
        been received, or a shutdown event has been received by the process.

        Returns
        -------
        Optional[Tuple[SomeTensor]]
            A list of input tensors.  Will return None if process received a shutdown message.
        """

        return self._read_from_mp_queue(self.target_input_queue)

    def _get_recompute_inputs(self):
        """
        Get saved inputs from the previous forward pass. Blocking until data has been received, or a shutdown
        event has been received by the process.

        Returns
        -------
        Optional[Tuple[SomeTensor]]
            A list of input tensors.  Will return None if process received a shutdown message.
        """

        return self._read_from_mp_queue(self.recompute_input_queue)

    def push_to_command_queue(self, cmd):
        """
        Send command to the running main loop in another process
        """
        self.command_queue.put(cmd)

    def get_command_queue_response(self) -> Optional[Dict]:
        """
        Read from command queue response. This is blocking.

        Returns
        -------
        Optional[Dict]
            Command-specific dictionary with response data, or None in case of failures
        """
        while True:
            try:
                resp = self._command_queue_resp.get(timeout=0.1)
                break
            except queue.Empty as _:
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    logger.debug("Ending process on {} due to shutdown event", self)
                    if self.final_barrier is not None:
                        self.final_barrier.abort()
                    return # got a signal to shutdown and end the process
                continue
        return resp


    def _init_concurrent_run(self):
        """
        Callback before concurrent processes are launched
        """
        for dc in [self.forward_dc, self.backward_dc, self.forward_input_dc, self.backward_input_dc, self.target_input_dc, self.intermediates_dc]:
            if dc:
                dc.initialize()

    def _drain_queue(self, q: queue.Queue):
        """
        Drain and discard queue contents
        """
        while True:
            try:
                q.get(timeout=0.1)
                continue
            except queue.Empty as _:
                return 

    def get_next_command(self, command_queue: queue.Queue) -> Optional["Command"]:
        """
        Read next command to run, from the given command queue. Blocking.

        Parameters
        ----------
        command_queue: queue.Queue
            Queue of commands

        Returns
        -------
        Command
            Next command from the queue, or None if shutdown_even was set
        """

        while True:
            try:
                cmd = command_queue.get(timeout=1)
                logger.trace("{}: Got command from queue: {}", self, cmd)
                break
            except queue.Empty as _:
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    logger.debug("Ending process on {} due to shutdown event", self)
                    if self.final_barrier is not None:
                        self.final_barrier.abort()
                    self._drain_queue(command_queue)
                    return None # got a signal to shutdown and end the process
                continue
            except KeyboardInterrupt as _:
                logger.info("Keyboard interrupt detected on {}", self)
                if self.shutdown_event is not None:
                    self.shutdown_event.set()
                if self.final_barrier is not None:
                    self.final_barrier.abort()  # prevent deadlock on other processes
                self._drain_queue(command_queue)
                return None

        return cmd

    def push_command_response(self, resp: Dict[str, Any]):
        logger.trace("Pushing command response: {}", resp)
        self._command_queue_resp.put(resp)

    @contextmanager
    def _try_run(self, msg: str) -> Iterator[None]:
        """
        Wrapper around arbitrary code that catches exceptions and raises abort flags
        """
        try:
            yield
        except Exception as e:
            logger.error("{} error: {}", msg, e)
            import traceback
            print(traceback.format_exc())
            if self.shutdown_event is not None:
                self.shutdown_event.set()
            if self.final_barrier is not None:
                self.final_barrier.abort()  # prevent deadlock on other processes

            # Drain command queue
            self._drain_queue(self.command_queue)

    def run_next_command(self, cmd: "Command") -> bool:
        """
        In concurrent mode, this is called in a forever loop by the process dedicated to this device. 
        In sequential mode, the main process will call this until there's no more work to do.

        Parameters
        ----------
        command_queue: queue.Queue
            Command queue to read commands from

        Returns
        -------
        bool
            True if quit command was seen

        """
        from .run.commands import CommandType

        if cmd.command_type == CommandType.QUIT:
            logger.debug("Received SHUTDOWN command on {}", self)
            self._shutdown_threads()
            if self.final_barrier is not None:
                logger.debug("Waiting for barrier on {}", self)
                self.final_barrier.wait()
            logger.debug("Shutting down on {}", self)
            self.shutdown_device()
            return True # Done

        if cmd.command_type == CommandType.RUN_FORWARD:
            logger.debug("Received RUN_FORWARD command on {} / {}", self, os.getpid())
            with self._try_run("Forward"):
                self.forward(loop_count=cmd.params["loop_count"])

        elif cmd.command_type == CommandType.RUN_BACKWARD:
            logger.debug("Received RUN_BACKWARD command on {} / {}", self, os.getpid())
            with self._try_run("Backward"):
                self.backward(loop_count=cmd.params["loop_count"], zero_grad=cmd.params["zero_grad"])

        elif cmd.command_type == CommandType.RUN_GENERATE:
            logger.debug("Received RUN_GENERATE command on {} / {}", self, os.getpid())
            with self._try_run("Generate"):
                self.generate(loop_count=cmd.params["loop_count"], write_index=cmd.params["write_index"], tokens_per_iter=cmd.params["tokens_per_iter"], token_id=cmd.params["token_id"])

        elif cmd.command_type == CommandType.RUN_OPTIMIZER:
            logger.debug("Received RUN_OPTIMIZER command on {} / {}", self, os.getpid())
            with self._try_run("Optimizer"):
                self._step_optimizer()
            
        elif cmd.command_type == CommandType.RUN_SCHEDULERS:
            logger.debug("Received RUN_SCHEDULERS command on {} / {}", self, os.getpid())
    
            with self._try_run("Schedulers"):
                self._step_schedulers()

        elif cmd.command_type == CommandType.COMPILE:
            logger.debug("Received COMPILE command on {} / {}", self, os.getpid())
            logger.trace("Compile command: {}", cmd.params)
            try:
                ret = self.compile_for(
                        cmd.params["inputs"],
                        cmd.params["compiler_cfg"],
                        cmd.params["targets"],
                        cmd.params["microbatch_size"],
                        cmd.params["microbatch_count"],
                        cmd.params["verify_cfg"])
                        
                self.push_command_response({"outputs": ret})
            except Exception as e:
                import traceback
                logger.error("Compile error: {}\n{}", e, traceback.format_exc())
                self.push_command_response(e)

        elif cmd.command_type == CommandType.GET_QUEUES:
            assert "queue_type" in cmd.params
            (
                queues,
                tile_broadcast_dims,
                original_shapes,
                requires_grad,
                runtime_tensor_transforms,
                constant_inputs,
                tile_dims,
            ) = self.get_dram_io_queues(cmd.params["queue_type"])
            self.push_command_response(
                    {
                        "queues": queues, 
                        "tile_broadcast_dims": tile_broadcast_dims, 
                        "original_shapes": original_shapes,
                        "requires_grad": requires_grad,
                        "runtime_tensor_transforms": runtime_tensor_transforms,
                        "constant_inputs": constant_inputs,
                        "tile_dims": tile_dims,
                    })

        elif cmd.command_type == CommandType.SET_QUEUES:
            logger.trace("Set DRAM IO queues on {}: {}", self, cmd.params['direction'])
            self.set_dram_io_queues(
                    cmd.params["direction"], 
                    cmd.params["queues"], 
                    cmd.params["tile_broadcast_dims"], 
                    cmd.params["original_shapes"],
                    cmd.params["requires_grad"],
                    cmd.params["runtime_tensor_transforms"],
                    cmd.params["constant_inputs"],
                    cmd.params["tile_dims"])

        elif cmd.command_type == CommandType.DC_TRANSFER:
            logger.trace("DC Transfer on {}: {}", self, cmd.params['direction'])
            direction = cmd.params["direction"]
            if self._sequential:
                self.dc_transfer(direction)
            else:
                # Push to thread and move on
                if direction not in self.dc_transfer_threads:
                    # Start a new thread
                    dir_q = queue.Queue()
                    self.dc_transfer_threads[direction] = (
                            threading.Thread(target=self.dc_transfer_thread, args=(direction, dir_q)),
                            dir_q)
                    self.dc_transfer_threads[direction][0].start()

                self.dc_transfer_threads[direction][1].put(direction)

        elif cmd.command_type == CommandType.CPUEVAL_FORWARD:
            logger.trace("CPUEVAL_FORWARD on {}", self)
            ret = self.cpueval_forward(cmd.params["inputs"], cmd.params["parameters"], cmd.params["save_for_backward"], cmd.params["targets"])
            ret = detach_tensors(ret)
            self.push_command_response({"result": ret})

        elif cmd.command_type == CommandType.CPUEVAL_BACKWARD:
            logger.trace("CPUEVAL_BACKWARD on {}", self)
            input_grads, params_grads = self.cpueval_backward(
                    cmd.params["bw_inputs"], 
                    cmd.params["parameters"])
            self.push_command_response({"input_grads": input_grads, "params_grads": params_grads})

        elif cmd.command_type == CommandType.CPUEVAL_LOSS:
            logger.trace("CPUEVAL_LOSS on {}", self)
            ret = self.cpueval_loss(cmd.params["fw_outputs"], cmd.params["targets"], cmd.params["scale_loss"])
            #ret = tuple(t.detach() for t in ret)
            self.push_command_response({"loss": ret})


        elif cmd.command_type == CommandType.GET_PARAMETER_CHECKPOINT:
            logger.trace("GET_PARAMETER_CHECKPOINT on {}", self)
            self.push_command_response({"checkpoint": self.get_parameter_checkpoint()})

        elif cmd.command_type == CommandType.GET_PARAMETER_GRADIENTS:
            logger.trace("GET_PARAMETER_GRADIENTS on {}", self)
            self.push_command_response({"gradients": self.get_parameter_gradients()})

        elif cmd.command_type == CommandType.UPDATE_DEVICE_PARAMETERS:
            logger.trace("UPDATE_DEVICE_PARAMETERS on {}", self)
            self.update_device_parameters(cmd.params["parameters"])

        elif cmd.command_type == CommandType.SYNC:
            logger.trace("SYNC on {}", self)
            self.sync()
            self.push_command_response({"sync": True})

        else:
            raise RuntimeError("Unknown command received by ", self)

        return False

    def dc_transfer_thread(self, direction: str, direction_queue: queue.Queue):
        """
        Keep transfering data in a thread. One per direction.
        """
        while True:
            try:
                cmd = direction_queue.get(timeout=0.1)
                logger.trace("DC transfer thread {} got cmd={}", direction, cmd)
                if cmd == "quit":
                    return

                assert cmd == direction
                self.dc_transfer(direction)
                
            except queue.Empty as _:
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    logger.debug("Ending dc transfer thread {} on {} due to shutdown event", direction, self)
                    return 
                continue

    def dc_transfer(self, direction: str):
        """
        Transfer data between devices
        """
        if direction == "forward":
            if self.forward_dc is not None:
                self.forward_dc.transfer(blocking=True)
        elif direction == "forward_input":
            self.forward_input_dc.transfer(blocking=True)
        elif direction == "target":
            self.target_input_dc.transfer(blocking=True)
        elif direction == "backward":
            self.backward_dc.transfer(blocking=True)
        elif direction == "intermediates":
            if self.intermediates_dc is not None:
                self.intermediates_dc.transfer(blocking=True)
        else:
            raise RuntimeError(f"Invalid direction: {direction}")

    
    def set_dram_io_queues(self, direction: str, queues: List["DramIODesc"], tile_broadcast_dims: Optional[List[List[int]]] = None, 
            original_shapes: Optional[List[Tuple[int, ...]]] = None, requires_grad: Optional[List[bool]] = None,
            runtime_tensor_transforms: Optional[List[RuntimeTensorTransform]] = None, constant_inputs: Optional[List[bool]] = None,
            tile_dims: Optional[List[List[int]]] = None):

        if direction == "forward_in":
            assert original_shapes is not None
            assert requires_grad is not None
            self._io_queues[direction] = {"queues" : queues, "original_shapes" : original_shapes, "requires_grad" : requires_grad, "runtime_tensor_transforms" : runtime_tensor_transforms}
            self.forward_input_dc.set_dram_io_pop_queues(queues, original_shapes, requires_grad, runtime_tensor_transforms)

        elif direction == "forward_in_push":
            assert tile_broadcast_dims is not None
            self._io_queues[direction] = {
                "queues" : queues, "tile_broadcast_dims" : tile_broadcast_dims, "runtime_tensor_transforms" : runtime_tensor_transforms, "constant_inputs" : constant_inputs, "tile_dims" : tile_dims}
            self.forward_input_dc.set_dram_io_push_queues(queues, tile_broadcast_dims, runtime_tensor_transforms, constant_inputs, tile_dims)

        elif direction == "forward_out":
            assert tile_broadcast_dims is not None
            self._io_queues[direction] = {"queues" : queues, "tile_broadcast_dims" : tile_broadcast_dims, "runtime_tensor_transforms" : runtime_tensor_transforms}
            self.forward_dc.set_dram_io_push_queues(queues, tile_broadcast_dims, runtime_tensor_transforms)

        elif direction == "forward_out_pop":
            assert original_shapes is not None
            assert requires_grad is not None
            self._io_queues[direction] = {"queues" : queues, "original_shapes" : original_shapes, "requires_grad" : requires_grad, "runtime_tensor_transforms" : runtime_tensor_transforms}
            self.forward_dc.set_dram_io_pop_queues(queues, original_shapes, requires_grad, runtime_tensor_transforms)

        elif direction == "backward_in":
            assert original_shapes is not None
            assert requires_grad is not None
            self._io_queues[direction] = {"queues" : queues, "original_shapes" : original_shapes, "requires_grad" : requires_grad, "runtime_tensor_transforms" : runtime_tensor_transforms}
            self.backward_input_dc.set_dram_io_pop_queues(queues, original_shapes, requires_grad, runtime_tensor_transforms)

        elif direction == "backward_out":
            assert original_shapes is not None
            assert requires_grad is not None
            self._io_queues[direction] = {"queues" : queues, "original_shapes" : original_shapes, "requires_grad" : requires_grad, "runtime_tensor_transforms" : runtime_tensor_transforms}
            self.backward_dc.set_dram_io_pop_queues(queues, original_shapes, requires_grad, runtime_tensor_transforms)

        elif direction == "backward_out_push":
            assert tile_broadcast_dims is not None
            self._io_queues[direction] = {"queues" : queues, "tile_broadcast_dims" : tile_broadcast_dims, "runtime_tensor_transforms" : runtime_tensor_transforms}
            self.backward_dc.set_dram_io_push_queues(queues, tile_broadcast_dims, runtime_tensor_transforms)

        elif direction == "target_in_push":
            assert tile_broadcast_dims is not None
            self._io_queues[direction] = {"queues" : queues, "tile_broadcast_dims" : tile_broadcast_dims, "runtime_tensor_transforms" : runtime_tensor_transforms}
            self.target_input_dc.set_dram_io_push_queues(queues, tile_broadcast_dims, runtime_tensor_transforms)

        elif direction == "intermediates_pop":
            assert original_shapes is not None
            assert requires_grad is not None
            self._io_queues[direction] = {"queues" : queues, "original_shapes" : original_shapes, "requires_grad" : requires_grad, "runtime_tensor_transforms" : runtime_tensor_transforms}
            self.intermediates_dc.set_dram_io_pop_queues(queues, original_shapes, requires_grad, runtime_tensor_transforms)

        else:
            raise RuntimeError("Unknown direction")



    def get_dram_io_queues(self, queue_type: str) -> List["DramIODesc"]:
        raise RuntimeError("Only TTDevice implements get_dram_io_queues")

    def run(self, output_dir: str):
        """
        Main process loop in concurrent mode. 
        
        The loop receives commands through its command queue, which indicate how many epochs & iterations to
        run, whether to run training or inference, and position in the pipeline. 

        The loop will run until shutdown command is sent in the command queue, or shutdown event is raised due
        to an exception in another process

        Parameters
        ----------

        output_dir: str
            Output directory needed by perf trace on every process
        """

        atexit.register(atexit_handler, devices=(self,))
        self._init_concurrent_run()
        initialize_child_process(output_dir)

        try:
            while True:
                cmd = self.get_next_command(self.command_queue)
                if cmd is None:
                    break
                done = self.run_next_command(cmd)
                if done:
                    break

        except Exception as e:
            if self.shutdown_event is not None:
                self.shutdown_event.set()
            if self.final_barrier is not None:
                self.final_barrier.abort()  # prevent deadlock on other processes
            logger.debug("Ending process on {} due to exception: {}", self, e)
            self.shutdown_device()
            raise

        #finally:
        #    # TODO: this should only be done if we're really done... in concurrent mode, we should
        #    # keep processes alive
        #    self.shutdown_device()

    def compile_for(self, training: bool, microbatch_size: int = 0, microbatch_count: int = 1):
        """ 
        Save microbatch size and count
        """
        self._microbatch_size = microbatch_size
        self._microbatch_count = microbatch_count
        self._training = training

    def _get_first_tensors(self, first_tensors: Tuple[Tensor, ...]) -> Tuple[int, Tuple[Tensor, ...] ]:

        # detect microbatch size
        def get_microbatch_size(t):
            return t.shape[0]

        first_inputs = first_tensors
        microbatch_size = get_microbatch_size(first_inputs[0])

        for input in first_inputs:
            mb_size = get_microbatch_size(input)
            if (mb_size != microbatch_size) and (mb_size != 1):
                raise RuntimeError("Microbatch size doesn't match for all inputs")

        out_first_inputs = remove_microbatch(first_inputs)
        return microbatch_size, out_first_inputs

    def get_first_targets(self) -> List[Tensor]:
        """
        Return the tuple of first targets pushed to this device
        """
        if self._first_targets is None:
            raise RuntimeError("Targets must be pushed into the last device before trying to compile the model.")

        _, first_targets = self._get_first_tensors(self._first_targets)

        return first_targets

    def get_first_inputs(self, peek=False) -> Tuple[int, Tuple[Tensor, ...] ]:
        """
        Return the microbatch size, and first input in microbatch pushed into the device. If input_shapes/input_types
        are provided, then those will be used to create input tensors.

        This is used to compile and optimize the model for dimensions provided by the first input.
        """
        if self._first_inputs is None:
            if peek:
                return None, None
            raise RuntimeError("Inputs must be pushed into the first device before trying to compile the model.")

        microbatch_size, first_inputs = self._get_first_tensors(self._first_inputs)
        if not peek:
            self._first_inputs = None # release

        return microbatch_size, first_inputs

    def shutdown_device(self):
        """
        Check for any mp queues that are not empty, and drain them
        """
        if self._sequential:
            return # notthing to clean up here if we're not multi-processing

        finish_child_process()

    def _read_from_mp_queue(self, q: queue.Queue):
        """
        Read from mp queue, and abort if shutdown event has been received by the process.

        Returns
        -------
        Any
            Data from the queue, or None if aborted
        """

        while True:
            try:
                out = q.get(timeout=0.1)
                break
            except queue.Empty as _:
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    logger.trace("_read_from_mp_queue aborting on {}", self)
                    return None # got a signal to shutdown and end the process
                continue
        
        return out

    def _shutdown_threads(self):

        for d in self.dc_transfer_threads.values():
            d[1].put("quit")
            d[0].join()
        self.dc_transfer_threads = {} # clear threads

        for dc in [self.forward_dc, self.backward_dc, self.forward_input_dc, self.backward_input_dc, self.target_input_dc]:
            if dc:
                dc.shutdown()

    # Device connector for forward inputs
    def _set_forward_input_dc(self, dc: DeviceConnector):
        self.forward_input_dc = dc

    # Device connector for backward inputs
    def _set_backward_input_dc(self, dc: DeviceConnector):
        self.backward_input_dc = dc

    def cpueval_backward(self, 
            bw_inputs: List[torch.Tensor],
            parameters: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Evaluate backward pass for verification. `cpueval_forward` should've been called first, with 
        `save_for_backward` set.

        Parameters
        ----------
        bw_inputs: List[torch.Tensor]
            BW inputs, i.e. losses for each fw output

        parameters: Dict[str, torch.Tensor]
            Module parameters

        Returns
        -------
        List[Tensor]
            Gradients on ordered inputs

        Dict[str, Tensor]
            Gradients on parameters
        """

        assert self._saved_fw_inputs is not None, "cpueval_forward has not been called with save_for_backward"
        assert self._saved_fw_outputs is not None, "cpueval_forward has not been called with save_for_backward"

        fw_outputs = self._saved_fw_outputs
        self._saved_fw_outputs = None
        fw_inputs = self._saved_fw_inputs
        self._saved_fw_inputs = None

        fw_outputs = [t for t in fw_outputs if t.requires_grad]
        
        if self.loss_module:
            for fw_output in fw_outputs:
                fw_output.backward()
        else:
            assert len(bw_inputs) == len(fw_outputs)

            for i, (bw_input, fw_output) in enumerate(zip(bw_inputs, fw_outputs)):
                fw_output.backward(bw_input, retain_graph=(i < len(bw_inputs) - 1))

        param_grads = {name : value.grad.clone() for name, value in parameters.items() if value.requires_grad}
        input_grads = [t.grad for t in fw_inputs if t.requires_grad]

        return input_grads, param_grads

    def generate(self, loop_count: int, write_index: int):
        """
        Run generate forward pass on each module on this device, in order

        Parameters
        ----------
        loop_count: int
            Number of micro-batches to run

        write_index: int
            Write location for past cache buffers

        """
        raise RuntimeError("Children should implement this")

    def forward(self, loop_count: int):
        """
        Run forward pass on each module on this device, in order

        Parameters
        ----------
        loop_count: int
            Number of micro-batches to run
        """
        raise RuntimeError("Children should implement this")

    def backward(self, loop_count: int, zero_grad: bool):
        """
        Run backward pass on each module on this device, in reverse order

        Parameters
        ----------
        loop_count: int
            Each mini-batch is broken into micro-batches. This is necessary to fill a multi-device pipeline, 
            and should be roughly 4-6x the number of devices in the pipeline for ideal performance.

        zero_grad: bool
            Set to true to have optimizer zero out gradients before the run
        """
        raise RuntimeError("Children should implement this")

    def _step_optimizer(self):
        """
        Step optimizer
        """
        raise RuntimeError("Children should implement this")
    
    def _step_schedulers(self):
        """
        Step schedulers
        """
        raise RuntimeError("Child should implement this")

def atexit_handler(devices: Tuple[Optional[Device], ...]):
    """
    Shutdown the device on process exit (if not handled cleanly already)
    """
    logger.debug("atexit handler called for {}", devices)
    for d in devices:
        if d is not None:
            d.shutdown_device()
    logger.debug("atexit handler completed")
