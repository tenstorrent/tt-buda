# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import collections
from typing import Union, Tuple, List, Callable, Optional, Dict
import queue

import tensorflow as tf
import torch
import torch.multiprocessing as mp

from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Barrier as BarrierClass
from loguru import logger

from .device import Device
from .module import Module, PyTorchModule, TFModule
from .tensor import SomeTensor, buda_dataformat_to_pytorch_dtype, Tensor, to_pt_tensors, to_tf_variables, to_buda_tensors
from .verify import VerifyConfig
from .compile import CompilerConfig
from .pybudaglobal import lazy_trace_data
from .device_connector import DeviceConnector, TransferType, DirectPusherDeviceConnector
from .utils import detach_tensors

from pybuda.tvm_utils import map_tf_dtype_to_pt, map_pt_dtype_to_tf

from torch import nn

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if torch.is_tensor(input): 
                input = (input,)
            input = module(*input)
        return input

class CPUDevice(Device):
    """
    CPUDevice represents a CPU processor. It will spawn a process and run local operations on the assigned processor.
    """

    # Recorded as class variables since they don't need to be sent over to target process
    optimizer_f: Dict[Device, Callable] = {}
    scheduler_f: Dict[Device, Callable] = {}

    def __init__(self, 
        name: str, 
        optimizer_f: Callable = None,
        scheduler_f: Callable = None,
        mp_context = None,
        retain_backward_graph = False,
        module: Union[PyTorchModule, List[PyTorchModule]] = None,
        input_dtypes: List[torch.dtype] = None,
    ):
        """
        Create a CPU device with a given name. Optionally override Python multi-procesing context.

        Parameters
        ----------
        name: str
            Device name

        optimizer_f: Callable, optional
            Function that takes in a module and returns an optimizer. Required for training.

        scheduler_f: Callable, optional
            Function that takes in an optimizer, and returns one or more schedulers to step on each epoch.
            If None, no scheduler will be used during training.

        mp_context: mp.context, optional
            If provided, mp_context will be used to create multi-processing queues, instead of the default one

        module: Union[PyTorchModule, List[PyTorchModule]], optional
            Optionally place given module(s) one the device has been created

        """
        super().__init__(name, mp_context)
        self.sequential_module: Optional[torch.nn.Module] = None

        # record as class variables to avoid pickling and sending to target process
        CPUDevice.optimizer_f[self] = optimizer_f
        CPUDevice.scheduler_f[self] = scheduler_f

        self.optimizer: torch.optim.Optimizer = None
        self.schedulers: List[torch.optim.lr_scheduler._LRScheduler] = []

        self.retain_backward_graph = retain_backward_graph
        self.devtype = None
        self.device = "cpu"
        self.framework = None
        self.tf_grads = None
        self.tf_gradient_tape = None
        self.cpueval_tf_grads = None
        self.cpueval_tf_gradient_tape = None
        self.input_dtypes = input_dtypes

        if module is not None:
            if not isinstance(module, list):
                module = [module]
            for m in module:
                self.place_module(m)

        self._saved_fwd_data = None

    def __repr__(self):
        return f"CPUDevice '{self.name}'"
    
    def _initialize(self, 
            training: bool, 
            sequential: bool,
            final_barrier: Optional[BarrierClass] = None,
            shutdown_event: Optional[EventClass] = None,
            scale_loss: float = 1.0,
            checkpoint_interval: int = 0,
            perf_trace: bool = False):
        """
        Initialize the CPU device.

        Parameters
        ----------
        training: bool
            If true, create optimizer and schedulers for trainig, linking them to the modules on the device

        sequential: bool
            Set sequential/concurrent mode for this device

        final_barrier: mp.Event, optional
            If provided, forward will wait for the wait event before completing, allowing processes and queues to
            be alive until everything has completed.

        shutdown_event: mp.Event, optional
            If provided, forward will trigger the event in case of an exception, letting other processes know to
            shut down. This should always be set in concurrent mode.

        scale_loss: float, optional
            If this device is calculating loss, multiply the value with scale_loss after calculating it

        checkpoint_interval: int, optional
            The weights will be checkpointed into checkpoint queues on host every `checkpoint_interval` optimizer
            steps, if set to non-zero. Zero by default.

        perf_trace: bool, optional
            Ignored by CPU device
        """

        Device._initialize(self, sequential, final_barrier, shutdown_event)

        if not training:
            return # nothing to do here right now

        self._scale_loss = scale_loss

        if CPUDevice.optimizer_f[self] is None:
            logger.warning("Warning: no optimizer function provided for {}. No optimization will be done.", self)
        else:
            module = self._get_sequential().module
            params = module.parameters() if self.framework == "pytorch" else module.weights
            if len(list(params)) > 0:
                self.optimizer = CPUDevice.optimizer_f[self](module)
                if (self.optimizer is None or 
                    (isinstance(module, torch.nn.Module) ^ isinstance(self.optimizer, torch.optim.Optimizer)) or
                    (isinstance(module, (tf.keras.Model, tf.keras.layers.Layer)) ^ isinstance(self.optimizer, tf.keras.optimizers.legacy.SGD))
                ):
                    raise RuntimeError(f"Optimizer function for {self} didn't return a PyTorch optimizer")
            else:
                self.optimizer = None


        if self.optimizer is not None and CPUDevice.scheduler_f[self] is not None:
            schedulers = CPUDevice.scheduler_f[self](self.optimizer)
            if schedulers is not None:
                if isinstance(schedulers, (tuple, list)):
                    self.schedulers = list(schedulers)
                else:
                    self.schedulers = [schedulers]

                # TODO: Any reason we ever want multiple schedulers for a particular device? 
                # Maybe this should be refactored to just be one scheduler
                assert len(self.schedulers) == 1, "Only one scheduler per device is currently supported"
                for s in self.schedulers:
                    if not isinstance(s, torch.optim.lr_scheduler._LRScheduler):
                        raise RuntimeError(f"Schedule function for {self} returned a non-scheduler")

    def forward_pt(self, loop_count: int):
        """
        Run forward pass on each module on this device, in order

        Parameters
        ----------
        loop_count: int
            Number of micro-batches to run
        """

        logger.debug("Starting forward on {}", self)
        assert self._compiled, f"Module not compiled yet on {self}"

        if not self._training:
            self._modules_eval() # Set the module(s) to eval mode 

        try:
            for _ in range(loop_count):
                inputs = self.forward_input_dc.read()

                logger.trace("Forward inputs on {}:", self)
                lazy_trace_data(inputs)

                # Convert to pytorch tensors, if needed
                inputs = to_pt_tensors(inputs)
                torch_inputs = tuple(t.value() if isinstance(t, Tensor) else t for t in inputs)
                torch_inputs = tuple(t.to(self.device) for t in torch_inputs)
                for t in torch_inputs:
                    if t.requires_grad:
                        t.retain_grad()
                        
                if self.input_dtypes:
                    assert len(self.input_dtypes) == len(torch_inputs), f"CPUDevice input_dtypes specified, but differs in size from number of actual inputs. Types specified: {len(self.input_dtypes)}, num inputs: {len(torch_inputs)}"
                    torch_inputs = tuple(t.type(typ) for t, typ in zip(torch_inputs, self.input_dtypes))
                    torch_inputs = detach_tensors(torch_inputs)
                
                elif any(t.dtype in (torch.float16, torch.bfloat16) for t in torch_inputs):
                    torch_inputs = tuple(t.type(torch.float32) for t in torch_inputs)
                    torch_inputs = detach_tensors(torch_inputs)

                if self.loss_module is not None and len(self.modules) == 1:
                    outputs = torch_inputs
                else:
                    self._get_sequential().compilation = False
                    outputs: Tuple[SomeTensor] = self._modules_forward(*torch_inputs)

                if self.loss_module is None:
                    # Push data on to the output or next device
                    outputs = tuple(o.to('cpu') for o in outputs)
                    logger.trace("Forward outputs on {}:", self)
                    #lazy_trace_data(outputs)

                    detached_outputs = tuple(Tensor.create_from_torch(o).detach() for o in outputs)
                    self.forward_dc.push(detached_outputs)

                else:

                    # Calculate loss
                    targets = self.target_input_dc.read()
                    targets = tuple(t.to(self.device) for t in targets)
                    outputs = tuple(t.to(self.device) for t in outputs)

                    if len(outputs) == 1:
                        outputs = outputs[0]
                    if len(targets) == 1:
                        targets = targets[0]

                    lout = self.loss_module.forward(outputs, targets)
                    lout = self._scale_loss * lout
                    lout = [lout]

                    logger.info("Loss: {}", lout[0].item())

                    outputs = lout
                    if self.forward_dc:
                        self.forward_dc.push(tuple(l.item() for l in lout))

                if self._training:
                    if self._saved_fwd_data is None:
                        self._saved_fwd_data = queue.Queue() # local, no need for mp
                    self._saved_fwd_data.put((torch_inputs, outputs))

                self.forward_input_dc.pop()

            logger.debug("Ending forward on {}", self)

        except Exception as e:

            # Let other processes know to stop
            if self.shutdown_event is not None:
                self.shutdown_event.set()
            logger.debug("Ending forward due to exception on {}: {}", self, e)
            raise

    def forward_tf(self, loop_count: int):
        """
        Run forward pass on each module on this device, in order

        Parameters
        ----------
        loop_count: int
            Number of micro-batches to run
        """
        logger.debug("Starting forward on {}", self)
        assert self._compiled, f"Module not compiled yet on {self}"

        try:
            for _ in range(loop_count):
                inputs = self.forward_input_dc.read()

                logger.trace("Forward inputs on {}:", self)
                lazy_trace_data(inputs)
                inputs = to_tf_variables(inputs)
                outputs = inputs
                module = self._get_sequential()
                if self._training:
                    with tf.GradientTape(persistent=True) as tape:
                        [tape.watch(output) for output in outputs if output.trainable]
                        outputs = module.call(*outputs)
                else:
                    outputs = module.call(*outputs)

                if not isinstance(outputs, (list, tuple)):
                    outputs = (outputs, )
                detached_outputs = to_buda_tensors(outputs)
                logger.trace("Forward outputs on {}:", self)
                lazy_trace_data(detached_outputs)


                if self.loss_module is None:
                    # Push data on to the output or next device
                    self.forward_dc.push(detached_outputs)

                else:
                    assert False, "TODO"

                if self._training:
                    if self._saved_fwd_data is None:
                        self._saved_fwd_data = queue.Queue() # local, no need for mp
                    self.tf_gradient_tape = tape
                    self._saved_fwd_data.put((inputs, outputs,))

                self.forward_input_dc.pop()

            logger.debug("Ending forward on {}", self)

        except Exception as e:

            # Let other processes know to stop
            if self.shutdown_event is not None:
                self.shutdown_event.set()
            logger.debug("Ending forward due to exception on {}: {}", self, e)
            raise

    
    def forward(self, loop_count: int):
        """
        Run forward pass on each module on this device, in order

        Parameters
        ----------
        loop_count: int
            Number of micro-batches to run
        """
        #TODO Implement support for multiple subgraphs on cpu device
        if self.framework == "pytorch":
            forward_fn = self.forward_pt
        elif self.framework == "tensorflow":
            forward_fn = self.forward_tf
        else:
            raise RuntimeError(f"Unsupported framework: {self.framework}")
        
        return forward_fn(
            loop_count,
        )

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

        if self.framework == "pytorch":
            self.backward_pt(loop_count, zero_grad)
        elif self.framework == "tensorflow":
            self.backward_tf(loop_count, zero_grad)
        else:
            raise RuntimeError(f"Unsupported framework: {self.framework}")

    def backward_tf(self, loop_count: int, zero_grad: bool):
        logger.debug("Starting backward on {}", self)
        self._modules_train() # Set the module(s) to train mode

        try:
            module = self._get_sequential()
            # if zero_grad and self.optimizer:
            #     self.optimizer.zero_grad()  #  zero out the gradients

            for _ in range(loop_count):
                fwd_inputs, fwd_outputs = self._saved_fwd_data.get()

                if self.loss_module:
                    incoming_grad = fwd_outputs
                else:
                    bw_inputs = self.backward_input_dc.read()
                    # Convert to pytorch tensors, if needed
                    incoming_grad = to_tf_variables(bw_inputs)

                    self.backward_input_dc.pop()

                self.tf_grads = self.tf_gradient_tape.gradient(
                    fwd_outputs, 
                    module.module.trainable_variables,
                    output_gradients=incoming_grad)
                # what if it has multiple outputs?
                bw_inputs = self.tf_gradient_tape.gradient(
                    fwd_outputs,
                    [fwd_input for fwd_input in fwd_inputs if fwd_input.trainable],
                    output_gradients=incoming_grad)

                # Automatic device selection
                bw_inputs_on_cpu = tuple(Tensor.create_from_torch(torch.Tensor(bw_inp.numpy())) for bw_inp in bw_inputs if bw_inp is not None)

                logger.trace("Pushing bw inputs {} into {}", bw_inputs_on_cpu, type(self.backward_dc))
                self.backward_dc.push(bw_inputs_on_cpu)

                logger.trace("GRADIENT_CHECK: gradient out from {}: ", self)
                lazy_trace_data(bw_inputs_on_cpu)

            logger.debug("Ending backward on {}", self)

        except Exception as e:

            # Let other processes know to stop
            if self.shutdown_event is not None:
                self.shutdown_event.set()
            logger.debug("Ending backward due to exception on {}", self)
            raise


    def backward_pt(self, loop_count: int, zero_grad: bool):
        logger.debug("Starting backward on {}", self)
        self._modules_train() # Set the module(s) to train mode

        try:
            if zero_grad and self.optimizer:
                self.optimizer.zero_grad()  #  zero out the gradients

            for _ in range(loop_count):
                    
                fwd_inputs, fwd_outputs = self._saved_fwd_data.get()

                fwd_outputs = tuple(fwd_output.to(self.device) for fwd_output in fwd_outputs)

                if self.loss_module:

                    for l in fwd_outputs:
                        l.backward()

                else:
                    # Get inputs from the next stage, and run backward
                    bw_inputs = self.backward_input_dc.read()

                    # Convert to pytorch tensors, if needed
                    bw_inputs = tuple(t.value() if isinstance(t, Tensor) else t for t in bw_inputs)
                    bw_inputs = tuple(t.to(self.device) for t in bw_inputs)

                    logger.trace("GRADIENT_CHECK: bw_inputs into {}: ", self)
                    lazy_trace_data(bw_inputs)

                    req_grad_outs = [out for out in fwd_outputs if out.requires_grad]
                    for i, rec_out in enumerate(req_grad_outs):
                        if rec_out.requires_grad:
                            rec_out.backward(bw_inputs[i], retain_graph=self.retain_backward_graph or (i < len(req_grad_outs) - 1))

                    self.backward_input_dc.pop()
                    

                bw_inputs = tuple(fwd_input.grad for fwd_input in fwd_inputs if fwd_input.requires_grad or (fwd_input.grad_fn is not None))
                bw_inputs_on_cpu = tuple(t.to("cpu") for t in bw_inputs)
                logger.trace("Pushing bw inputs {} into {}", bw_inputs_on_cpu, type(self.backward_dc))
                self.backward_dc.push(bw_inputs_on_cpu)

                logger.trace("GRADIENT_CHECK: gradient out from {}: ", self)
                lazy_trace_data(bw_inputs_on_cpu)

            logger.debug("Ending backward on {}", self)

        except Exception as e:

            # Let other processes know to stop
            if self.shutdown_event is not None:
                self.shutdown_event.set()
            logger.debug("Ending backward due to exception on {}", self)
            raise

    def generate(self, loop_count: int, write_index: int):
        """
        Run forward pass on each module on this device, in order

        Parameters
        ----------
        loop_count: int
            Number of micro-batches to run

        """

        logger.debug("Starting generate on {}", self)
        assert self._compiled, f"Module not compiled yet on {self}"
        return self.forward(loop_count=loop_count)

    def compile_for_pt(self, 
            inputs: Tuple[Tensor, ...],
            compiler_cfg: CompilerConfig,
            targets: List[Tensor] = [],
            microbatch_size: int = 0,
            microbatch_count: int = 1,
            verify_cfg: Optional[VerifyConfig] = None,
            ) -> Tuple[Tensor, ...]:
        """
        For a CPU device, there is currently no compilation. This function propagates input shapes through the model
        to return output shapes and formats.

        Parameters
        ----------
        inputs: Tuple[Tensor, ...]
            Tuple of input tensors. They must have shape and format set, but do not need to hold data unless
            auto-verification is set.

        compiler_cfg: CompilerConfig
            Compiler configuration

        targets: List[Tensor], optional
            Optional list of target tensors, if this device has a loss module

        microbatch_size: int, optional
            The size of microbatch. Must be non-zero for training mode.

        microbatch_count: int
            Only relevant for training and TT devices.

        verify_cfg: Optional[VerifyConfig]
            Optional auto-verification of compile process

        Returns
        -------
        Tuple[Tensor, ...]
            Output tensors

        """
        assert not self._compiled, "Trying to compile a design that's already been compiled"

        training = compiler_cfg.enable_training
        Device.compile_for(self, training, microbatch_size, microbatch_count)

        if len(targets) > 0: # has loss module, only output is loss
            self._compiled = True
            return tuple([Tensor.create_from_torch(torch.tensor(1.0)).detach() for _ in targets])

        # Create inputs of the right shape and format, if needed
        torch_inputs = []
        for t in inputs:
            if t.has_value():
                torch_inputs.append(t.value())
            else:
                torch_inputs.append(torch.zeros(*t.shape.get_pytorch_shape(), dtype=buda_dataformat_to_pytorch_dtype(t.data_format)))

        torch_inputs = tuple(x.to(self.device) for x in torch_inputs)
        self._get_sequential().compilation = True
        outputs = self._modules_forward(*torch_inputs, targets=targets)
        outputs = [o if o.is_floating_point() else o.float() for o in outputs]
        outputs = tuple(x.to('cpu') for x in outputs)

        while isinstance(outputs[0], tuple):
            outputs = outputs[0]

        outputs = tuple(Tensor.create_from_torch(o).detach() for o in outputs)
        self._compiled = True
        return outputs

    def compile_for_tf(self, 
            inputs: Tuple[Tensor, ...],
            compiler_cfg: CompilerConfig,
            targets: List[Tensor] = [],
            microbatch_size: int = 0,
            verify_cfg: Optional[VerifyConfig] = None,
            ) -> Tuple[Tensor, ...]:
        """
        For a CPU device, there is currently no compilation. This function propagates input shapes through the model
        to return output shapes and formats.

        Parameters
        ----------
        inputs: Tuple[Tensor, ...]
            Tuple of input tensors. They must have shape and format set, but do not need to hold data unless
            auto-verification is set.

        compiler_cfg: CompilerConfig
            Compiler configuration

        targets: List[Tensor], optional
            Optional list of target tensors, if this device has a loss module

        microbatch_size: int, optional
            The size of microbatch. Must be non-zero for training mode.

        verify_cfg: Optional[VerifyConfig]
            Optional auto-verification of compile process

        Returns
        -------
        Tuple[Tensor, ...]
            Output tensors

        """
        assert not self._compiled, "Trying to compile a design that's already been compiled"

        training = compiler_cfg.enable_training
        Device.compile_for(self, training, microbatch_size)

        # Create inputs of the right shape and format, if needed
        tf_inputs = to_tf_variables(inputs)

        outputs = tf_inputs
        self._get_sequential().compilation = True
        outputs = self._modules_forward(*tf_inputs, targets=targets)

        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs, )

        outputs = to_buda_tensors(outputs)
        self._compiled = True
        return outputs

    def compile_for(self, 
            inputs: Tuple[Tensor, ...],
            compiler_cfg: CompilerConfig,
            targets: List[Tensor] = [],
            microbatch_size: int = 0,
            microbatch_count: int = 1,
            verify_cfg: Optional[VerifyConfig] = None,
            ) -> Tuple[Tensor, ...]:
        """
        For a CPU device, there is currently no compilation. This function propagates input shapes through the model
        to return output shapes and formats.

        Parameters
        ----------
        inputs: Tuple[Tensor, ...]
            Tuple of input tensors. They must have shape and format set, but do not need to hold data unless
            auto-verification is set.

        compiler_cfg: CompilerConfig
            Compiler configuration

        targets: List[Tensor], optional
            Optional list of target tensors, if this device has a loss module

        microbatch_size: int, optional
            The size of microbatch. Must be non-zero for training mode.

        microbatch_count: int
            Only relevant for training and TT devices.

        verify_cfg: Optional[VerifyConfig]
            Optional auto-verification of compile process

        Returns
        -------
        Tuple[Tensor, ...]
            Output tensors

        """
        assert not self._compiled, "Trying to compile a design that's already been compiled"

        if self.framework == "pytorch":
            compile_for_fn = self.compile_for_pt
        elif self.framework == "tensorflow":
            compile_for_fn = self.compile_for_tf
        else:
            raise RuntimeError(f"Unsupported framework: {self.framework}")
        
        return compile_for_fn(
            inputs,
            compiler_cfg,
            targets,
            microbatch_size,
            verify_cfg
        )

    def _get_sequential(self) -> Union[PyTorchModule, TFModule]:
        """
        Combine modules into one sequential module, if needed. Otherwise return one module.
        """
        contains_loss_module = 1 if self.loss_module else 0
        num_network_modules = len(self.modules) - contains_loss_module
        if self.sequential_module is None and num_network_modules > 1:
            od = collections.OrderedDict()
            for i, m in enumerate(self.modules):
                if m != self.loss_module:
                    od[m.name] = m.module

            if self.framework == "pytorch":
                self.sequential_module = PyTorchModule(self.name + "_sequential", mySequential(od))
            elif self.framework == "tensorflow":
                model = tf.keras.Sequential()
                for module in od.values():
                    model.add(module)
                self.sequential_module = TFModule(self.name + "_sequential", model)
            else:
                raise RuntimeError(f"Unsupported framework: {self.framework}")

        module = self.sequential_module if self.sequential_module is not None else self.modules[0]
        return module

    def _modules_forward(self, *args, targets=None) -> Tuple[torch.Tensor, ...]:
        """
        Run forward on all modules on device and return outputs
        """
        if len(self.modules) == 0:
            raise RuntimeError("Trying to run device with no modules")

        module = self._get_sequential()

        if self.framework == "pytorch":
            outputs: Tuple[SomeTensor] = module.forward(*args)
        elif self.framework == "tensorflow":
            tf_inputs = to_tf_variables(args)
            outputs: Tuple[SomeTensor] = module.call(*tf_inputs)
        else:
            raise RuntimeError(f"Unsupported framework: {self.framework}")
        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        return outputs

    def _modules_eval(self):
        """
        Set the module(s) to eval mode
        """
        self._get_sequential().module.eval()

    def _modules_train(self):
        """
        Set the module(s) to train mode
        """
        if self.loss_module:
            self._get_sequential().module.train()

    def update_device_parameters_pt(self, parameters: Dict[str, torch.Tensor]):
        self.sync() # wait until queued up commands have completed
        module: PyTorchModule = self._get_sequential()
        state_dict = module.module.state_dict()
        for p in parameters:
            if p not in state_dict:
                continue
            state_dict[p] = parameters[p]
        module.module.load_state_dict(state_dict)

    def update_device_parameters_tf(self, parameters: Dict[str, tf.Tensor]):
        self.sync() # wait until queued up commands have completed
        module: TFModule = self._get_sequential()
        # module.module.trainable_variables = parameters
        for param in module.module.trainable_variables:
            name = param.name
            param.assign(tf.convert_to_tensor(parameters[name].detach().numpy()))

    def update_device_parameters(self, parameters: Dict[str, torch.Tensor]):
        if self.framework == "pytorch":
            update_device_parameters_fn = self.update_device_parameters_pt
        elif self.framework == "tensorflow":
            update_device_parameters_fn = self.update_device_parameters_tf

        update_device_parameters_fn(parameters)

    def cpueval_forward_pt(self, inputs: List[torch.Tensor], parameters: Dict[str, torch.Tensor], save_for_backward: bool, targets: List[torch.Tensor] = []) -> List[torch.Tensor]:
        """
        Evaluate forward pass for verification

        Parameters
        ----------
        inputs: List[torch.Tensor]
            One input into the model (for each ordered input node)

        parameters: Dict[str, torch.Tensor]
            Map of model parameters

        save_for_backward: bool
            If set, input and output tensors will be saved so we can run the backward pass later.

        targets: List[torch.Tensor], optional
            If we're running training, and there's a loss module on this device, provide target

        Returns
        -------
        List[Tensor]
            Forward graph output
        """
        if len(self.modules) == 0:
            raise RuntimeError("Trying to run device with no modules")

        module: PyTorchModule = self._get_sequential()
        module.module = module.module.cpu()
        if not save_for_backward:
            module.module.eval() # disable dropout

        # Override parameters values
        if len(parameters) > 0:
            self.update_device_parameters_pt(parameters)

            # Copy back so that we can extract grad
            for name, p in module.module.named_parameters():
                parameters[name] = p

        if save_for_backward:
            self._saved_fw_inputs = inputs

        module.compilation = False
        outputs: Tuple[SomeTensor] = module.forward(*inputs)
        if self.loss_module:
            if len(outputs) == 1:
                outputs = outputs[0]
            if len(targets) == 1:
                targets = targets[0]
            lout = self.loss_module.forward(outputs, targets)
            lout = self._scale_loss * lout
            outputs = lout

        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        
        if save_for_backward:
            self._saved_fw_outputs = outputs

        module.module = module.module.to(self.device)

        return outputs

    def cpueval_forward_tf(self, inputs: List[torch.Tensor], parameters: Dict[str, torch.Tensor], save_for_backward: bool, targets: List[torch.Tensor] = []) -> List[torch.Tensor]:
        """
        Evaluate forward pass for verification

        Parameters
        ----------
        inputs: List[torch.Tensor]
            One input into the model (for each ordered input node)

        parameters: Dict[str, torch.Tensor]
            Map of model parameters

        save_for_backward: bool
            If set, input and output tensors will be saved so we can run the backward pass later.

        targets: List[torch.Tensor], optional
            If we're running training, and there's a loss module on this device, provide target

        Returns
        -------
        List[Tensor]
            Forward graph output
        """
        if len(self.modules) == 0:
            raise RuntimeError("Trying to run device with no modules")

        module: TFModule = self._get_sequential()

        # Override parameters values
        if len(parameters) > 0:
            # assert False, f"TODO"
            self.update_device_parameters_tf(parameters)

        inputs = to_tf_variables(inputs)
        if save_for_backward:
            self._saved_fw_inputs = inputs

        outputs = inputs
        if self._training:
            with tf.GradientTape(persistent=True) as tape:
                [tape.watch(output) for output in outputs if output.trainable]
                outputs = module.call(*outputs)

            self.cpueval_tf_gradient_tape = tape
        else:
            outputs = module.call(*outputs)
    
        if self.loss_module:
            if len(outputs) == 1:
                outputs = outputs[0]
            if len(targets) == 1:
                targets = targets[0]
            lout = self.loss_module.forward(outputs, targets)
            lout = self._scale_loss * lout
            outputs = lout

        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        
        if save_for_backward:
            self._saved_fw_outputs = outputs
        
        outputs = to_pt_tensors(outputs)
        return outputs

    def cpueval_forward(self, inputs: List[torch.Tensor], parameters: Dict[str, torch.Tensor], save_for_backward: bool, targets: List[torch.Tensor] = []) -> List[torch.Tensor]:
        """
        Evaluate forward pass for verification

        Parameters
        ----------
        inputs: List[torch.Tensor]
            One input into the model (for each ordered input node)

        parameters: Dict[str, torch.Tensor]
            Map of model parameters

        save_for_backward: bool
            If set, input and output tensors will be saved so we can run the backward pass later.

        targets: List[torch.Tensor], optional
            If we're running training, and there's a loss module on this device, provide target

        Returns
        -------
        List[Tensor]
            Forward graph output
        """
        if self.framework == "pytorch":
            cpueval_forward_fn = self.cpueval_forward_pt
        elif self.framework == "tensorflow":
            cpueval_forward_fn = self.cpueval_forward_tf
        else:
            raise RuntimeError(f"Unsupported framework: {self.framework}")

        return cpueval_forward_fn(
            inputs,
            parameters,
            save_for_backward,
            targets
        )

    def cpueval_backward_tf(self, bw_inputs: List[torch.Tensor], parameters: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:

        fw_outputs = self._saved_fw_outputs
        self._saved_fw_outputs = None
        fw_inputs = self._saved_fw_inputs
        self._saved_fw_inputs = None

        module = self._get_sequential()
        incoming_grad = to_tf_variables(bw_inputs)
        if self.loss_module:
            grads = self.cpueval_tf_gradient_tape.gradient(
                fw_outputs,
                module.module.trainable_variables
            )
            input_grads = self.cpueval_tf_gradient_tape.gradient(
                fw_outputs,
                [fw_input for fw_input in fw_inputs if fw_input.trainable],
            )
        else:
            grads = self.cpueval_tf_gradient_tape.gradient(
                fw_outputs,
                module.module.trainable_variables,
                output_gradients=incoming_grad)

            input_grads = self.cpueval_tf_gradient_tape.gradient(
                fw_outputs,
                [fw_input for fw_input in fw_inputs if fw_input.trainable],
                output_gradients=incoming_grad)
        self.cpueval_tf_grads = grads
        param_grads = {}

        for grad, param in zip(grads, module.module.trainable_variables):
            if isinstance(grad, tf.Tensor):
                param_grads[param.name] = torch.Tensor(grad.numpy())
            elif isinstance(grad, tf.IndexedSlices):
                param_grads[param.name] = torch.Tensor(tf.convert_to_tensor(grad).numpy())
            else:
                assert False, f"Hit unsupported gradient type {grad.__class__}"

        input_grads = [torch.Tensor(t.numpy()) for t in input_grads if t is not None]

        return input_grads, param_grads
    
    def cpueval_backward(self, bw_inputs: List[torch.Tensor], parameters: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.framework == "pytorch":
            cpueval_backward_fn = super().cpueval_backward
        elif self.framework == "tensorflow":
            cpueval_backward_fn = self.cpueval_backward_tf
        else:
            raise RuntimeError(f"Unsupported framework: {self.framework}")


        return cpueval_backward_fn(bw_inputs=bw_inputs, parameters=parameters)

    def place_module(self, module: Union[Module, Tuple[Module], List[Module]]):

        if not isinstance(module, (tuple, list)):
            module = (module,)

        for m in module:
            if isinstance(m, PyTorchModule):
                if self.framework is not None and self.framework != "pytorch":
                    raise RuntimeError("Cannot mix frameworks on a single CPUDevice")
                self.framework = "pytorch"
            elif isinstance(m, TFModule):
                if self.framework is not None and self.framework != "tensorflow":
                    raise RuntimeError("Cannot mix frameworks on a single CPUDevice")
                self.framework = "tensorflow"
            else:
                raise RuntimeError("Only PyTorch and TensorFlow modules can be placed on CPUDevices at this time.")

        Device.place_module(self, module)

    def _step_optimizer(self):
        """
        Step optimizer
        """
        if self.optimizer is None:
            return
        logger.debug("Stepping optimizer on {}", self)
        if self.framework == "tensorflow":
            assert self.tf_grads is not None
            self.optimizer.apply_gradients(zip(self.tf_grads, self._get_sequential().module.trainable_variables))
        elif self.framework == "pytorch":
            self.optimizer.step()
        else:
            assert False, f"Only support Pytorch and TF CPU device, got {self.framework}"


    def _step_schedulers(self):
        """
        Step schedulers
        """
        for s in self.schedulers:
            s.step()
            
    def pop_parameter_checkpoint(self) -> Dict:
        """
        Return a dictionary of current parameter values for the models on this device.
        """
        raise RuntimeError("Not supported by cpu device yet")

    def set_debug_gradient_trace_queue(self, q: queue.Queue):
        """
        [debug feature] Provide a queue to which incoming and outgoing gradients will be stored, for debug tracing.
        """
        self.debug_gradient_trace = q

    def _create_forward_device_connector(self, target_device: Union["TTDevice", "CPUDevice"], sequential: bool, d2d_fwd_queue: Optional[queue.Queue] = None, microbatch = 1):

        logger.debug("Creating forward device connector from {} to {}", self, target_device)
        if isinstance(target_device, CPUDevice):
            # Queues
            self.forward_dc = DeviceConnector(TransferType.MP_QUEUE, TransferType.MP_QUEUE, self.shutdown_event, side_queue=d2d_fwd_queue)
        else:
            # Tilize to TTDevice
            self.forward_dc = DirectPusherDeviceConnector(self.shutdown_event, sequential, side_queue=d2d_fwd_queue, microbatch=microbatch)

        target_device._set_forward_input_dc(self.forward_dc)

    def _create_backward_device_connector(self, target_device: Device, sequential: bool, d2d_bwd_queue: Optional[queue.Queue] = None, microbatch = 1):

        logger.debug("Creating backward device connector from {} to {}", self, target_device)
        if isinstance(target_device, CPUDevice):
            # Queues
            self.backward_dc = DeviceConnector(TransferType.MP_QUEUE, TransferType.MP_QUEUE, self.shutdown_event, side_queue=d2d_bwd_queue)
        else:
            # TTDevice copies directly to host, no pushing
            self.backward_dc = DirectPusherDeviceConnector(self.shutdown_event, sequential, side_queue=d2d_bwd_queue, microbatch=microbatch)

        target_device._set_backward_input_dc(self.backward_dc)

    # Create device connector for the last device, pushing forward
    def _create_forward_output_queue_device_connector(self, q: queue.Queue):
        logger.debug("Creating forward output queue connector on {}", self)
        self.forward_dc = DeviceConnector(TransferType.MP_QUEUE, TransferType.NONE, self.shutdown_event, q)

    # Create device connector for the first device, pushing backward
    def _create_backward_output_queue_device_connector(self, q: queue.Queue):
        logger.debug("Creating backward output queue connector on {}", self)
        self.backward_dc = DeviceConnector(TransferType.MP_QUEUE, TransferType.NONE, self.shutdown_event, q)

    # Create device connector for the first device, reading from a Queue
    def _create_input_queue_device_connector(self, q: queue.Queue, sequential: bool):
        logger.debug("Creating input queue connector on {}", self)
        self.forward_input_dc = DeviceConnector(TransferType.NONE, TransferType.MP_QUEUE, self.shutdown_event, q)

    # Create device connector for the last device, reading from a Queue
    def _create_target_queue_device_connector(self, q: queue.Queue, sequential: bool):
        logger.debug("Creating target queue connector on {}", self)
        self.target_input_dc = DeviceConnector(TransferType.NONE, TransferType.MP_QUEUE, self.shutdown_event, q)


    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor], lr = None) -> Optional[torch.optim.Optimizer]:
        if CPUDevice.optimizer_f[self] is None:
            return None
        if len(parameters) == 0:
            return None
        return CPUDevice.optimizer_f[self](list(parameters.values()))
    
    def get_pytorch_scheduler(self):
        if CPUDevice.scheduler_f[self] is None:
            return None
        
        return CPUDevice.scheduler_f[self](self.optimizer)

    def get_parameter_checkpoint(self) -> Dict[str, Tensor]:
        self.sync() # wait until queued up commands have completed
        ret = {}
        if self.framework == "pytorch":
            for name, p in self._get_sequential().module.named_parameters():
                ret[name] = Tensor.create_from_torch(p.cpu().data)
        elif self.framework == "tensorflow":
            for param in self._get_sequential().module.trainable_variables:
                name = param.name
                data = param.numpy()
                ret[name] = Tensor.create_from_torch(torch.Tensor(data))
        else:
            assert False, f"Only support Pytorch and TF CPU device, got {self.framework}"
        return ret

    def get_parameter_gradients(self) -> Dict[str, Tensor]:
        self.sync() # wait until queued up commands have completed
        return {} # TODO
    
    def get_device_intermediates(self) -> Dict[str, Tensor]:
        logger.warning("Fetching intermediate activations not supported on CPUDevice")
        return {}

    def sync(self):
        """
        Block until queued up commands have completed and the device is idle.
        """
        # TODO
        pass
