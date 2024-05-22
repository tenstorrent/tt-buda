# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List, Tuple, Union, Dict, Set
from collections import deque
import os
import queue
import inspect
import copy

import torch
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Barrier as BarrierClass
from loguru import logger

from .device import Device
from .pybudaglobal import PYBUDA_DEVMODE, lazy_trace_data, is_silicon, profiler, state_changed, set_state_changed, clear_state_changed, start_tracing, stop_tracing, reset_unique_node_id
from .module import Module, PyBudaModule
from .tensor import Tensor, to_pt_tensors, remove_microbatch, consteval_input_bw, consteval_shape
from .parameter import Parameter
from .optimizers import Optimizer
from .schedulers import LearningRateScheduler
from .utils import budabackend_path

from pybuda._C.graph import Graph, create_op_node, create_data_edge, create_parameter_input, create_activation_input, create_output, create_constant_input, create_target_input, add_partial_datacopy_edge, RuntimeTensorTransform, RuntimeTensorTransformType, Shape, OpType
from pybuda._C.graph import eval as graph_eval
from pybuda._C import DataFormat
from pybuda.tvm_utils import flatten_inputs

from .pybudaglobal import TILE_DIM, create_queue
from .verify import VerifyConfig
from .config import CompilerConfig, _get_global_compiler_config
from .backend import BackendAPI, BackendCompileException
from pybuda._C.backend_api import BackendDevice, BackendType, DeviceMode, StrideDescriptor, DramIODesc, DeviceConfig, get_device_descs_for_available_devices, get_custom_device_desc, get_device_cluster_yaml
from .device_connector import (
    DeviceConnector, 
    TransferType, 
    DirectPopperDeviceConnector, 
    OutputQueueDirectPoppperDeviceConnector, 
    InputQueueDirectPusherDeviceConnector,
    DirectPusherPopperDeviceConnector)


class TTDevice(Device):
    """
    TTDevice represents one or more Tenstorrent devices that will receive modules to run.
    """

    def __init__(self, 
            name: str, 
            num_chips: int = None,
            chip_ids: Union[List[int], List[Tuple[int]]] = None,
            arch: Optional[BackendDevice] = None,
            devtype: Optional[BackendType] = None, 
            device_mode: Optional[DeviceMode] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[LearningRateScheduler] = None,
            fp32_fallback: DataFormat = DataFormat.Float16_b,
            mp_context = None,
            module: Union[Module, List[Module]] = None,
    ):
        """
        Initialize a new Tenstorrent device.

        For development and debug purposes, device model or golden-model can be used instead of silicon. Device
        model is the default in development mode.

        Parameters
        ----------
        name: str
            Device name

        num_chips: int, optional
            On a system with multiple Tenstorrent silicon devices available, one TTDevice can span more than one chip by setting this parameter to more than 1.
            This allows a larger model to be spread over multiple chips. 

            To take all available devices, set num_chips to 0. 

        chip_ids: Union[List[int], List[Tuple[int]]], optional
            By default, TTDevice will allocate the first available set of chips. If the application requires a particular chip, or set of chips, to be used,
            chip_ids allows the user to pick particular ones.
            The user can directly provide the chip_ids or the coordinates of the chips on Nebula/Galaxy systems.

        arch: BackendDevice, optional
            Which Tenstorrent chip arch (GRAYSKULL, WORMHOLE etc.)

        devtype: BackendType, optional
            Type of Tenstorrent device. Only used to run testing on models of devices instead of a real silicon chip.

        optimizer: Optimizer, optional
            PyBuda optimizer to be used on this device. Mandatory if running in training mode.

        fp32_fallback: DataFormat
            If TT device doesn't support FP32, tensors will fall back to this format. Bfloat16 is the default.

        mp_context: mp.context, optional
            Optioanlly override Python multi-processing context when creating mp queues

        module: Union[Module, List[Module]], optional
            Optionally place given module(s) one the device has been created
        """
        super().__init__(name, mp_context)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fp32_fallback = fp32_fallback

        self._checkpoint_interval = 0
        self._unused_parameters = set() # populated during `self.generate_graph`; records unused params

        #self._perf_dump_mode: buda.PerfDumpMode = buda.PerfDumpMode.Disable
        #self._perf_desc: Optional[buda.PerfDesc] = None

        self.backend_api : Optional[BackendAPI] = None

        self.devtype = (BackendType.Golden if PYBUDA_DEVMODE() else BackendType.Silicon) if devtype is None else devtype
        if self.devtype != BackendType.Silicon:
            if "GOLDEN_WORMHOLE_B0" in os.environ:
                if arch is None:
                    arch = BackendDevice.Wormhole_B0
            elif "PYBUDA_GOLDEN_BLACKHOLE" in os.environ:
                if arch is None:
                    arch = BackendDevice.Blackhole
            else:
                if arch is None:
                    arch = BackendDevice.Grayskull

        # chip_ids        num_chips       chip_ids used
        # ---------------------------------------------
        # None or []      None            [0]
        # None or []      0               [0..num_devices_detected-1]
        # None or []      2               [0,1]
        # [0,1]           None or 2       [0,1]
        # [0,1]           not 2           error
        
        if chip_ids is None or len(chip_ids) == 0:
            if num_chips is None:
                self.chip_ids = [0]
                self.num_chips = 1
            else:
                self.chip_ids = list(range(num_chips))
                self.num_chips = num_chips
        else:
                assert num_chips is None or len(chip_ids) == num_chips, f"num_chips:{num_chips} does not match chip_ids:{chip_ids}"
                self.chip_ids = chip_ids
                self.num_chips = len(chip_ids)

        self.arch = arch

        # device_mode is modified when we execute device save/load
        self.device_mode = DeviceMode.CompileAndRun if device_mode is None else device_mode

        self.graph = None
        self.intermediate_tensors = {}
        self.compiled_netlists = []
        self.allocated_blocks = []
        self.current_host_address = 0
        self._active_subgraph = 0
        reset_unique_node_id()

        if module is not None:
            if not isinstance(module, list):
                module = [module]
            for m in module:
                self.place_module(m)

    def __repr__(self):
        return f"TTDevice '{self.name}'"

    def get_device_config(self, compiler_cfg=None) -> DeviceConfig:
        """
        Figure out which silicon devices will be used, if in silicon mode
        """

        # Call get_device_config here without device.yaml if:
        # (1) it's running on compute machine with targeting Golden backend (devtype = Golden)
        # (2) it's running on compute machine but targeting Silicon device (eg. generating TTI on compute machine)
        # For the following case, get_device_config is not called here, but later with device.yaml obtained from backend
        # (3) it's running on Silicon machine with setting device-mode to CompileOnly (eg. generating TTI on silion machine)
        device_descs = get_device_descs_for_available_devices(compiler_cfg.backend_output_dir) # possibly modify here
        harvesting_mask = compiler_cfg.harvesting_mask
        if self.devtype != BackendType.Silicon or (self.device_mode == DeviceMode.CompileOnly and len(device_descs) == 0):
            assert self.arch is not None, "Unknown arch for non-silicon compile"

            default_device_desc = get_custom_device_desc(self.arch, mmio=True, harvesting_mask=harvesting_mask, out_dir=compiler_cfg.backend_output_dir)
            return get_device_config(self.arch,
                                     self.chip_ids,
                                     compiler_cfg.backend_cluster_descriptor_path,
                                     compiler_cfg.backend_runtime_params_path,
                                     compiler_cfg.store_backend_db_to_yaml,
                                     self.devtype,
                                     default_device_desc.soc_desc_yaml,
                                     backend_output_dir=compiler_cfg.backend_output_dir,
                                     backend_device_descriptor_path_override=compiler_cfg.backend_device_descriptor_path,
                                     harvesting_mask=harvesting_mask)

        device_list = [d.arch for d in device_descs if d.mmio]
        if len(device_list) == 0:
            raise RuntimeError("No Tenstorrent devices present.")

        for desc in device_descs:
            assert desc.arch == device_descs[0].arch, f"Device {desc.arch} architecture doesn't match the system"
        detected_arch = device_list[0]

        if self.arch:
            assert detected_arch == self.arch, f"User constructed a TTDevice of {self.arch} but detected: {detected_arch}"
        self.arch = detected_arch

        # Pick chips ids based on the arch
        if len(self.chip_ids) == 0:
            self.num_chips = len(device_list)
            self.chip_ids = list(range(self.num_chips))

        first_id = 0
        # if PYBUDA_NEBULA_GALAXY_PLACER is specified, use soc_desc of unharvested_chip
        if "PYBUDA_NEBULA_GALAXY_PLACER" in os.environ:
            for device_id, desc in enumerate(device_descs):
                if desc.harvesting_mask == 0:
                    first_id = device_id
                    break

        soc_desc = device_descs[first_id].soc_desc_yaml
        cluster_yaml = get_device_cluster_yaml(compiler_cfg.backend_output_dir) if compiler_cfg.backend_cluster_descriptor_path == "" else compiler_cfg.backend_cluster_descriptor_path
        dev_cfg = get_device_config(self.arch,
                      chip_ids=self.chip_ids,
                      backend_cluster_descriptor_path=cluster_yaml,
                      backend_runtime_params_path=compiler_cfg.backend_runtime_params_path,
                      store_backend_db_to_yaml=compiler_cfg.store_backend_db_to_yaml,
                      backend_type=self.devtype,
                      device_yaml=soc_desc,
                      backend_output_dir=compiler_cfg.backend_output_dir,
                      backend_device_descriptor_path_override=compiler_cfg.backend_device_descriptor_path,
                      harvesting_mask=harvesting_mask)

        if "PYBUDA_FORCE_EMULATE_HARVESTED" in os.environ and dev_cfg.grid_size.r == 10: # non-harvested
            if self.arch == BackendDevice.Wormhole_B0:
                harvesting_mask = 2048
            else:
                harvesting_mask = 2050
            dev_cfg = get_device_config(self.arch,
                        chip_ids=self.chip_ids,
                        backend_cluster_descriptor_path=cluster_yaml,
                        backend_runtime_params_path=compiler_cfg.backend_runtime_params_path,
                        store_backend_db_to_yaml=compiler_cfg.store_backend_db_to_yaml,
                        backend_type=self.devtype,
                        device_yaml=soc_desc,
                        backend_output_dir=compiler_cfg.backend_output_dir,
                        backend_device_descriptor_path_override=compiler_cfg.backend_device_descriptor_path,
                        harvesting_mask=harvesting_mask)
        return dev_cfg


    def place_module(self, module: Union[Module, Tuple[Module], List[Module]]):
        if not isinstance(module, (tuple, list)):
            module = (module,)

        for m in module:
            if not isinstance(m, Module):
                raise RuntimeError("Only PyBuda modules can be placed on TTDevices at this time.")

        Device.place_module(self, module)

    def _initialize(self, 
            training: bool, 
            sequential: bool,
            final_barrier: Optional[BarrierClass] = None, 
            shutdown_event: Optional[EventClass] = None,
            scale_loss: float = 1.0,
            checkpoint_interval: int = 0,
            perf_trace: bool = False):
        """
        Initialize the Tenstorrent device.

        Parameters
        ----------
        training: bool
            If true, create optimizer and schedulers for training, linking them to the modules on the device

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
            NOT CURRENTLY SUPPORTED ON TTDEVICE

        checkpoint_interval: int, optional
            The weights will be checkpointed into checkpoint queues on host every `checkpoint_interval` optimizer
            steps, if set to non-zero. Zero by default.

        perf_trace: bool, optional
            Set performance tracing mode when running on silicon
        """

        Device._initialize(self, sequential, final_barrier, shutdown_event)

        self._training = training
        self._checkpoint_interval = checkpoint_interval
        self._perf_trace = perf_trace

        if self._checkpoint_interval > 0:
            self._checkpoint_queues: Dict[str, queue.Queue] = {}
            self._optimizer_state_checkpoint_queues: Dict[str, queue.Queue] = {}
            # Create queues for each of the parameters
            mp_context = mp.get_context('spawn')
            for module in self.modules:
                for parameter in module.get_parameters():
                    name = parameter.get_name()
                    if name in self._checkpoint_queues or name in self._optimizer_state_checkpoint_queues:
                        raise RuntimeError(f"Duplicate parameter name found on device {self}: {name}")

                    self._checkpoint_queues[name] = create_queue(mp_context)
                    self._optimizer_state_checkpoint_queues[name] = create_queue(mp_context)

    def remove_modules(self):
        """
        Remove placed modules, and clear the device
        """
        self._compiled = False
        self._compile_output = {}

        if self.backend_api:
            self.backend_api.shutdown()
            self.backend_api = None

        Device.remove_modules(self)

    def set_active_subgraph(self, subgraph_index: int):
        """
        Set the currently active subgraph by limiting the io queues.
        """
        full_io_queues = copy.copy(self._io_queues)
        self._active_subgraph = subgraph_index
        forward_in_push = {}
        for k, v in self._io_queues["forward_in_push"].items():
            forward_in_push[k] = []
            for i, sgi in enumerate(self._compiled_graph_state.ordered_input_subgraph_indices):
                if (sgi == subgraph_index):
                    forward_in_push[k].append(self._io_queues["forward_in_push"][k][i])

        forward_out_pop = {}
        for k, v in self._io_queues["forward_out_pop"].items():
            forward_out_pop[k] = []
            for i, sgi in enumerate(self._compiled_graph_state.ordered_output_subgraph_indices):
                if (sgi == subgraph_index):
                    forward_out_pop[k].append(self._io_queues["forward_out_pop"][k][i])

        self.set_dram_io_queues("forward_in_push", **forward_in_push)
        self.set_dram_io_queues("forward_out_pop", **forward_out_pop)

        # restore to the full set
        self._io_queues = full_io_queues

    def get_active_subgraph(self):
        """
        Gets the currently active subgraph.
        """
        return self._active_subgraph
        

    def generate_graph(self, 
            *inputs: Tensor, 
            target_tensors: List[Tensor] = [],
            return_intermediate: bool = False, 
            graph_name: str = "default_graph", 
            compiler_cfg: Optional[CompilerConfig] = None, 
            trace_only: bool = False, 
            verify_cfg: Optional[VerifyConfig] = None) -> Tuple[Graph, Tuple[Tensor, ...], Dict[str, Tensor], Tuple[Tensor, ...], Optional[Tensor]]:
        """
        Generate a buda graph from the modules on the device, and return the graph and output tensors.
        If input tensors have a value set, the output tensor will also have the calculated output value
        set.

        Parameters
        ----------
        inputs: Tuple[Tensor, ....]
            Input tensors

        target_tensors: List[Tensor]
            Target inputs. Optional, if trace_only is set. Otherwise, value must be provided.

        return_intermediate: bool
            Optional. If set, a dictionary of node IDs -> tensors will be return with intermediate values, for data mismatch debug.

        trace_only: bool
            If set, the graph is made for a quick trace only and shouldn't have side-effects

        Returns
        -------
        Graph, Tuple[Tensor, ...], Dict[str, Tensor], Tuple[Tensor, ...], Optional[Tensor]
            Buda graph, outputs, optional intermediates, original inputs, target tensor
        """

        output_to_module_name_prefix = {}
        output_to_subgraph_index = {}

        # Create the graph
        graph = Graph(graph_name)
        graph.set_microbatch(1)

        if compiler_cfg is None:
            compiler_cfg = _get_global_compiler_config()

        graph.set_enable_training(compiler_cfg.enable_training)

        reset_unique_node_id()

        # Trace through the modules
        all_subgraph_outputs = []
        outputs = inputs
        for idx, module in enumerate(self.modules):
            if compiler_cfg.compile_subgraphs:
                outputs = inputs[idx]

            if not isinstance(module, PyBudaModule):
                # TODO multiple modules and mixing of pybuda and pytorch modules. 
                from .tvm import compile_tvm_for_pybuda


                # Convert to target format, and fall-back from fp32 if that's what left
                # Getting "unsupported scalar BFloat16 error"
                #pytorch_inputs = (t.to_format(t.data_format).value() if isinstance(t, Tensor) else t for t in inputs)
                #pytorch_inputs = tuple(t.type(buda_dataformat_to_pytorch_dtype(self.fp32_fallback)) if t.dtype == torch.float32 else t for t in pytorch_inputs)
                pytorch_inputs = to_pt_tensors(inputs)

                prev_state = state_changed()
                graph, buda_module, inputs, outputs, intermediate = compile_tvm_for_pybuda(graph, module, pytorch_inputs, compiler_cfg, graph_name, verify_cfg=verify_cfg)
                if not trace_only:
                    self.modules.remove(module)
                    self.modules.insert(0, buda_module)
                if not(prev_state):
                    clear_state_changed()
                return graph, outputs, intermediate, inputs, target_tensors

            start_tracing()
            if module == self.loss_module:
                if len(target_tensors) == 0:
                    assert trace_only, "Target tensors must be provided for each output if generate_graph is not in trace only mode"
                    target_tensors = [Tensor.create_from_trace(None, out.shape, out.data_format) for out in outputs]

                assert len(target_tensors) == len(outputs), "Invalid number of target tensor for outputs"
                if len(outputs) == 1:
                    outputs = module.forward(outputs[0], target_tensors[0])
                else:
                    outputs = module.forward(tuple(outputs), tuple(target_tensors))
            else:
                outputs = module.forward(*outputs)
            stop_tracing()
            if isinstance(outputs, Tensor):
                outputs = (outputs,) # Force a tuple

            for output in outputs:
                output_to_module_name_prefix[output] = module.get_name()
                if compiler_cfg.compile_subgraphs:
                    assert output not in output_to_subgraph_index, "Output tensor {} is produced by multiple modules".format(output)

                output_to_subgraph_index[output] = module.subgraph_idx

            if compiler_cfg.compile_subgraphs == False and idx == len(self.modules) - 1:
                all_subgraph_outputs += outputs
            elif compiler_cfg.compile_subgraphs == True:
                all_subgraph_outputs += outputs


        if trace_only:
            return graph, all_subgraph_outputs, {}, inputs, target_tensors

        visited_tensors = {}
        pending_tensors = deque()
        intermediate = {}
        module_input_tensor_to_node: Dict[str, Tensor] = {}
        module_output_tensor_to_node: Dict[str, Tensor] = {}
        module_target_tensor_to_node: Dict[str, Tensor] = {}
        module_loopback_tensor_to_node: Dict[str, Tensor] = {}
        passthroughs: Set = set()

        input_node_names = []
        input_names_known = True
        if isinstance(inputs[0], Tensor):
            inputs = (inputs,)
        for index, (module, submodule_input) in enumerate(zip(self.modules, inputs)):
            submodule_input_node_names = list(inspect.signature(super(PyBudaModule, module).__getattribute__("forward")).parameters.keys())
            if len(self.modules) > 1:
                submodule_input_node_names = [f"{input_name}_{index}" for input_name in submodule_input_node_names]
            input_node_names += submodule_input_node_names
            if len(submodule_input_node_names) != len(submodule_input):
                input_names_known = False
        inputs, _, _ = flatten_inputs(inputs)

        for out in all_subgraph_outputs:
            is_loss_output = self.loss_module is not None
            if out.src_op is None:

                # No source op. It could be a pass-through, so let's compare to inputs
                found = False
                for input in inputs:
                    if input == out:
                        # Found a passthrough
                        outq = create_output(graph, 
                            output_to_module_name_prefix.get(out, "") + f".output_passthrough_{len(passthroughs)}",
                            out.shape.get_pytorch_shape(), 
                            out.data_format,
                            is_loss_output,
                            output_to_subgraph_index.get(out, 0))
                        passthroughs.add(input)
                        found = True
                        break

                if not found:
                    raise RuntimeError("Untraced output tensor encountered")

            else:
                outq = create_output(graph, 
                        output_to_module_name_prefix.get(out, "") + ".output_" + out.src_op.name, 
                        out.shape.get_pytorch_shape(), 
                        out.data_format,
                        is_loss_output,
                        output_to_subgraph_index.get(out, 0))
            module_output_tensor_to_node[out] = outq
            pending_tensors.append( (out, outq, 0, [], output_to_subgraph_index.get(out, 0)) )

        recorded_parameters = {}

        while pending_tensors:

            tensor, output, port_index, operand_broadcast, subgraph_idx = pending_tensors.popleft()

            if tensor in visited_tensors:
                # Already created the note - let's add the edge and move on
                create_data_edge(graph, visited_tensors[tensor], 0, output, port_index, operand_broadcast)
                continue

            if isinstance(tensor, int):
                # integer constant. Don't add to visited tensors.
                assert False # not supported any more

            if isinstance(tensor, Parameter):
                # parameter tensor
                if tensor.get_name() is not None:
                    name = tensor.get_name()
                else:
                    name = "parameter_" + graph.get_node_name(output)

                if name in recorded_parameters:
                    # Multiple subgraphs might use the same parameter. If it is used in the same subgraph,
                    # we should have already found it in the visited_tensors dictionary. Putting an assert here
                    # to catch fallouts.
                    assert graph.get_subgraph_id_for_node(recorded_parameters[name]) != subgraph_idx, \
                            "Trying to add parameter with name: {} that is used in the same subgraph".format(name)
                    create_data_edge(graph, recorded_parameters[name], 0, output, port_index, operand_broadcast)
                    continue

                inq = create_parameter_input(
                        graph, 
                        name,
                        tensor.shape.get_pytorch_shape(),
                        tensor.requires_grad,
                        tensor.data_format,
                        subgraph_idx)
                create_data_edge(graph, inq, 0, output, port_index, operand_broadcast)
                visited_tensors[tensor] = inq
                recorded_parameters[name] = inq
                continue
            
            if tensor.src_op is None:
                input_name = input_node_names[inputs.index(tensor)] if input_names_known and tensor in inputs else "input_" + str(port_index) + "_" + graph.get_node_name(output)
                if tensor in passthroughs:
                    # passthrough input->output, add a nop
                    inq = create_activation_input(
                            graph,
                            input_name,
                            tensor.shape.get_pytorch_shape(),
                            tensor.requires_grad,
                            tensor.data_format,
                            subgraph_idx)

                    nop = create_op_node(graph, f"_passthrough_nop_{output}", 
                            OpType("nop"), tensor.shape.get_pytorch_shape(), tensor.data_format, subgraph_idx, {})

                    create_data_edge(graph, inq, 0, nop, 0, operand_broadcast)
                    create_data_edge(graph, nop, 0, output, 0, operand_broadcast)
                    visited_tensors[tensor] = inq
                    module_input_tensor_to_node[tensor] = inq
                    continue

                elif tensor in target_tensors:
                    # Target input
                    inq = create_target_input(
                            graph,
                            input_name,
                            tensor.shape.get_pytorch_shape(),
                            tensor.requires_grad,
                            tensor.data_format,
                            subgraph_idx)
                    create_data_edge(graph, inq, 0, output, port_index, operand_broadcast)
                    visited_tensors[tensor] = inq
                    module_target_tensor_to_node[tensor] = inq
                    continue

                elif tensor.is_constant():
                    # Target input
                    inq = create_constant_input(
                            graph,
                            input_name,
                            tensor.value(),
                            tensor.shape.get_pytorch_shape(),
                            tensor.data_format,
                            subgraph_idx)
                    create_data_edge(graph, inq, 0, output, port_index, operand_broadcast)
                    visited_tensors[tensor] = inq
                    module_target_tensor_to_node[tensor] = inq
                    continue

                else:
                    # input tensor
                    input_creator = create_activation_input if input_name not in compiler_cfg.loopback_outputs else create_parameter_input

                    if input_name in compiler_cfg.loopback_outputs:
                        module.add_parameter(input_name, Parameter(tensor.value(), requires_grad=True, name=input_name))

                    inq = input_creator(
                            graph,
                            input_name,
                            tensor.shape.get_pytorch_shape(),
                            tensor.requires_grad,
                            tensor.data_format,
                            subgraph_idx)
                    create_data_edge(graph, inq, 0, output, port_index, operand_broadcast)
                    visited_tensors[tensor] = inq
                    if input_name not in compiler_cfg.loopback_outputs:
                        module_input_tensor_to_node[tensor] = inq
                    elif input_name in compiler_cfg.loopback_outputs:
                        module_loopback_tensor_to_node[tensor] = inq
                        recorded_parameters[input_name] = inq
                    continue

            elif tensor.src_op.op_type == "constant":
                constant_value = tensor.src_op.attrs[0]
                constant = create_constant_input(
                        graph,
                        "constant_" + str(port_index) + "_" + graph.get_node_name(output),
                        constant_value,
                        tensor.data_format,
                        subgraph_idx)

                create_data_edge(graph, constant, 0, output, port_index, operand_broadcast)
                visited_tensors[tensor] = constant
                continue

            '''
            print("ttdevice.py, create_op_node")
            print(f"graph type: {type(graph)}")
            print(f"src_op name: {tensor.src_op.name}")
            print(f"src_op op_type: {tensor.src_op.op_type}")
            print(f"src_op attrs: {tensor.src_op.attrs}")
            print(f"shape: {tensor.shape.get_pytorch_shape()}")
            print(f"data format: {tensor.data_format}")
            '''

            tags = {}
            if tensor.src_layer is not None:
                tags["layer"] = tensor.src_layer
            op = create_op_node(graph, tensor.src_op.name, tensor.src_op.cpp_op_type, tensor.shape.get_pytorch_shape(), tensor.data_format, subgraph_idx, tags)

            visited_tensors[tensor] = op
            if return_intermediate and tensor.has_value():
                intermediate[op] = tensor.value()

            create_data_edge(graph, op, 0, output, port_index, operand_broadcast)

            for i, t in enumerate(tensor.src_op.operands):
                pending_tensors.append( (t, op, i, tensor.src_op.operand_broadcast, subgraph_idx) )

        # Register input/output order of the module to the graph now that the nodes are created
        module_inputs = [module_input_tensor_to_node[input_tensor] for input_tensor in inputs if input_tensor in module_input_tensor_to_node]
        module_outputs = [module_output_tensor_to_node[output_tensor] for output_tensor in all_subgraph_outputs if output_tensor in module_output_tensor_to_node]
        module_targets = [module_target_tensor_to_node[target_tensor] for target_tensor in target_tensors]
        out_requires_grad = [output_tensor.requires_grad for output_tensor in all_subgraph_outputs if output_tensor in module_output_tensor_to_node]

        # Remove unused inputs from list of module inputs
        inputs = [input_tensor for input_tensor in inputs if input_tensor in module_input_tensor_to_node or input_tensor in module_output_tensor_to_node]

        # Remove loopback inputs from list of module inputs
        inputs = [input_tensor for input_tensor in inputs if input_tensor not in module_loopback_tensor_to_node]

        if len(compiler_cfg.loopback_outputs):
            output_to_remove = []
            out_requires_grad_to_remove = []
            for input_name, output_indices in compiler_cfg.loopback_outputs.items():
                if isinstance(output_indices, int):
                    output_indices = [output_indices]
                for output_index in output_indices:
                    input_id = graph.get_node_id(input_name)
                    output_id = module_outputs[output_index]
                    add_partial_datacopy_edge(graph, output_id, 0, input_id, 0)
                    output_to_remove.append(module_outputs[output_index])
                    out_requires_grad_to_remove.append(out_requires_grad[output_index])
            [module_outputs.remove(value) for value in output_to_remove]
            [out_requires_grad.remove(value) for value in out_requires_grad_to_remove]
    
        graph.register_module_inputs(module_inputs)
        graph.register_module_targets(module_targets)
        graph.register_module_outputs(module_outputs, out_requires_grad)

        for parameter in self.get_parameters():
            parameter_name = parameter.get_name()
            if parameter_name not in recorded_parameters:
                self._unused_parameters.add(parameter_name)

        if return_intermediate:
            return graph, outputs, intermediate, inputs, target_tensors

        return graph, outputs, {}, inputs, target_tensors

    def compile_for(self, 
            inputs: Tuple[Tensor, ...],
            compiler_cfg: CompilerConfig,
            targets: List[Tensor] = [],
            microbatch_size: int = 0,
            microbatch_count: int = 1,
            verify_cfg: Optional[VerifyConfig] = None,
            ) -> Tuple[Tensor, ...]:

        """
        Compile modules placed on this device, with given input shapes, input formats, and microbatch size.

        Parameters
        ----------
        training: bool
            Specify whether to compile for training or inference. If set to true, autograd will be executed
            before the compile.

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
            Only relevant for training. This represents the number of microbatches that are pushed through
            fwd path before bwd path runs. The device will ensure that buffering is large enough to contain
            microbatch_count number of microbatch intermediate data.

        verify_cfg: Optional[VerifyConfig]
            Optional auto-verification of compile process

        Returns
        -------
        Tuple[Tensor, ...]
            Output tensors


        """
        if self.device_mode != DeviceMode.RunOnly:
            assert not self._compiled, "Trying to compile a design that's already been compiled"

        training = compiler_cfg.enable_training
        if compiler_cfg.compile_subgraphs:
            input_shapes_group = []
            for input in inputs:
                input_shapes_group.append(tuple(i.shape for i in input))
            input_shapes = tuple(input_shapes_group)
        else:
            input_shapes = tuple(i.shape for i in inputs)
        # input_formats = tuple(i.data_format for i in inputs)
        Device.compile_for(self, training, microbatch_size, microbatch_count)

        if training:
            logger.debug("Compiling for Training mode on {}", self)
        else:
            logger.debug("Compiling for Inference mode on {}", self)

        self.input_shapes = input_shapes # record for checking later

        if verify_cfg is None:
            verify_cfg = VerifyConfig.disabled() # no verification config provided, disable by default

        losses = None

        should_compile = self.device_mode == DeviceMode.CompileAndRun or self.device_mode == DeviceMode.CompileOnly

        from .compile import pybuda_compile_from_context, handle_backend_error, CompileContext
        from .compiled_graph_state import CompiledGraphState

        compile_context: Optional[CompileContext] = None
        if should_compile:
            compiler_cfg.apply_env_config_overrides()
            compile_context = CompileContext(
                dev=self,
                graph_name=self.modules[0].get_name(),
                inputs=inputs,
                compiler_cfg=compiler_cfg,
                verify_cfg=verify_cfg,
                device_cfg=self.get_device_config(compiler_cfg),
                microbatch_size=microbatch_size,
                microbatch_count=microbatch_count,
                targets=targets,
                losses=losses,
            )

        while self._compiled == False or self.backend_api == None:

            if should_compile:
                assert compile_context is not None
                compile_context.device_cfg = self.get_device_config(compiler_cfg)
                self._compile_output = pybuda_compile_from_context(compile_context)

                self._compiled_graph_state = CompiledGraphState.from_compiled_graph(self, self._compile_output)

            device_mode_for_backend = DeviceMode.RunOnly if "PYBUDA_SKIP_BACKEND_COMPILE" in os.environ else self.device_mode
            backend_runtime_args = compiler_cfg.backend_runtime_args if "PYBUDA_FORCE_SEQUENTIAL" in os.environ else compiler_cfg.backend_runtime_args + " --concurrent-mode"

            # Set some perf defaults for WH
            if self.arch == BackendDevice.Wormhole_B0:
                os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"

            try:
                self.backend_api = BackendAPI(
                    self.devtype,
                    self.arch,
                    self,
                    self._compiled_graph_state.netlist_filename,
                    self._compiled_graph_state,
                    not self._sequential,
                    None,
                    None,
                    compiler_cfg.performance_trace,
                    device_mode_for_backend,
                    verify_cfg.golden_ignore_df_precision,
                    compiler_cfg.backend_opt_level,
                    compiler_cfg.backend_output_dir,
                    # for nebula+galaxy, backend_device_descriptor_path is for unharvested device_desc
                    # creating backend with it will cause crashes when runtime tries to reset the harvested cores in nebulas
                    # not passing device_desc allows runtime to create unharvested&harvested device_desc's for each chip
                    compiler_cfg.backend_device_descriptor_path if "PYBUDA_NEBULA_GALAXY_PLACER" not in os.environ else "",
                    compiler_cfg.backend_cluster_descriptor_path,
                    backend_runtime_args)
            except BackendCompileException as ex:
                if compile_context is not None:
                    if handle_backend_error(compile_context, ex):
                        # Continue to recompile
                        continue

                raise RuntimeError("Backend compile failed!")

            if compile_context is not None and compile_context.recompile_count > 0:
                logger.info("Compile successfully completed after {} retries!", compile_context.recompile_count)

            self._compiled = True

        if self.device_mode == DeviceMode.CompileAndRun or self.device_mode == DeviceMode.RunOnly:
            # Copy constants and parameters to device - probably shouldn't be part of compile, but explicit on run!
            self.backend_api.push_constants_and_parameters(translate=True)
            self.backend_api.push_optimizer_parameters(translate=True)

        if self._compile_output and self._compile_output.outputs:
            return [t.detach() for t in self._compile_output.outputs] # detach so it can pushed into mp queues
        else:
            assert self.device_mode == DeviceMode.RunOnly, (
                "We should only be returning empty tensors when configuring the device from image."
            )
            # don't necessarily need to contain any contents, but disallow auto-verification
            return [
                Tensor.create_from_trace(None, shape, data_format)
                for shape, data_format in 
                zip(self._compiled_graph_state.ordered_output_shapes, self._compiled_graph_state.ordered_output_data_formats)
            ]

    def update_device_parameters(self, parameters: Dict[str, torch.Tensor]):

        assert self.backend_api
        self.sync() # wait until queued up commands have completed
        self.backend_api.update_device_paramaters(parameters)

    def _post_graph_callback(self):
        """
        Called after buda graph has been generated, but the compile process hasn't yet happened.
        """
        pass

    def forward(self, loop_count: int):
        """
        Run forward pass on each module on this device, in order

        Parameters
        ----------
        loop_count: int
            Number of micro-batches to run

        """

        logger.debug("Starting forward on {}", self)
        assert self._compiled, f"Module not compiled yet on {self}"
        assert self.backend_api is not None

        self.backend_api.schedule_run_forward(loop_count)

    def generate(self, loop_count: int, write_index: int, tokens_per_iter: int, token_id: int):
        """
        Run forward pass on each module on this device, in order

        Parameters
        ----------
        loop_count: int
            Number of micro-batches to run

        """

        logger.debug("Starting generate on {}", self)
        assert self._compiled, f"Module not compiled yet on {self}"
        assert self.backend_api is not None

        if tokens_per_iter != -1:
            if tokens_per_iter == TILE_DIM: #pre-populating cache
                write_index = token_id // TILE_DIM
                inner_loop_count = loop_count
                outer_loop_count = 1
                inner_increment = 1
                outer_increment = 0
            elif tokens_per_iter != 1: #last pre-population step
                assert loop_count == 1
                write_index = token_id // TILE_DIM
                inner_loop_count = 1
                outer_loop_count = 1
                inner_increment = 0
                outer_increment = 0
            else: #token generation
                write_index = token_id // TILE_DIM
                outer_loop_count = loop_count // TILE_DIM if loop_count > TILE_DIM else 1
                inner_loop_count = loop_count % TILE_DIM if loop_count < TILE_DIM else TILE_DIM
                inner_increment = 0
                outer_increment = 1
        else: #manual_mode
            inner_loop_count = loop_count
            inner_increment = 0
            outer_loop_count = 1
            outer_increment = 0
        logger.debug("Generating: write_index: {}, inner_loop_count: {}, inner_increment: {}, outer_loop_count: {}, outer_increment: {}", write_index, inner_loop_count, inner_increment, outer_loop_count, outer_increment)
        self.backend_api.schedule_run_generate(write_index, inner_loop_count, inner_increment, outer_loop_count, outer_increment)
        
        
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
        assert self.device_mode != DeviceMode.RunOnly, (
            "Device has been configured from image. We disallow auto-verification in this mode."
        )
        assert self._compile_output is not None and self._compile_output.initial_graph is not None

        if save_for_backward:
            self._saved_fw_inputs = inputs

        microbatch_size = self._compile_output.initial_graph.get_microbatch()
        assert inputs[0].shape[0] == microbatch_size
        output_list = []
        for i in range(microbatch_size):
            if microbatch_size > 1:
                mb_inputs = tuple(input[i:i+1] for input in inputs)
            else:
                mb_inputs = inputs 

            if targets is not None:
                mb_targets = tuple(target[i:i+1] for target in targets)
            else:
                mb_targets = None
            output, *_ = graph_eval(self._compile_output.initial_graph, mb_inputs, parameters, self, 0.1, 1.00, targets=mb_targets)

            output_list.append(output)

        outputs = []
        for out in zip(*output_list):
            outputs.append(torch.cat(out, 0))
        outputs = tuple(outputs)

        if save_for_backward:
            self._saved_fw_outputs = outputs

        return outputs

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
        logger.debug("Starting backward on {}", self)

        # Since we don't support loss on ttdevice yet, we will always do a forward first, which will compile
        assert self._compiled, "Model not compiled yet"
        assert self.backend_api is not None

        self.backend_api.schedule_run_backward(loop_count, zero_grad)

    def _step_optimizer(self):
        """
        Step optimizer
        """
        logger.debug("Stepping optimizer on {}", self)
        assert self.backend_api is not None
        self.backend_api.schedule_run_optimizer()

    def _step_schedulers(self):
        """
        Step schedulers
        """
        if self.scheduler:
            logger.debug("Stepping schedulers on {}", self)
            assert self.backend_api is not None
            self.backend_api.schedule_run_schedulers(self)

    def get_parameter_checkpoint(self) -> Dict[str, Tensor]:
        """
        Return a dictionary of current parameter values for the models on this device
        """
        self.sync() # wait until queued up commands have completed
        assert self.backend_api is not None
        ret = {}
        queues = []
        shapes = []
        names = []
        for module in self.modules:
            for parameter in module.get_parameters():
                if parameter.requires_grad:
                    name = parameter.get_name()
                    names.append(name)
                    queues.append(self.backend_api.be_api.get_queue_descriptor(name))
                    constevaled_shape = consteval_shape(self._compiled_graph_state, name, parameter.value())
                    shapes.append(constevaled_shape)

        values = BackendAPI.read_queues(queues, shapes, runtime_tensor_transforms=None, requires_grad= [False] * len(queues), single_output=True, rd_ptr=0, 
                shutdown_event=self.shutdown_event, clone=True, has_microbatch_dim=False)
        for name, value in zip(names, values):
            if self._training:
                ret[name] = Tensor.create_from_torch(consteval_input_bw(self._compiled_graph_state, name, value.value(), False))
            else:
                # We don't have a backward consteval graph recorded, return the raw paramater value
                ret[name] = Tensor.create_from_torch(value.value())

        return ret

    def get_all_parameters(self) -> Dict[str, Tensor]:
        """
        Return a dictionary of current parameter values for the models on this device
        """
        self.sync() # wait until queued up commands have completed
        assert self.backend_api is not None
        ret = {}
        queues = []
        shapes = []
        names = []
        for name, param in self._compiled_graph_state.post_const_eval_parameters.items():
            names.append(name)
            queues.append(self.backend_api.be_api.get_queue_descriptor(name))
            shapes.append(param.shape)

        values = BackendAPI.read_queues(queues, shapes, runtime_tensor_transforms=None, requires_grad= [False] * len(queues), single_output=True, rd_ptr=0, 
                shutdown_event=self.shutdown_event, clone=True, has_microbatch_dim=False)
        for name, value in zip(names, values):
            if self._training:
                ret[name] = Tensor.create_from_torch(consteval_input_bw(self._compiled_graph_state, name, value.value(), False))
            else:
                # We don't have a backward consteval graph recorded, return the raw paramater value
                ret[name] = Tensor.create_from_torch(value.value())

        return ret


    def get_parameter_gradients(self) -> Dict[str, Tensor]:
        """
        Return a dictionary of currently accumulated gradient values for the models on this device
        """
        self.sync() # wait until queued up commands have completed
        assert self.backend_api is not None
        ret = {}
        queues = []
        shapes = []
        names = []
        for module in self.modules:
            for parameter in module.get_parameters():
                if parameter.requires_grad:
                    name = parameter.get_name()
                    queue_name = "grad_acc_" + name
                    constevaled_shape = consteval_shape(self._compiled_graph_state, name, parameter.value())
                    names.append(name)
                    shapes.append(constevaled_shape)
                    queues.append(self.backend_api.get_output_queue_descriptor(queue_name))

        values = BackendAPI.read_queues(queues, shapes, runtime_tensor_transforms=None, requires_grad = [False] * len(queues), single_output=True, rd_ptr=0,
                shutdown_event=self.shutdown_event, clone=True, has_microbatch_dim=False)
        for name, value in zip(names, values):
            ret[name] = Tensor.create_from_torch(consteval_input_bw(self._compiled_graph_state, name, value.value(), False))

        return ret
    
    def _model_pop_optimizer_state_checkpoint(self) -> Dict:
        """
        """
        
        if len(self.optimizer.get_optimizer_state_keys()) == 0:
            return {}

        ret = {}
        for module in self.modules:
            for parameter in module.get_parameters():
                if parameter.requires_grad:
                    name = parameter.get_name()
                    tensor = parameter.get_empty_tensor().tensor
                    optimizer_states = buda.pop_optimizer_state_checkpoint(
                        self._get_gstate(),
                        0,
                        tensor,
                        self.devtype
                    )

                    ret[name] = optimizer_states
        return ret

    def _get_fw_tilizer_target_device_id(self):
        """
        Return the device_id that we push forward inputs to. In single-device setup, that's always 0
        """
        return 0

    def _get_bw_tilizer_target_device_id(self):
        """
        Return the device_id that we push backward inputs to. In single-device setup, that's always 0
        """
        return 0

    def get_parameters(self, ignore_unused_parameters: bool = True) -> List[Parameter]:
        """
        Parameters
        ----------
        ignore_used_parameters: bool
            If true, any parameter not being recorded by the graph-trace (i.e. parameter is unused in
            graph execution) is not included in the returned list to user.
        """

        # In traing mode, we need to gather gradients from the modules, so can't use pre-stored parameters
        is_training = self.optimizer is not None
        ret: List[Parameter] = []
        for module in self.modules:
            ret.extend(module.get_parameters())

        if ignore_unused_parameters:
            ret = [parameter for parameter in ret if parameter.get_name() not in self._unused_parameters]

        return ret

    def get_optimizer(self) -> Optional[Optimizer]:
        return self.optimizer

    def get_optimizer_params(self, is_buda: bool) -> Dict[str, Dict[str, Tensor]]:
        """
        Return a dictionary of dictionaries of optimizer parameters for each model parameter.
        """
        if not self.optimizer:
            return {}

        ret = {}
        for param in self.get_parameters():
            if not param.requires_grad:
                continue

            name = param.get_name()
            optimizer_params = self.optimizer.get_optimizer_params(name, is_buda)
            if optimizer_params is None:
                continue

            ret[name] = optimizer_params

        return ret

    def get_scheduler_params(self, is_buda: bool) -> Dict[str, Dict[str, Tensor]]:
        """
        Return a dictionary of dictionaries of optimizer parameters used by scheduler.
        """
        if not self.optimizer:
            return {}

        ret = {}
        for param in self.get_parameters():
            if not param.requires_grad:
                continue

            name = param.get_name()
            optimizer_params = self.scheduler.get_scheduler_params(name, is_buda)
            if optimizer_params is None:
                continue

            ret[name] = optimizer_params

        return ret

    def _get_fwd_inputs_tile_broadcast_dims(self) -> List[List[int]]:
        """
        Return a list of tile broadcast dims for each direct input into the device (fwd)
        """
        assert self._compiled_graph_state
        return self._compiled_graph_state.ordered_input_tile_broadcast_dims

    def _get_target_inputs_tile_broadcast_dims(self) -> List[List[int]]:
        """
        Return a list of tile broadcast dims for each target input into the device
        """
        assert self._compiled_graph_state
        return self._compiled_graph_state.ordered_target_tile_broadcast_dims

    def _get_bwd_inputs_tile_broadcast_dims(self) -> List[List[int]]:
        """
        Return a list of tile broadcast dims for each direct input into the device (bwd)
        """
        assert self._compiled_graph_state
        return self._compiled_graph_state.ordered_bw_input_tile_broadcast_dims

    def _get_input_shapes(self, grad_only: bool) -> List[Tuple[int, ...]]:
        """
        Return a list of original input shapes. If `grad_only`, only return those that have requires_grad set
        """
        assert self._compiled_graph_state
        input_shapes = self._compiled_graph_state.ordered_input_shapes
        requires_grad = self._compiled_graph_state.ordered_input_requires_grad
        microbatch = self._compiled_graph_state.microbatch

        for i, in_shape in enumerate(input_shapes):
            if in_shape[0] == 1:
                in_shape[0] = microbatch

        if grad_only:
            input_shapes = [s for i, s in enumerate(input_shapes) if requires_grad[i]]
        return input_shapes
    
    def _adjust_shapes_for_microbatch(self, shapes: List[Tuple[int, ...]], microbatch: int) -> List[Tuple[int, ...]]:
        for i, out_shape in enumerate(shapes):
            if out_shape[0] != 1 and out_shape[0] != microbatch:
                out_shape.insert(0, 1)
            out_shape[0] =  microbatch
        return shapes


    def _get_output_shapes(self, grad_only: bool) -> List[Tuple[int, ...]]:
        """
        Return a list of original output shapes. If `grad_only`, only return those that have requires_grad set
        """
        assert self._compiled_graph_state
        output_shapes = self._compiled_graph_state.ordered_output_shapes
        requires_grad = self._compiled_graph_state.ordered_output_requires_grad
        output_shapes = self._adjust_shapes_for_microbatch(output_shapes, self._compiled_graph_state.microbatch)

        if grad_only:
            output_shapes = [s for i, s in enumerate(output_shapes) if requires_grad[i]]

        return output_shapes

    def _get_intermediate_shapes(self) -> List[Tuple[int, ...]]:
        assert self._compiled_graph_state
        shapes = self._compiled_graph_state.ordered_intermediate_shapes
        return self._adjust_shapes_for_microbatch(shapes, self._compiled_graph_state.microbatch)

    def _get_input_runtime_tensor_transforms(self) -> List["RuntimeTensorTransform"]:
        assert self._compiled_graph_state

        input_runtime_tensor_transforms = self._compiled_graph_state.ordered_input_runtime_tensor_transforms
        microbatch = self._compiled_graph_state.microbatch

        # If RuntimeTensorTransform is a ReinterpretShape:
        #   If microbatch 1, reinterpret input shapes will drop it... we need
        #   that unary dimension because the backwards reinterpret shapes uses 
        #   this and the narrow tensor code expects activations and activation
        #   gradients to have a microbatch dimension
        for i, transform in enumerate(input_runtime_tensor_transforms):
            if transform.type != RuntimeTensorTransformType.ReinterpretShape:
                continue

            while len(transform.reinterpreted_shape) < 3:
                transform.reinterpreted_shape = Shape.create_with_type_from_other(transform.reinterpreted_shape, [1] + transform.reinterpreted_shape.as_list())

            if transform.reinterpreted_shape.as_list()[0] not in [1, microbatch]:
                transform.reinterpreted_shape = Shape.create_with_type_from_other(transform.reinterpreted_shape, [1] + transform.reinterpreted_shape.as_list())

            if transform.reinterpreted_shape.as_list()[0] == 1:
                reinterpreted_shape_as_list = transform.reinterpreted_shape.as_list()
                reinterpreted_shape_as_list[0] = microbatch
                transform.reinterpreted_shape = Shape.create_with_type_from_other(transform.reinterpreted_shape, reinterpreted_shape_as_list)

        return input_runtime_tensor_transforms

    def _get_output_runtime_tensor_transforms(self) -> List[List[int]]:
        assert self._compiled_graph_state

        output_runtime_tensor_transforms = self._compiled_graph_state.ordered_output_runtime_tensor_transforms
        microbatch = self._compiled_graph_state.microbatch

        # # If RuntimeTensorTransform is a ReinterpretShape:
        # #   If microbatch 1, reinterpret input shapes will drop it... we need
        # #   that unary dimension because the backwards reinterpret shapes uses 
        # #   this and the narrow tensor code expects activations and activation
        # #   gradients to have a microbatch dimension
        for i, transform in enumerate(output_runtime_tensor_transforms):
            if transform.type == RuntimeTensorTransformType.ReinterpretShape:
                if transform.reinterpreted_shape.as_list()[0] not in [1, microbatch]:
                    transform.reinterpreted_shape = Shape.create_with_type_from_other(transform.reinterpreted_shape, [1] + transform.reinterpreted_shape.as_list())

                reinterpreted_shape_as_list = transform.reinterpreted_shape.as_list()
                reinterpreted_shape_as_list[0] = microbatch
                transform.reinterpreted_shape = Shape.create_with_type_from_other(transform.reinterpreted_shape, reinterpreted_shape_as_list)
            elif transform.type == RuntimeTensorTransformType.Unpad:
                if transform.unpadded_shape.as_list()[0] not in [1, microbatch]:
                    transform.unpadded_shape = Shape.create_with_type_from_other(transform.unpadded_shape, [1] + transform.unpadded_shape.as_list())

                unpadded_shape_as_list = transform.unpadded_shape.as_list()
                unpadded_shape_as_list[0] = microbatch
                transform.unpadded_shape = Shape.create_with_type_from_other(transform.unpadded_shape, unpadded_shape_as_list)

        return output_runtime_tensor_transforms

    def _get_output_requires_grad(self) -> List[bool]:
        """
        Return a list of requires_grad flags on each output
        """
        assert self._compiled_graph_state
        return self._compiled_graph_state.ordered_output_requires_grad

    def _get_input_requires_grad(self) -> List[bool]:
        """
        Return a list of requires_grad flags on each input
        """
        assert self._compiled_graph_state
        return self._compiled_graph_state.ordered_input_requires_grad


    def _create_forward_device_connector(self, target_device: Union["TTDevice", "CPUDevice"], sequential: bool, d2d_fwd_queue: Optional[queue.Queue] = None, microbatch = 1):

        logger.debug("Creating forward device connector from {} to {}", self, target_device)
        if isinstance(target_device, TTDevice):
            # direct transfer both ways
            self.forward_dc = DirectPusherPopperDeviceConnector(self.shutdown_event, sequential, side_queue=d2d_fwd_queue)
        else:
            # TTDevice copies directly to host, no pushing
            self.forward_dc = DirectPopperDeviceConnector(self.shutdown_event, side_queue=d2d_fwd_queue)

        target_device._set_forward_input_dc(self.forward_dc)

    def _create_backward_device_connector(self, target_device: Device, sequential: bool, d2d_bwd_queue: Optional[queue.Queue] = None, microbatch = 1):

        logger.debug("Creating backward device connector from {} to {}", self, target_device)
        if isinstance(target_device, TTDevice):
            # direct transfer both ways
            self.backward_dc = DirectPusherPopperDeviceConnector(self.shutdown_event, sequential, side_queue=d2d_bwd_queue)
        else:
            # TTDevice copies directly to host, no pushing
            self.backward_dc = DirectPopperDeviceConnector(self.shutdown_event, side_queue=d2d_bwd_queue)
        target_device._set_backward_input_dc(self.backward_dc)

    # Create device connector for the last device, pushing forward
    def _create_forward_output_queue_device_connector(self, q: queue.Queue):
        logger.debug("Creating forward output queue connector on {}", self)
        self.forward_dc = OutputQueueDirectPoppperDeviceConnector(q, self.shutdown_event)

    # Create device connector for the first device, pushing backward
    def _create_backward_output_queue_device_connector(self, q: queue.Queue):
        logger.debug("Creating backward output queue connector on {}", self)
        self.backward_dc = OutputQueueDirectPoppperDeviceConnector(q, self.shutdown_event)

    # Create device connector for the first device, reading from a Queue
    def _create_input_queue_device_connector(self, q: queue.Queue, sequential: bool):
        logger.debug("Creating input queue connector on {}", self)
        self.forward_input_dc = InputQueueDirectPusherDeviceConnector(q, self.shutdown_event, sequential)

    # Create device connector for the last device, reading from a Queue
    def _create_target_queue_device_connector(self, q: queue.Queue, sequential: bool):
        logger.debug("Creating input queue connector on {}", self)
        self.target_input_dc = InputQueueDirectPusherDeviceConnector(q, self.shutdown_event, sequential)
        
    # Create device connector for the last device, reading from a Queue
    def _create_intermediates_queue_device_connector(self, q: queue.Queue):
        logger.debug("Creating fwd intermediates queue connector on {}", self)
        self.intermediates_dc = OutputQueueDirectPoppperDeviceConnector(q, self.shutdown_event)


    def get_dram_io_queues(self, queue_type: str) -> Tuple[List[DramIODesc], Optional[List[List[int]]], Optional[List], Optional[List[bool]], Optional[List[Tensor]]]:
        """
        Returns the appropriate queue description, tile broadcast information, and original shapes, where applicable
        """
        assert self.backend_api
        if (queue_type == "input"):
            input_qs = self.backend_api.get_ordered_input_queues()
            transforms = self._get_input_runtime_tensor_transforms()
            constant_inputs = [None for _ in self._compiled_graph_state.ordered_input_names]
            input_tile_dims = [ 
                self._compiled_graph_state.input_to_tile_dims[name]
                for name in self._compiled_graph_state.ordered_input_names
            ]

            for idx, transform in enumerate(transforms):
                if transform.type == RuntimeTensorTransformType.Prestride:
                    assert transform.stride_height == transform.stride_width, "Backend supports only square strides for prestriding transform"
                    stride = transform.stride_height
                    stride_desc = StrideDescriptor()
                    stride_desc.stride = stride
                    stride_desc.xy_offsets = [(x, y) for y in range(stride) for x in range(stride)]

                    input_qs[idx].s_descriptor = stride_desc
                elif transform.type == RuntimeTensorTransformType.ConstantInput:
                    constant_inputs[idx] = self._compiled_graph_state.constant_to_tensor[self._compiled_graph_state.ordered_input_names[idx]]
            return input_qs, self._get_fwd_inputs_tile_broadcast_dims(), None, None, transforms, constant_inputs, input_tile_dims

        if (queue_type == "target"):
            return self.backend_api.get_ordered_target_queues(), self._get_target_inputs_tile_broadcast_dims(), None, None, None, None, None

        if (queue_type == "output"): 
            return self.backend_api.get_ordered_output_queues(), None, self._get_output_shapes(grad_only=False), self._get_output_requires_grad(), self._get_output_runtime_tensor_transforms(), None, None

        if (queue_type == "bw_input"):
            return self.backend_api.get_ordered_bw_input_queues(), self._get_bwd_inputs_tile_broadcast_dims(), None, None, self._get_output_runtime_tensor_transforms(), None, None

        if (queue_type == "bw_output"):
            qs = self.backend_api.get_ordered_bw_output_queues()
            return qs, None, self._get_input_shapes(grad_only=True), [True] * len(qs), self._get_input_runtime_tensor_transforms(), None, None

        if (queue_type == "intermediates"):
            intermediate_shapes = self._get_intermediate_shapes()
            requires_grad = [False] * len(intermediate_shapes)
            qs = self.backend_api.get_intermediate_activation_queues()

            return qs, None, intermediate_shapes, requires_grad, None, None, None

        raise RuntimeError("Unknown type of queue")

    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor], lr = None) -> Optional[torch.optim.Optimizer]:
        if self.optimizer is None or len(parameters) == 0:
            return None
        return self.optimizer.get_pytorch_optimizer(parameters, lr)
    
    def get_pytorch_scheduler(self):
        if self.scheduler is None:
            return None
        return self.scheduler.get_pytorch_scheduler(self.optimizer.torch_optimizer)

    def shutdown_device(self):
        """
        Shutdown device at the end of the workload
        """
        self.remove_modules()

        logger.trace("start shutdown threads")
        self._shutdown_threads()

        Device.shutdown_device(self)

    def sync(self):
        """
        Block until queued up commands have completed and the device is idle.
        """
        assert self.backend_api is not None
        self.backend_api.sync()

    def get_compiled_results(self) -> Optional["CompileResults"]:
        from .compile import CompileResults
        if not self._compiled or not self._compile_output:
            logger.error(f"User has not yet compiled a device")
            return None
        return self._compile_output

    def compile_to_image(
        self, 
        *,
        img_path: str = None,
        training: bool = False, 
        sample_inputs: Tuple[Union[torch.Tensor, Tensor], ...] = tuple(),
        sample_targets: Tuple[Union[torch.Tensor, Tensor], ...] = tuple(),
        microbatch_count: int = 1,
        verify_cfg: Optional[VerifyConfig] = None,
        cpueval_outputs: Optional[List[torch.Tensor]] = None,
    ) -> "TTDeviceImage":

        assert self.arch, "When compiling to image, TTDevice must be explicitly constructed with target-arch"
        assert self.devtype, "When compiling to image, TTDevice must be explicitly constructed with dev_type"

        compiler_cfg = _get_global_compiler_config()
        if not self._compiled:
            self.device_mode = DeviceMode.CompileOnly
            from .run import initialize_pipeline
            initialize_pipeline(
                training=training,
                sample_inputs=sample_inputs,
                sample_targets=sample_targets,
                microbatch_count=microbatch_count,
                _sequential=True,
                _verify_cfg=verify_cfg,
                _device_mode=self.device_mode
            )

        from .tti import TTDeviceImage
        device_image = TTDeviceImage.create_image_from_device(
            self, 
            training,
            microbatch_count,
            verify_cfg,
            compiler_cfg,
            cpueval_outputs=cpueval_outputs,
        )
        TTDeviceImage.save_to_disk(device_image, img_path, self.backend_api)

        return device_image

    
    @staticmethod
    def load_image(*, img: Optional["TTDeviceImage"] = None, img_path: Optional[str] = None) -> "TTDevice":
        from .tti import TTDeviceImage
        if img and img_path:
            logger.error("only one of image/image-path should be specified")
        if img is None:
            img = TTDeviceImage.load_from_disk(img_path)
        return TTDeviceImage.create_device_from_image(img)


def get_backend_string(backend_type: BackendType) -> str:
    BACKEND_TYPE_TO_DEVICE_GRID = {
            BackendType.Golden: "golden",
            BackendType.Model: "model",
            BackendType.Versim: "versim",
            BackendType.Emulation: "emulation",
            BackendType.NoBackend: "nobackend",
            BackendType.Silicon: "silicon",
    }
    if backend_type in BACKEND_TYPE_TO_DEVICE_GRID:
        return BACKEND_TYPE_TO_DEVICE_GRID[backend_type]
    else:
        raise Exception("Running pybuda_compile with unknown backend_type config")


def get_default_device_yaml(
    arch: BackendDevice,
    device_yaml: str,
    backend_output_dir: str, 
    device_yaml_override: Optional[str],
    harvesting_mask: int
) -> str:
    if arch not in {BackendDevice.Grayskull, BackendDevice.Wormhole_B0, BackendDevice.Blackhole}:
        raise RuntimeError("Running pybuda_compile with unknown arch config")
    if device_yaml_override:
        return device_yaml_override

    if harvesting_mask:
        default_device_desc = get_custom_device_desc(arch, mmio=True, harvesting_mask=harvesting_mask, out_dir=backend_output_dir)
        return default_device_desc.soc_desc_yaml
    else:
        return device_yaml

def get_default_cluster_descriptor(backend_output_dir: str, backend_cluster_descriptor_path: str = "") -> str:
    cluster_override = os.environ.get("PYBUDA_OVERRIDE_CLUSTER_YAML", None)
    if cluster_override:
        if os.path.isfile(cluster_override):
            return cluster_override
        elif os.path.isfile(budabackend_path() + f"/{cluster_override}"):
            return budabackend_path() + f"/{cluster_override}"
        else:
            raise RuntimeError(f"PYBUDA_OVERRIDE_CLUSTER_YAML={cluster_override} is not a valid file.")
    elif backend_cluster_descriptor_path == "":
        backend_cluster_descriptor_path = get_device_cluster_yaml(backend_output_dir)

    return backend_cluster_descriptor_path

def get_device_config(arch: BackendDevice,
                      chip_ids: Union[List[int], List[Tuple[int]]] = None,
                      backend_cluster_descriptor_path = "",
                      backend_runtime_params_path = "",
                      store_backend_db_to_yaml = False,
                      backend_type = BackendType.NoBackend,
                      device_yaml = "",
                      backend_output_dir = "./tt_build",
                      backend_device_descriptor_path_override = None,
                      harvesting_mask: int = 0) -> str: 
    return DeviceConfig(
        arch.to_string(),
        get_default_device_yaml(arch, device_yaml, backend_output_dir, backend_device_descriptor_path_override, harvesting_mask),
        get_default_cluster_descriptor(backend_output_dir, backend_cluster_descriptor_path),
        backend_runtime_params_path, 
        get_backend_string(backend_type),
        store_backend_db_to_yaml,
        chip_ids,
    )

