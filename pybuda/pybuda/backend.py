# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Backend API wrapper

import os
import threading
import queue
import time
from typing import Optional, List, Tuple, Dict, Union

import torch
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Barrier as BarrierClass

from loguru import logger

import pybuda
from .pybudaglobal import TILE_DIM
from pybuda._C import DataFormat
from pybuda._C.backend_api import BackendType, BackendDevice, BackendApi, BackendConfig, DramIODesc, PytorchTensorDesc, TilizedTensorDesc, BackendStatusCode, BackendCompileResult, clear_backend_param_cache, release_backend_ptr, push_input, pop_output, get_output, translate_addresses, free_tensor, DeviceMode, debinarize_tensor
from pybuda._C.graph import Graph, get_constant_input_value, get_optimizer_param_info, RuntimeTensorTransform, RuntimeTensorTransformType
from pybuda._C.balancer import OutputHostTM
from .tensor import Tensor, consteval_input, pytorch_tensor_to_tensor_desc, pad_pytorch_tensor_to_buda, tensor_desc_to_pytorch_tensor, get_device_constant_and_parameters, const_eval_tensor
from .utils import detach_tensors
from .config import PerfTraceLevel

class BackendCompileException(Exception):
    def __init__(self, compile_result: BackendCompileResult):
        self.compile_result = compile_result

class BackendAPI:

    def __init__(self,
            type: BackendType,
            device_type: BackendDevice,
            device: "TTDevice",
            netlist: str,
            compiled_graph_state: "CompiledGraphState",
            feeder_thread: bool,
            shutdown_event: Optional[EventClass],
            final_barrier: Optional[BarrierClass],
            performance_trace: PerfTraceLevel,
            device_mode: DeviceMode = DeviceMode.CompileAndRun,
            golden_ignore_df_precision: bool = True,
            opt_level: int = 0,
            output_dir: str = "tt_build/test_out",
            device_descriptor_path: str = "",
            cluster_descriptor_path: str = "",
            runtime_args: str = ""):

        self.type = type
        self.device = device
        self.device_type = device_type
        self.netlist = netlist
        self.compiled_graph_state = compiled_graph_state
        self.shutdown_event = shutdown_event
        self.final_barrier = final_barrier
        self.feeder_thread = None
        self.feeder_thread_queue = None
        self.cache_zerod = False
        self.output_dir = output_dir

        # If set, we'll wait for idle after every program
        # It shouldn't be needed, but ok for debug
        self.explicit_barrier_between_programs = False

        bcfg = BackendConfig(
            self.type,
            self.device_type,
            device_mode,
            opt_level,
            output_dir,
            device_descriptor_path,
            cluster_descriptor_path,
        )
        bcfg.set_golden_ignore_df_precision(golden_ignore_df_precision)
        bcfg.set_performance_trace_args(performance_trace.get_backend_cfg_string())
        bcfg.set_runtime_args(runtime_args)

        self.be_api = BackendApi(self.netlist, bcfg)
        self.compile_result = BackendCompileResult()
        if self.be_api.initialize(self.compile_result) != BackendStatusCode.Success:
            logger.info(f"Backend compile {self.compile_result.success}, target: {self.compile_result.failure_target}, error type: {self.compile_result.failure_type}, error: {self.compile_result.failure_message}\n"
            f"target chip id: {self.compile_result.device_id}, target core(x,y): {self.compile_result.logical_core_x} {self.compile_result.logical_core_y}, temporal epoch id: {self.compile_result.temporal_epoch_id}\n"
            f"requires extra size bytes: {self.compile_result.extra_size_bytes}\n")
            self.shutdown()
            raise BackendCompileException(self.compile_result)

        # Create and start a feeder thread, if requested
        if feeder_thread:
            self.feeder_thread_queue = queue.Queue()
            self.feeder_thread = threading.Thread(target=self.feeder_thread_main, args=(self.feeder_thread_queue,))
            self.feeder_thread.start()

    def shutdown(self):
        """
        Shutdown the device
        """
        if self.feeder_thread_queue:
            self.feeder_thread_queue.put("quit")
            self.feeder_thread.join()
            self.feeder_thread= None
            self.feeder_thread_queue = None

        if self.be_api:
            clear_backend_param_cache()
            self.be_api.finish()
            release_backend_ptr(self.be_api)
            self.be_api = None

    def sync(self):
        """
        Wait until device is idle, and queued up commands have completed
        """
        if self.feeder_thread_queue:
            while not self.feeder_thread_queue.empty():
                time.sleep(0.01)
                if self.shutdown_event and self.shutdown_event.is_set():
                    return
        assert self.be_api
        assert self.be_api.wait_for_idle() == BackendStatusCode.Success, "Failed while waiting for device to go idle"

    def feeder_thread_main(self, cmdqueue: queue.Queue):
        """
        Run in a loop, reading commands and executing them, until quit has been received, or an exception occured
        """
        logger.info("Feeder thread on {} starting", self)
        while True:
            while True:
                try:
                    cmd = cmdqueue.get(timeout=0.1)
                    break
                except queue.Empty as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Ending feeder thread on {} due to shutdown event", self)
                        if self.final_barrier is not None:
                            self.final_barrier.abort()
                        return # got a signal to shutdown and end the process
                    continue
            
            if cmd == "token":
                continue
            else:
                params = []
                if isinstance(cmd, tuple):
                    params = cmd[1:]
                    cmd = cmd[0]

            logger.debug("Run feeder thread cmd: {}", cmd)
            if cmd == "fwd":
                self._run_forward(loop_count=params[0])
            elif cmd == "gen":
                self._run_generate(write_index=params[0], inner_loop_count=params[1], inner_increment=params[2], outer_loop_count=params[3], outer_increment=params[4])
            elif cmd == "bwd":
                assert len(params) == 1
                self._run_backward(zero_grad=False, loop_count=params[0])
            elif cmd == "bwd_zero_grad":
                assert len(params) == 1
                self._run_backward(zero_grad=True, loop_count=params[0])
            elif cmd == "opt":
                self._run_optimizer()
            elif cmd == "sch":
                self._step_schedulers()
            elif cmd == "quit":
                break
            else:
                raise RuntimeError(f"Invalid feeder thread command: {cmd}")

    def schedule_run_forward(self, loop_count: int):
        if self.feeder_thread_queue:
            self._schedule_feeder_cmd(("fwd", loop_count))
        else:
            self._run_forward(loop_count)

    def schedule_run_generate(self, write_index: int, inner_loop_count: int, inner_increment: int, outer_loop_count: int, outer_increment: int):
        if self.feeder_thread_queue:
            self._schedule_feeder_cmd(("gen", write_index, inner_loop_count, inner_increment, outer_loop_count, outer_increment))
        else:
            self._run_generate(write_index, inner_loop_count, inner_increment, outer_loop_count, outer_increment)

    def schedule_run_backward(self, loop_count: int, zero_grad: bool):
        if self.feeder_thread_queue:
            name = "bwd_zero_grad" if zero_grad else "bwd"
            self._schedule_feeder_cmd((name, loop_count))
        else:
            self._run_backward(zero_grad=zero_grad, loop_count=loop_count)

    def schedule_run_optimizer(self):
        if self.feeder_thread_queue:
            self._schedule_feeder_cmd("opt")
        else:
            self._run_optimizer()

    def schedule_run_schedulers(self, device):
        if self.feeder_thread_queue:
            self._schedule_feeder_cmd("sch")
        else:
            self._step_schedulers()

    def _schedule_feeder_cmd(self, item):
        assert self.feeder_thread_queue
        self.feeder_thread_queue.put(item)
        self.feeder_thread_queue.put("token")

    def _run_forward(self, loop_count: int):
        assert self.be_api
        params = {
            "$p_loop_count": str(loop_count)
        }
        assert self.be_api.run_program("run_fwd_" + f"{self.device.get_active_subgraph()}", params) == BackendStatusCode.Success, "Failed while running fwd program"
        if self.explicit_barrier_between_programs:
            assert self.be_api.wait_for_idle() == BackendStatusCode.Success, "Failed while waiting for idle"

    def _run_generate(self, write_index, inner_loop_count, inner_increment, outer_loop_count, outer_increment):
        assert self.be_api
        if not self.cache_zerod:
            zero_cache = True
            self.cache_zerod = True
        else:
            zero_cache = False
        params = {
            "$p_cache_write_index": str(write_index),
            "$p_inner_loop_count": str(inner_loop_count),
            "$p_inner_increment": str(inner_increment),
            "$p_outer_loop_count": str(outer_loop_count),
            "$p_outer_increment": str(outer_increment),
        }
        assert self.be_api.run_program("run_fwd_" + f"{self.device.get_active_subgraph()}", params) == BackendStatusCode.Success, "Failed while running fwd program"
        if self.explicit_barrier_between_programs:
            assert self.be_api.wait_for_idle() == BackendStatusCode.Success, "Failed while waiting for idle"

    def _run_backward(self, zero_grad: bool, loop_count: int):
        assert self.be_api
        params = {
            "$p_zero_grad" : "True" if zero_grad else "False",
            "$p_loop_count": str(loop_count)
        }
        logger.info("run_backward: zero_grad={}", zero_grad)
        assert self.be_api.run_program("run_bwd_" + f"{self.device.get_active_subgraph()}", params) == BackendStatusCode.Success, "Failed while running bwd program"
        if self.explicit_barrier_between_programs:
            assert self.be_api.wait_for_idle() == BackendStatusCode.Success, "Failed while waiting for idle"

    def _run_optimizer(self):
        assert self.be_api
        assert self.be_api.run_program("run_opt_" + f"{self.device.get_active_subgraph()}", {}) == BackendStatusCode.Success, "Failed while running opt program"
        if self.explicit_barrier_between_programs:
            assert self.be_api.wait_for_idle() == BackendStatusCode.Success, "Failed while waiting for idle"

    def _step_schedulers(self):
        if hasattr(self.device, "scheduler") and self.device.scheduler is not None:
            self.device.scheduler.step()
            self.push_optimizer_parameters(translate=True, only_scheduler_params=True)
        else:
            self.device._step_schedulers()

    @classmethod
    def _capture_tensor(cls, desc: PytorchTensorDesc, q: DramIODesc):
        if not hasattr(cls, "should_capture_tensors"):
            cls.should_capture_tensors = pybuda.ci.capture_tensors()

        if not cls.should_capture_tensors:
            return

        base = pybuda.ci.get_netlist_dir()
        assert isinstance(desc, PytorchTensorDesc)
        tensor = Tensor.create_from_tensor_descriptor(desc).value()
        path = f"{base}/{q.name}"
        assert tensor.shape[0] == q.input_count
        for entry in range(q.input_count):
            pybuda.op.eval.common.dump_tensor(tensor[entry], path, entry=entry)

    def get_output_queue_descriptor(self, output_name) -> DramIODesc:
        desc = self.be_api.get_queue_descriptor(output_name)
        assert translate_addresses(desc) == BackendStatusCode.Success, f"Failed to translate addresses: {desc.name}"
        if output_name in self.compiled_graph_state.output_host_tms:
            tm = self.compiled_graph_state.output_host_tms[output_name]
            desc.hstack_factor = tm.hstack_factor
            desc.vstack_factor = tm.vstack_factor
            desc.stack_row_major = tm.row_major
        return desc

    def get_ordered_output_queues(self) -> List[DramIODesc]:
        assert self.be_api
        ordered_output_queues: List[DramIODesc] = []
        ordered_outputs = self.compiled_graph_state.ordered_output_names
        for output_name in ordered_outputs:
            ordered_output_queues.append(self.get_output_queue_descriptor(output_name))
        return ordered_output_queues

    def get_ordered_bw_output_queues(self) -> List[DramIODesc]:
        """
        For each input, find the queue that holds its gradients (if requires_grad was set), and return its descriptor
        """
        ordered_bw_output_queues: List[DramIODesc] = []
        ordered_input_gradients = self.compiled_graph_state.ordered_input_gradient_names
        for output_name in ordered_input_gradients:
            ordered_bw_output_queues.append(self.get_output_queue_descriptor(output_name))
        return ordered_bw_output_queues
    
    def get_intermediate_activation_queues(self) -> List[DramIODesc]:
        assert self.be_api
        return [self.get_output_queue_descriptor(output_queue_name) for op_name, output_queue_name in self.compiled_graph_state.ordered_intermediate_activation_names]

    @classmethod
    def read_queues(
        cls, 
        queues: List[DramIODesc], 
        original_shapes: List[Tuple[int, ...]], 
        runtime_tensor_transforms: Optional[List[RuntimeTensorTransform]], 
        requires_grad: List[bool], 
        single_output: bool, 
        rd_ptr: int = -1, 
        shutdown_event: Optional[EventClass] = None, 
        clone: bool = False,
        has_microbatch_dim: bool = True
    ) -> List[Tensor]:
        ret = []
        tensors = []
        out_descs = []
        if runtime_tensor_transforms is None:
            runtime_tensor_transforms = [None] * len(queues)
        for i, outq in enumerate(queues):
            logger.debug("Reading output queue {}", outq.name)
            out_desc = PytorchTensorDesc()
            retry_count = 10 # TODO: add control
            timeout = 1
            # Increase timeout for Versim and Emulation device
            if bool(int(os.environ.get("PYBUDA_ENABLE_VERSIM_DEVICE", "0"))):
                retry_count = 100
                timeout = 300
            elif bool(int(os.environ.get("PYBUDA_ENABLE_EMULATION_DEVICE", "0"))):
                retry_count = 100
                timeout = 100
            resp = BackendStatusCode.RuntimeError
            for _ in range(retry_count):
                resp = get_output(outq, out_desc, single_output, timeout, rd_ptr)
                if resp != BackendStatusCode.TimeoutError:
                    break

                logger.debug("{} Reading output queue {} timed out after {}", _, outq.name, timeout)

                if shutdown_event and shutdown_event.is_set():
                    break

            if resp == BackendStatusCode.TimeoutError:
                shutdown_event.set()
                raise RuntimeError("Timeout while reading " + outq.name)

            assert resp == BackendStatusCode.Success, "Error while reading output"
            cls._capture_tensor(out_desc, outq)
            tensors.append(Tensor.create_from_tensor_descriptor(out_desc))
            out_descs.append(out_desc)

        concat_transforms = [transform for transform in runtime_tensor_transforms if transform is not None and transform.type == RuntimeTensorTransformType.Concatenate]
        if len(concat_transforms) > 0:
            def get_index(transform):
                return transform.concat_index
            for i in range(len(concat_transforms) // 2):
                group = [transform for transform in concat_transforms if transform.concat_group == i]
                group.sort(key=get_index)
                tensors_to_concat = [tensors[runtime_tensor_transforms.index(transform)] for transform in group]

                inserted = False
                out_descs_to_remove = []
                for i in range(len(tensors)):
                    if (tensors[i]) in tensors_to_concat:
                        if not inserted:
                            tensor = Tensor.create_from_tensor_descriptor(pytorch_tensor_to_tensor_desc(torch.cat([t.value() for t in tensors_to_concat], dim=group[0].concat_dim)))
                            tensors[i] = tensor
                            inserted = True
                        elif clone:
                            out_descs_to_remove.append(out_descs[i])
                            free_tensor(out_descs[i])
                for out_desc in out_descs_to_remove:
                    out_descs.remove(out_desc)
                tensors = [tensor for tensor in tensors if tensor not in tensors_to_concat]


        assert len(original_shapes) == len(tensors)
        assert len(requires_grad) == len(tensors)
        for i, tensor in enumerate(tensors):
            tensor = tensors[i].narrow_to_original_shape(tuple(original_shapes[i]), runtime_tensor_transforms[i].reinterpreted_shape.as_list() if runtime_tensor_transforms[i] is not None else None, \
                                                     has_microbatch_dim=has_microbatch_dim, unpadded_shape=runtime_tensor_transforms[i].unpadded_shape.as_list() if runtime_tensor_transforms[i] is not None else None)

            if requires_grad[i]:
                tensor = tensor.detach()
                tensor.set_requires_grad(True)

            if clone:
                tensor = tensor.clone()
                free_tensor(out_descs[i])
            ret.append(tensor)
        logger.debug("Done reading queues")
        return ret

    @classmethod
    def pop_queues(cls, queues: List[DramIODesc], single_output: bool):
        for outq in queues:
            logger.debug("Popping from queue {}", outq.name)
            assert pop_output(outq, single_output, 1) == BackendStatusCode.Success, "Error while popping output"

    def _get_ordered_queues(self, names: List[str]) -> List[DramIODesc]:
        ordered_queues = []
        for i, name in enumerate(names):
            desc = self.be_api.get_queue_descriptor(name)
            assert translate_addresses(desc) == BackendStatusCode.Success, f"Failed to translate addresses: {desc.name}"
            ordered_queues.append(desc)
        return ordered_queues

    def get_ordered_input_queues(self) -> List[DramIODesc]:
        return self._get_ordered_queues(self.compiled_graph_state.ordered_input_names)

    def get_ordered_bw_input_queues(self) -> List[DramIODesc]:
        return self._get_ordered_queues(self.compiled_graph_state.ordered_output_gradient_names)

    def get_ordered_target_queues(self) -> List[DramIODesc]:
        return self._get_ordered_queues(self.compiled_graph_state.ordered_target_names)

    @classmethod
    def push_input(cls, queue_desc: DramIODesc, tensor_desc: Union[PytorchTensorDesc, TilizedTensorDesc], single_input: bool = True, timeout_secs: int = 1, ram_address: int = 0):
        if isinstance(tensor_desc, TilizedTensorDesc):
            assert push_input(queue_desc, tensor_desc, timeout_secs, ram_address) == BackendStatusCode.Success, "Error while pushing tilized inputs"
        else:
            assert push_input(queue_desc, tensor_desc, single_input, timeout_secs, ram_address) == BackendStatusCode.Success, "Error while pushing inputs"

    @classmethod
    def push_to_queues(cls, ordered_input_queues: List[DramIODesc], tensors: List[PytorchTensorDesc], single_input: bool):
        assert len(tensors) == len(ordered_input_queues), "Incorrect number of tensors provided on input"
        for i, inq in enumerate(ordered_input_queues):
            logger.debug("Pushing to queue {}", inq.name)
            logger.trace(tensors[i].shape)
            logger.trace(tensor_desc_to_pytorch_tensor(tensors[i]))
            cls._capture_tensor(tensors[i], inq)
            BackendAPI.push_input(inq, tensors[i], single_input, 1, -1) == BackendStatusCode.Success, "Error while pushing inputs"

    def update_device_paramaters(self, parameter_values: Dict[str, torch.Tensor]):
        """
        Push new parameter values to the device
        """
        device_constants_and_parameters = get_device_constant_and_parameters(self.device, updated_parameter_values=parameter_values)
        for parameter_name in self.compiled_graph_state.ordered_parameter_node_names:
            pq = self.be_api.get_queue_descriptor(parameter_name)
            assert translate_addresses(pq) == BackendStatusCode.Success, f"Failed to translate addresses: {pq.name}"
            logger.debug("Pushing to parameter {}", pq.name)
            value = const_eval_tensor(device_constants_and_parameters, self.compiled_graph_state.consteval_trace, self.compiled_graph_state.parameter_to_tile_dims, parameter_name)
            value = detach_tensors([value], fix_non_contiguos=True)[0]
            BackendAPI.push_input(pq, pytorch_tensor_to_tensor_desc(value), True, 1, 0) == BackendStatusCode.Success

    def push_constants_and_parameters(self, translate: bool = False):
        # Push constants
        assert self.be_api

        for constant_name in self.compiled_graph_state.ordered_constant_node_names:
            inq = self.be_api.get_queue_descriptor(constant_name)
            if translate:
                assert translate_addresses(inq) == BackendStatusCode.Success, f"Failed to translate addresses: {inq.name}"
            value = self.compiled_graph_state.get_constant_tensor(constant_name)
            logger.debug("Pushing to constant {}", inq.name)
            df = None
            if inq.data_format == DataFormat.RawUInt32:
                df = DataFormat.RawUInt32
            BackendAPI.push_input(inq, pytorch_tensor_to_tensor_desc(value, df), True, 1, -1) == BackendStatusCode.Success

        # Push parameters
        for parameter_name in self.compiled_graph_state.ordered_parameter_node_names:
            pq = self.be_api.get_queue_descriptor(parameter_name)
            if translate:
                assert translate_addresses(pq) == BackendStatusCode.Success, f"Failed to translate addresses: {pq.name}"
            logger.debug("Pushing to parameter {}", pq.name)
            value = self.compiled_graph_state.get_parameter_tensor(parameter_name)
            BackendAPI.push_input(pq, pytorch_tensor_to_tensor_desc(value), True, 1, 0) == BackendStatusCode.Success

    def push_optimizer_parameters(self, translate: bool = False, only_scheduler_params: bool = False):
        assert self.be_api
        if only_scheduler_params:
            params_to_push = self.device.get_scheduler_params(is_buda=True)
        else:
            params_to_push = self.device.get_optimizer_params(is_buda=True)

        for param_name, opt_params in params_to_push.items():
            for input_name, param_key in self.compiled_graph_state.optimizer_param_info[param_name]:
                if param_key not in opt_params:
                    # If only_scheduler_params, opt_params contains subset of param keys
                    continue
                tensor = opt_params[param_key]
                assert tensor is not None, f"Optimizer parameter tensor missing for {param_name} / {param_key}"

                opq = self.be_api.get_queue_descriptor(input_name)
                if translate:
                    assert translate_addresses(opq) == BackendStatusCode.Success, f"Failed to translate addresses: {opq.name}"

                value = const_eval_tensor({input_name: tensor.value()}, self.compiled_graph_state.consteval_trace, self.compiled_graph_state.parameter_to_tile_dims, input_name)
                value = detach_tensors([value], fix_non_contiguos=True)[0]
                logger.debug("Pushing to optimizer parameter {}", opq.name)
                BackendAPI.push_input(opq, pytorch_tensor_to_tensor_desc(value), True, 1, 0) == BackendStatusCode.Success


