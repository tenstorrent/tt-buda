# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import threading
import queue
import time

from typing import Optional, Union, List

from loguru import logger

from .ttdevice import TTDevice
from .optimizers import Optimizer
from .pybudaglobal import is_silicon
from pybuda._C import NodeEpochType, DataFormat
from pybuda._C.backend_api import BackendDevice, BackendType

class TTCluster(TTDevice):
    """
    TTCluster represents a group of Tenstorrent devices on one system. A single model can be spread over all 
    devices in a cluster.
    """
    def __init__(self, 
            name: str, 
            cluster_size: int = 0,
            arch: Optional[BackendDevice] = None,
            devtype: Optional[BackendType] = None, 
            optimizer: Optional[Optimizer] = None,
            fp32_fallback: DataFormat = DataFormat.Float16_b,
            param_fp32_fallback: DataFormat = DataFormat.Float16_b,
            mp_context = None):
        """
        Initialize a cluster of Tenstorrent devices on same system. All parameters pass through to underlying
        individual devices.

        For development and debug purposes, device model or golden-model can be used instead of silicon. Device
        model is the default in development mode.

        Parameters
        ----------
        name: str
            Device name

        cluster_size: int, optional
            Number of devices from the current system to be used in this cluster. If not provided, the maximum will
            be automatically assigned.

        devtype: BackendType, optional
            Type of Tenstorrent device. Only used to run testing on models of devices instead of a real silicon chip.

        optimizer: Optimizer, optional
            PyBuda optimizer to be used on this device. Mandatory if running in training mode.

        fp32_fallback: DataFormat
            If TT device doesn't support FP32, tensors will fall back to this format. Bfloat16 is the default.

        mp_context: mp.context, optional
            Optioanlly override Python multi-processing context when creating mp queues
        """
        super().__init__(name, 0, arch, devtype, optimizer, fp32_fallback, param_fp32_fallback, mp_context)
        self.cluster_size = cluster_size

        # List of op-names to map to ordered chip breaks
        self.device_start_ops: List[str] = []

    def get_cluster_size(self) -> int:
        """
        Return the number of devices in the given cluster.

        Returns
        -------
        int
            Number of devices in the cluster
        """
        # Figure out the number of devices on the system, and make sure that we're only setting 
        # breaks for less or equal to that
        
        if not is_silicon(self.devtype):
            return 1

        if self.cluster_size == 0:
            return buda.get_number_of_chips(self._get_gstate())

        max_chips = buda.get_number_of_chips(self._get_gstate())
        if self.cluster_size > max_chips:
            raise RuntimeError(f"Cluster size ({self.cluster_size}) set to more than available devices ({max_chips})")
        return self.cluster_size
        
    def _init_concurrent_run(self):
        """
        Callback before concurrent processes are launched
        """
        if self.devtype == TTDeviceType.Silicon:
            if self.cluster_size == 0:
                self.cluster_size = self.get_cluster_size()
            self.feeder_thread_queues = [queue.Queue() for _ in range(self.cluster_size)]
            self.feeder_thread = [
                    threading.Thread(target=self._run_feeder_thread, args=(self.feeder_thread_queues[device], device)) 
                    for device in range(self.cluster_size)]

            for t in self.feeder_thread:
                t.start()

    def _send_to_feeder_thread(self, cmd: str, count: int = 1, wait: bool = False):
        """
        Push command `count` times to the feeder threads
        """
        assert self.feeder_thread_queues is not None
        for i in range(count):
            for q in self.feeder_thread_queues:
                q.put(cmd)

        if wait:
            for q in self.feeder_thread_queues:
                while not q.empty():
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        if self.final_barrier is not None:
                            self.final_barrier.abort()
                        return # got a signal to shutdown and end the process
                    time.sleep(0.01)

    def _run_forward(self, copy_inputs: bool):
        """
        Sequential run forward
        """
        assert self.cluster_size > 0
        for device_index in range(self.cluster_size):
            buda.run(self.gstate, 0, NodeEpochType.Forward, self.devtype, copy_inputs, False, self._perf_desc, device_index)

    def _run_backward(self, copy_inputs: bool):
        """
        Sequential run backward
        """
        assert self.cluster_size > 0
        for device_index in reversed(range(self.cluster_size)):
            buda.run(self.gstate, 0, NodeEpochType.Backward, self.devtype, copy_inputs, False, self._perf_desc, device_index)

    def _run_optimizer(self):
        """
        Sequential run of optimizer
        """
        assert self.cluster_size > 0
        for device_index in reversed(range(self.cluster_size)):
            buda.run(self.gstate, 0, NodeEpochType.Optimizer, self.devtype, True, False, self._perf_desc, device_index)

    def _run_forward_on_device(self, device_index: int, copy_inputs: bool = True):
        """
        Run from feeder thread
        """
        buda.run(self.gstate, 0, NodeEpochType.Forward, self.devtype, copy_inputs, False, self._perf_desc, device_index)

    def _run_backward_on_device(self, device_index: bool, copy_inputs: bool = True):
        """
        Run from feeder thread
        """
        buda.run(self.gstate, 0, NodeEpochType.Backward, self.devtype, copy_inputs, False, self._perf_desc, device_index)

    def _run_optimizer_on_device(self, device_index: int):
        """
        Run from feeder thread
        """
        buda.run(self.gstate, 0, NodeEpochType.Optimizer, self.devtype, True, False, self._perf_desc, device_index)

    def _run_zero_grad_on_device(self, device_index: int):
        buda.zero_grad(gstate=self._get_gstate(), graph_id=0, device_name=self.devtype, chip_id=device_index)
        for name, value in self.optimizer.get_param_dict().items():
            buda.push_optimizer_parameter(self._get_gstate(), 0, value, name, self.devtype, device_index)

    def _run_feeder_thread(self, cmdqueue: queue.Queue, device_index: int):
        """
        A thread that feeds the epoch programs into the device, in background
        """

        logger.info("Feeder thread on {}, device index {} starting", self, device_index)
        while True:
            cmd = cmdqueue.get()
            logger.info("Run feeder thread {} cmd: {}", device_index, cmd)
            if cmd == "fwd":
                self._run_forward_on_device(copy_inputs=True, device_index=device_index)
            elif cmd == "bwd":
                self._run_backward_on_device(copy_inputs=True, device_index=device_index)
            elif cmd == "opt":
                self._run_optimizer_on_device(device_index=device_index)
            elif cmd == "zero_grad":
                self._run_zero_grad_on_device(device_index=device_index)
            elif cmd == "quit":
                break
            else:
                raise RuntimeError(f"Invalid feeder thread command: {cmd}")

    def _post_graph_callback(self):
        """
        Called after buda graph has been generated, but the compile process hasn't yet happened.
        """
        TTDevice._post_graph_callback(self)

        if is_silicon(self.devtype):
            # Figure out if we're trying to use too many chips
            self.cluster_size = self.get_cluster_size()
            if len(self.device_start_ops) >= self.cluster_size:
                raise RuntimeError(f"Too many chip breaks ({len(self.device_start_ops)}) set for devices available ({self.cluster_size})")
            if len(self.device_start_ops) + 1 < self.cluster_size:
                self.cluster_size = len(self.device_start_ops) + 1
                logger.info("Reducing cluster size to {} due to smaller number of chip break ops.", self.cluster_size)
           
        # Register chip breaks with gstate
        for op_on_new_chip in self.device_start_ops:
            buda.place_on_new_chip(self._get_gstate(), op_on_new_chip)


    def set_device_start_op(self, op_name: Union[str, List[str]]):
        """
        This call allows manual paritition of a single model across multiple devices on this cluster. Each op provided
        (by name) will the "first" op on the next device in sequence.

        Parameters
        ----------
        op_name: Union[str, List[str]]
            An op, or a list of ops
        """
        if isinstance(op_name, str):
            self.device_start_ops.append(op_name)
        else:
            if not isinstance(op_name, (list, tuple)):
                raise RuntimeError("Invalid op_name parameter")
            self.device_start_ops.extend(op_name)

    def _get_bw_tilizer_target_device_id(self):
        """
        Return the device_id that we push backward inputs to. That's the last device in chain.
        """
        if self.cluster_size == 0:
            self.cluster_size = self.get_cluster_size()
        assert self.cluster_size > 0
        return self.cluster_size - 1

    def _shutdown_threads(self):
        if self.feeder_thread is not None:
            for q in self.feeder_thread_queues:
                q.put("quit")
            for t in self.feeder_thread:
                t.join()
            self.feeder_thread = None
