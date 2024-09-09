# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import threading
from enum import Enum
from typing import List, Optional, Union, Tuple
import queue
import os

from multiprocessing.synchronize import Event as EventClass
from queue import Queue
import torch
import torch.multiprocessing as mp
from loguru import logger

from .backend import BackendAPI
from .tensor import Tensor, pytorch_tensor_to_tensor_desc, is_equivalent_data_format, pad_pytorch_tensor_to_buda
from .utils import detach_tensors, align_up
from pybuda._C.backend_api import DramIODesc, PytorchTensorDesc
from pybuda._C.graph import RuntimeTensorTransform, RuntimeTensorTransformType, Shape
from pybuda._C import DataFormat
from .pybudaglobal import TILE_DIM, create_queue

class TransferType(Enum):
    MP_QUEUE = 1 # read from / write to a queue in shared memory (on host)
    DIRECT = 2   # read/write directly (tilize/untilize)
    NONE = 3     # no explicit transfer (i.e. device will do it on its own), so wrapper does nothing

class DeviceConnector:
    """
    DeviceConnector is a light-weight gasket between two devices, providing mechanism to push/pop data. It
    abstracts the mechanism for pushing and popping out, while implementing data transfer through mp queuees,
    direct tilize/untilize, etc.

    All structures within the class can be pickled and sent to other processes.
    """
    def __init__(self, 
            push_type: TransferType, 
            pop_type: TransferType, 
            shutdown_event: Optional[EventClass],
            queue: Optional[Queue] = None,
            side_queue: Optional[Queue] = None):

        self.push_type = push_type
        self.pop_type = pop_type
        self.shutdown_event = shutdown_event # if the event fires, any blocking actions should stop

        if queue is not None:
            self.queue = queue
        elif self.pop_type == TransferType.MP_QUEUE:
            mp_context = mp.get_context('spawn')
            self.queue = create_queue(mp_context)

        self.side_queue = side_queue

    def shutdown(self):
        pass # children will override

    def initialize(self):
        pass # children will override

    def push_to_side_queue(self, tensors: List[Tensor], clone: bool = False):
        """
        Push to side queue, if one is set, to store debug data
        """
        if self.side_queue is not None:
            if clone:
                tensors = [t.clone() for t in tensors]
            tensors = [t.detach() for t in tensors]
            while True:
                try:
                    self.side_queue.put(tensors) # TODO: timeout and break on shutdown_event
                    return
                except queue.Full as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Aborting side queue put due to shutdown event")
                        return [] # got a signal to shutdown and end the process
                    continue


    def push(self, tensors: List[Tensor]):

        self.push_to_side_queue(tensors)
        if self.push_type == TransferType.MP_QUEUE:
            while True:
                try:
                    self.queue.put(tensors) # TODO: timeout and break on shutdown_event
                    return
                except queue.Full as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Aborting queue put due to shutdown event")
                        return [] # got a signal to shutdown and end the process
                    continue

        
        raise RuntimeError(f"Can't handle push to this type: {type(self)}")

    def read(self) -> List[Tensor]:

        if self.queue is not None:
            while True:
                try:
                    data = self.queue.get(timeout=0.1)
                    return data
                except queue.Empty as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Aborting queue get due to shutdown event")
                        return [] # got a signal to shutdown and end the process
                    continue

        raise RuntimeError("No queue to read from")

    def pop(self):
        if self.queue is not None:
            return # no-op

        raise RuntimeError("Can't handle pop")

    def transfer(self, blocking: bool):
        pass # NOP by default, implemented by some versions

    def set_dram_io_pop_queues(self, _: List[DramIODesc]):
        pass

    def set_dram_io_push_queues(self, _: List[DramIODesc], __: List[List[int]], ___: Optional[List[RuntimeTensorTransform]], ____: Optional[List[Tensor]] = None):
        pass

    def empty(self) -> bool:
        if self.queue is None:
            raise RuntimeError("This type of connector can't be polled for emptiness")
        return self.queue.empty()

class DirectPusherDeviceConnector(DeviceConnector):
    """
    Connector in which case one device directly pushes (tilizes) to the other
    """
    def __init__(self, shutdown_event: Optional[EventClass], sequential: bool, pop_type: TransferType = TransferType.NONE, side_queue: Optional[queue.Queue] = None, microbatch=1):
        super().__init__(push_type=TransferType.DIRECT, pop_type=pop_type, shutdown_event=shutdown_event, side_queue=side_queue)
        self.direct_push_queues = None # Will be set after compile
        self.sequential = sequential
        self.tile_broadcast_dims = None
        self.runtime_tensor_transforms : List[RuntimeTensorTransform] = None
        self.constant_tensors = None
        self.microbatch = microbatch
        self.pusher_thread = None

    def pusher_thread_main(self, cmdqueue: queue.Queue):
        logger.info("Pusher thread on {} starting", self)
        while True:
            while True:
                try:
                    cmd = cmdqueue.get(timeout=0.1)
                    break
                except queue.Empty as _:
                    if self.shutdown_event is not None and self.shutdown_event.is_set():
                        logger.debug("Ending pusher thread on {} due to shutdown event", self)
                        return # got a signal to shutdown and end the process
                    continue

            if cmd == "quit":
                return

            logger.debug("Pusher thread pushing tensors")
            self._internal_push(cmd)

    def shutdown(self):
        if self.pusher_thread:
            self.pusher_thread_queue.put("quit")

    def initialize(self):
        # Create threads
        if not self.sequential and not self.pusher_thread:
            self.pusher_thread_queue = queue.Queue(maxsize=3) # don't allow pushes to go too far ahead, or we'll run out of memory
            self.pusher_thread = threading.Thread(target=self.pusher_thread_main, args=(self.pusher_thread_queue,))
            self.pusher_thread.start()

    def _convert_tensor_for_tilize(self, tensor: Tensor, q: DramIODesc) -> Tensor:
        """
        Convert formats to closest supported format, depending on the destination queue
        """
        if is_equivalent_data_format(tensor.pt_data_format, q.data_format):
            return tensor

        pt_format = tensor.value().dtype
        if not tensor.value().is_floating_point():
            return tensor.to_format(q.data_format)

        if q.data_format in [DataFormat.Float16, DataFormat.Bfp8, DataFormat.Bfp4, DataFormat.Bfp2]:
            # tensor has to be float16
            if pt_format != torch.float16:
                return tensor.to_format(DataFormat.Float16)
            return tensor

        if q.data_format in [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b]:
            # tensor can be bfloat or fp32
            if not pt_format in [torch.float32, torch.bfloat16]:
                return tensor.to_format(DataFormat.Float16_b)
            return tensor

        # Don't know what format it is... leave as-is and let back-end convert
        return tensor

    def _embedding_index(self, tensor: torch.Tensor, original_shape: Tuple[int, ...], q: DramIODesc) -> Tensor:
        assert q.data_format in [DataFormat.RawUInt8, DataFormat.RawUInt16, DataFormat.RawUInt32]
        assert len(tensor.shape) <= 2, "Must be a 1d tensor"
        assert len(original_shape) <= 1 or original_shape[-2] == 1, "Must be a 1d tensor"
        assert len(original_shape) <= 2 or original_shape[-3] == 1, "Must be a 1d tensor"

        q_rt = q.bufq_grid_dim_r * q.mblock_m * q.ublock_rt
        w = tensor.shape[0] if len(tensor.shape) > 1 else 1
        pad = align_up(tensor.shape[-1], TILE_DIM) - tensor.shape[-1]
        tensor = torch.nn.functional.pad(tensor, (0, pad))
        tensor = tensor.reshape(w, 1, 1, tensor.shape[-1])
        tensor[:, :, :, original_shape[-1]:] = ~torch.tensor(0, dtype=tensor.dtype)
        tensor = tensor.view(w, q_rt, -1, TILE_DIM)
        pad = align_up(tensor.shape[-2], TILE_DIM) - tensor.shape[-2]
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad))
        tensor = tensor.view(w, q_rt, -1, TILE_DIM, TILE_DIM)
        tensor = tensor.transpose(2, 3).view(w, 1, q_rt * TILE_DIM, -1)

        assert len(tensor.shape) == 4, "_embedding_index: rank changed"
        assert tensor.shape[0] == w, "_embedding_index: w changed"
        assert tensor.shape[1] == q.t, "_embedding_index: t changed"
        assert tensor.shape[2] == (q.bufq_grid_dim_r * q.mblock_m * q.ublock_rt * TILE_DIM), "_embedding_index: tensor dims mismatch q dims"
        assert tensor.shape[3] == (q.bufq_grid_dim_c * q.mblock_n * q.ublock_ct * TILE_DIM), "_embedding_index: tensor dims mismatch q dims"
        return tensor

    def _internal_push(self, tensors: List[Tensor]):

        tensor_dtypes = [None] * len(tensors)
        if not self.direct_push_queues:
            print(f"Direct push queues have not been set for {self}")
        assert self.direct_push_queues, "Direct push queues have not been set"
        assert self.tile_broadcast_dims is not None
        is_data_parallel = int(os.getenv("PYBUDA_N300_DATA_PARALLEL", "0"))
        assert len(tensors) == len(self.direct_push_queues) or is_data_parallel and len(tensors) * 2 == len(self.direct_push_queues), (
                f"Incorrect number of tensors provided on input: {len(tensors)} vs {len(self.direct_push_queues)}")
        assert self.runtime_tensor_transforms, "Runtime tensor transforms have not been set"
        assert len(tensors) == len(self.runtime_tensor_transforms) or is_data_parallel and len(tensors) * 2 == len(self.runtime_tensor_transforms)

        self.push_to_side_queue(tensors)

        # Convert to supported tilize conversion format, if needed
        if isinstance(tensors, tuple):
            tensors = list(tensors)

        for i, t in enumerate(tensors):
            if isinstance(t, Tensor):
                tensors[i] = self._convert_tensor_for_tilize(t, self.direct_push_queues[i])
            else:
                tensors[i] = t

        if is_data_parallel:
            new_tensors = []
            new_tensor_dtypes = []
            for i, t in enumerate(tensors):
                if isinstance(tensors[i], Tensor):
                    tensors[i] = tensors[i].to_pytorch()

                newtensor = tensors[i][int(tensors[i].shape[0] / 2):]
                newshape = tensors[i].shape
                tensors[i] = tensors[i][:int(tensors[i].shape[0] / 2)]
                new_tensors.append(newtensor)
                new_tensor_dtypes.append(tensor_dtypes[i])

            tensors = tensors + new_tensors
            tensor_dtypes = tensor_dtypes + new_tensor_dtypes

        # Handles RuntimeTensorTransform::ReinterpretShape
        for i, t in enumerate(tensors):
            if self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.EmbeddingIndex:
                if isinstance(tensors[i], Tensor):
                    t = t.value()
                assert t is not None
                t = self._embedding_index(t, self.runtime_tensor_transforms[i].original_shape, self.direct_push_queues[i])
                tensors[i] = t
                tensor_dtypes[i] = DataFormat.RawUInt32
            elif self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.ConstantInput:
                assert self.constant_tensors[i] is not None
                tensors[i] = self.constant_tensors[i]
                t = tensors[i]

            if isinstance(tensors[i], torch.Tensor):
                if self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.ReinterpretShape:
                    # TODO: RuntimeTensorTransform could do this transform (for all the RuntimeTensorTransformTypes)
                    t = t.contiguous().view(self.runtime_tensor_transforms[i].reinterpreted_shape.as_list())
                elif self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.Prestride:
                    continue
                tile_r = self.tile_dims[i][0] if self.tile_dims is not None else TILE_DIM
                tile_c = self.tile_dims[i][1] if self.tile_dims is not None else TILE_DIM
                tensors[i] = pad_pytorch_tensor_to_buda(
                    t, self.tile_broadcast_dims[i], squeeze=True, microbatch=self.microbatch, tile_r=tile_r, tile_c=tile_c)
            else:
                reinterpreted_shape = None
                if self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.ReinterpretShape:
                    reinterpreted_shape = self.runtime_tensor_transforms[i].reinterpreted_shape.as_list()
                    tensors[i] = t.to_buda_shape(self.tile_broadcast_dims[i], reinterpret_shape=reinterpreted_shape, clone=False, squeeze=True, microbatch=self.microbatch)
                elif self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.Prestride:
                    pass
                elif self.runtime_tensor_transforms[i].type == RuntimeTensorTransformType.NoTransform:
                    tensors[i] = t.to_buda_shape(self.tile_broadcast_dims[i], reinterpret_shape=None, clone=False, squeeze=True, microbatch=self.microbatch)


        def to_tensor_desc(t: Union[Tensor, torch.Tensor], type: Union[DataFormat, None]) -> PytorchTensorDesc:
            if isinstance(t, Tensor):
                return t.to_tensor_desc()
            return pytorch_tensor_to_tensor_desc(t, df=type)

        # TODO: create tensor_desc list with double queues and half tensors
        BackendAPI.push_to_queues(self.direct_push_queues, [to_tensor_desc(t, type) for t, type in zip(tensors, tensor_dtypes)], single_input=False)
        self.save_tensors = tensors

    def push(self, tensors: List[Tensor]):

        if not self.sequential:
            self.pusher_thread_queue.put(tensors)
        else:
            self._internal_push(tensors)

    def set_dram_io_push_queues(
            self, direct_push_queues: List[DramIODesc],
            tile_broadcast_dims: List[List[int]],
            runtime_tensor_transforms: Optional[List[RuntimeTensorTransform]],
            constant_tensors: Optional[List[Tensor]] = None,
            tile_dims: Optional[List[List[int]]] = None):

        self.direct_push_queues = direct_push_queues
        self.tile_broadcast_dims = tile_broadcast_dims
        self.runtime_tensor_transforms = runtime_tensor_transforms if runtime_tensor_transforms is not None else [RuntimeTensorTransform() for _ in range(len(direct_push_queues))]
        self.constant_tensors = constant_tensors if constant_tensors is not None else [None for _ in range(len(direct_push_queues))]
        self.tile_dims = tile_dims

class DirectPopperDeviceConnector(DeviceConnector):
    """
    Connector in which case one device produces data directly into queues, and other pops from them
    """
    def __init__(self, shutdown_event: Optional[EventClass], side_queue: Optional[queue.Queue] = None):
        super().__init__(push_type=TransferType.NONE, pop_type=TransferType.DIRECT, shutdown_event=shutdown_event, side_queue=side_queue)
        self.direct_pop_queues = None # Will be set after compile
        self.original_shapes = None
        self.runtime_tensor_transforms = None

    def read(self) -> List[Tensor]:
        assert self.direct_pop_queues is not None, "Direct pop queues have not been set"
        if len(self.direct_pop_queues) == 0:
            return []
        assert self.original_shapes is not None
        ret = BackendAPI.read_queues(self.direct_pop_queues, self.original_shapes, self.runtime_tensor_transforms, requires_grad=self.requires_grad, single_output=False, shutdown_event=self.shutdown_event, clone=False)
        self.push_to_side_queue(ret)
        return ret

    def pop(self):
        assert self.direct_pop_queues is not None, "Direct pop queues have not been set"
        if len(self.direct_pop_queues) == 0:
            return
        BackendAPI.pop_queues(self.direct_pop_queues, single_output=False)

    def set_dram_io_pop_queues(self, direct_pop_queues: List[DramIODesc], original_shapes: List[Tuple[int, ...]], requires_grad: List[bool], runtime_tensor_transforms: Optional[List[RuntimeTensorTransform]]):
        self.direct_pop_queues = direct_pop_queues
        self.original_shapes = original_shapes
        self.requires_grad = requires_grad
        self.runtime_tensor_transforms = runtime_tensor_transforms

class DirectPusherPopperDeviceConnector(DirectPusherDeviceConnector):
    """
    Connector between two direct devices (i.e. TT devices)
    """
    def __init__(self, shutdown_event: Optional[EventClass], sequential: bool, side_queue: Optional[queue.Queue] = None):
        super().__init__(pop_type=TransferType.DIRECT, shutdown_event=shutdown_event, sequential=sequential, side_queue=side_queue)
        self.direct_pop_queues = None # Will be set after compile
        self.original_shapes = None
        self.runtime_tensor_transforms = None

    def read(self) -> List[Tensor]:
        assert self.direct_pop_queues is not None, "Direct pop queues have not been set"
        if len(self.direct_pop_queues) == 0:
            return []
        assert self.original_shapes is not None
        ret = BackendAPI.read_queues(self.direct_pop_queues, self.original_shapes, self.runtime_tensor_transforms, requires_grad=self.requires_grad, single_output=False, shutdown_event=self.shutdown_event, clone=True)
        self.push_to_side_queue(ret)
        return ret
        
    def pop(self):
        assert self.direct_pop_queues is not None, "Direct pop queues have not been set"
        if len(self.direct_pop_queues) == 0:
            return
        BackendAPI.pop_queues(self.direct_pop_queues, single_output=False)

    def set_dram_io_pop_queues(self, direct_pop_queues: List[DramIODesc], original_shapes: List[Tuple[int, ...]], requires_grad: List[bool], runtime_tensor_transforms: Optional[List[RuntimeTensorTransform]]):
        self.direct_pop_queues = direct_pop_queues
        self.original_shapes = original_shapes
        self.requires_grad = requires_grad
        self.runtime_tensor_transforms = runtime_tensor_transforms

    def transfer(self, blocking: bool):
        """
        Transfer a piece of data from src to dest
        """
        data = self.read()
        self.push(data)


class InputQueueDirectPusherDeviceConnector(DirectPusherDeviceConnector):
    """
    Connector from which we can read, from the given queue, but there are no pushes. This is typically the first
    device in the pipeline.

    It implementes a "transfer" function to transfer 1 set of inputs from the queue into the device.
    """
    def __init__(self, q: Queue, shutdown_event: Optional[EventClass], sequential: bool):
        super().__init__(shutdown_event, sequential)
        self.queue = q

    def transfer(self, blocking: bool):
        """
        Transfer a piece of data from queue to device, if there are any. Optionally block.
        """
        if not blocking and self.queue.empty():
            return 

        data = self.read()
        self.push(data)

class OutputQueueDirectPoppperDeviceConnector(DirectPopperDeviceConnector):
    """
    Connector that has an external queue that pushes go to. No reading through this connector is allowed.

    It implementes a "transfer" function to transfer 1 set of outputs from device to the queue
    """
    def __init__(self, q: Queue, shutdown_event: Optional[EventClass], side_queue: Optional[queue.Queue] = None):
        super().__init__(shutdown_event, side_queue=side_queue)
        self.queue = q

    def transfer(self, blocking: bool):
        """
        Transfer a piece of data from device to read queue. Optionally blocking.
        """
        if not blocking:
            raise NotImplementedError("Non-blocking transfer on output not implemented yet")

        data = self.read()
        self.queue.put([t.clone().detach() for t in data])  # Need to clone, otherwise popping will erase the tensor
        self.pop()
