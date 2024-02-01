# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple, Union, List, Any, Optional
from enum import Enum
from pybuda.tensor import Tensor
import torch

class CommandType(Enum):
    QUIT = 1
    RUN_FORWARD = 2
    RUN_BACKWARD = 3
    RUN_OPTIMIZER = 4
    RUN_SCHEDULERS = 5
    GET_PARAMETER_CHECKPOINT = 6
    GET_PARAMETER_GRADIENTS = 7
    COMPILE = 8
    UPDATE_DEVICE_PARAMETERS = 9
    GET_QUEUES = 11
    SET_QUEUES = 12
    DC_TRANSFER = 13
    CPUEVAL_FORWARD = 14
    CPUEVAL_BACKWARD = 15
    CPUEVAL_LOSS = 16
    SYNC = 17
    RUN_GENERATE = 18

class Command:
    """
    Command sent to running processes indicating what they need to run, and for how long.
    """
    def __init__(self, command_type: CommandType, params: Dict[str, Any] = {}):
        self.command_type = command_type
        self.params: Dict[str, Any] = params

    def __repr__(self):
        return f"{self.command_type}: {self.params}"
        
    @classmethod
    def quit(cls) -> "Command":
        return Command(CommandType.QUIT)

    @classmethod
    def run_forward(cls, loop_count: int) -> "Command":
        return Command(CommandType.RUN_FORWARD, {"loop_count": loop_count})

    @classmethod
    def run_backward(cls, loop_count: int, zero_grad: bool) -> "Command":
        return Command(CommandType.RUN_BACKWARD, {"loop_count": loop_count, "zero_grad": zero_grad})

    @classmethod
    def run_generate(cls, loop_count: int, write_index: int, tokens_per_iter: int, token_id: int) -> "Command":
        return Command(CommandType.RUN_GENERATE, {"loop_count": loop_count, "write_index": write_index, "tokens_per_iter": tokens_per_iter, "token_id": token_id})

    @classmethod
    def run_optimizer(cls) -> "Command":
        return Command(CommandType.RUN_OPTIMIZER, {})
    
    @classmethod
    def run_schedulers(cls) -> "Command":
        return Command(CommandType.RUN_SCHEDULERS, {})

    @classmethod
    def compile(cls,
            inputs: Tuple["Tensor", ...],
            compiler_cfg: "CompilerConfig",
            targets: List["Tensor"],
            microbatch_size: int,
            microbatch_count: int,
            verify_cfg: "VerifyConfig") -> "Command":

        # Detach inputs in case they were calculated through some formulas before being pushed in
        if compiler_cfg.compile_subgraphs:
            input_groups = []
            for group in inputs:
                input_groups.append([t.detach() for t in group])
            detached_inputs = input_groups
        else:
            detached_inputs = [t.detach() for t in inputs]
        return Command(CommandType.COMPILE,
                {
                    "inputs": detached_inputs,
                    "compiler_cfg": compiler_cfg,
                    "targets": targets,
                    "microbatch_size": microbatch_size,
                    "microbatch_count": microbatch_count,
                    "verify_cfg": verify_cfg,
                })


    @classmethod
    def get_queues(cls, queue_type: str) -> "Command":
        return Command(CommandType.GET_QUEUES, {"queue_type": queue_type})

    @classmethod
    def set_queues(cls, direction: str, queues: List["DramIODesc"], tile_broadcast_dims: Optional[List[List[int]]], 
            original_shapes: Optional[List[Tuple[int, ...]]], requires_grad: Optional[List[bool]],
            runtime_tensor_transforms: Optional[List["RuntimeTensorTransform"]],
            constant_inputs: Optional[List[Tensor]],
            tile_dims: Optional[List[List[int]]]) -> "Command":
        return Command(CommandType.SET_QUEUES, 
                {
                    "direction": direction,
                    "queues": queues,
                    "tile_broadcast_dims": tile_broadcast_dims,
                    "original_shapes": original_shapes,
                    "requires_grad": requires_grad,
                    "runtime_tensor_transforms": runtime_tensor_transforms,
                    "constant_inputs": constant_inputs,
                    "tile_dims": tile_dims,
                })

    @classmethod
    def dc_transfer(cls, direction: str) -> "Command":
        return Command(CommandType.DC_TRANSFER, {"direction": direction})

    @classmethod
    def cpueval_forward(cls, inputs: List[torch.Tensor], parameters: Dict[str, torch.Tensor], save_for_backward: bool, targets: List[torch.Tensor]) -> "Command":
        return Command(CommandType.CPUEVAL_FORWARD, {"inputs": inputs, "parameters": parameters, "save_for_backward": save_for_backward, "targets": targets})

    @classmethod
    def cpueval_backward(cls, bw_inputs: List[torch.Tensor], parameters: Dict[str, torch.Tensor]) -> "Command":
        return Command(CommandType.CPUEVAL_BACKWARD, { "bw_inputs": bw_inputs, "parameters": parameters })

    @classmethod
    def cpueval_loss(cls, fw_outputs: List[torch.Tensor], targets: List[torch.Tensor], scale_loss: float) -> "Command":
        return Command(CommandType.CPUEVAL_LOSS, {"fw_outputs": fw_outputs, "targets": targets, "scale_loss": scale_loss})

    @classmethod
    def get_parameter_checkpoint(cls) -> "Command":
        return Command(CommandType.GET_PARAMETER_CHECKPOINT, {})

    @classmethod
    def get_parameter_gradients(cls) -> "Command":
        return Command(CommandType.GET_PARAMETER_GRADIENTS, {})

    @classmethod
    def update_device_parameters(cls, params: Dict[str, torch.Tensor]) -> "Command":
        return Command(CommandType.UPDATE_DEVICE_PARAMETERS, {"parameters": params})

    @classmethod
    def sync(cls) -> "Command":
        return Command(CommandType.SYNC, {})
