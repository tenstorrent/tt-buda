# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Optional, Set, Any, Tuple, ClassVar
from dataclasses import dataclass, field

from .archive import TTIArchive
import importlib
import inspect
import os
from pybuda.ttdevice import TTDevice, get_device_config

from pybuda.config import CompilerConfig, _get_global_compiler_config
from pybuda.verify.config import VerifyConfig
from pybuda.utils import get_pybuda_git_hash, get_budabackend_git_hash, as_json, dict_as_json, list_as_json, optional_as_json

from pybuda._C import DataFormat
from pybuda._C.backend_api import BackendDevice, DeviceMode, BackendType
from ..run.api import detect_available_devices

from pybuda.optimizers import Optimizer
from pybuda.compiled_graph_state import CompiledGraphState

import dataclasses
from dataclasses_json import dataclass_json, config 

from loguru import logger

import torch

@dataclass_json
@dataclasses.dataclass()
class TTDeviceImage:
    """
    A TTDeviceImage defines all required state sourced from TTDevice to produce a TTI-archive.
    """
    TTI_VERSION: ClassVar[str] = "1.1.0" 

    # Static device state
    version: str
    device_image_name: str
    arch: BackendDevice = field(metadata=as_json(BackendDevice))
    devtype: BackendType = field(metadata=as_json(BackendType))
    chip_ids: List[int]
    fp32_fallback: DataFormat = field(metadata=as_json(DataFormat))
    optimizer: Optional[Optimizer]
    training: bool
    microbatch_size: int
    microbatch_count: int
    grid_size: List[int] 
    harvested_rows: Optional[List[int]]

    # snapshot of the state generated from pybuda compile
    compiled_graph_state: CompiledGraphState

    # We probably don't need to serialize this but we'll do so to get a static
    # snapshot of both sets of configs.
    verify_cfg: Optional[VerifyConfig]
    compiler_cfg: CompilerConfig

    # snapshot of placed ops/modules onto device 
    loss_module: Optional[str]
    module_name_to_metadata: Dict[str, Dict[str, str]] = field(default_factory=dict)
    modules: List[Any] = field(default_factory=list)

    # generated config:
    #  Note: validation; error by default -> override with flag into warning
    pybuda_pip_version_id: int = field(init=False)
    pybuda_commit_hash: str = field(init=False)
    budabackend_commit_hash: str = field(init=False)

    def __post_init__(self):
        # generated attributes set here since TTDeviceImage default frozen
        object.__setattr__(self, "pybuda_pip_version_id", importlib.metadata.version('pybuda'))
        object.__setattr__(self, "pybuda_commit_hash", get_pybuda_git_hash())
        object.__setattr__(self, "budabackend_commit_hash", get_budabackend_git_hash())

    @staticmethod
    def get_harvested_rows(device: "TTDevice", device_cfg: "DeviceConfig") -> List[int]:
        if device.devtype == BackendType.Golden:
            harvested_rows = []
        else:
            harvested_rows = device_cfg.get_harvested_cfg()

        if len(harvested_rows) > 0:
            harvested_rows = [harvested_rows[c_id] for c_id in device.chip_ids] 
        
        return harvested_rows
    

    @staticmethod
    def create_image_from_device(
        device: "TTDevice",
        training: bool,
        microbatch_count: int,
        verify_cfg: VerifyConfig, 
        compiler_cfg: CompilerConfig,
        cpueval_outputs: Optional[List[torch.Tensor]] = None,
    ) -> "TTDeviceImage": 
        device_cfg = device.get_device_config(compiler_cfg=compiler_cfg)
        grid_size = device_cfg.grid_size 
        device._compiled_graph_state.cpueval_outputs = cpueval_outputs

        device_image = TTDeviceImage(
            version=TTDeviceImage.TTI_VERSION,
            device_image_name=device.name,
            arch=device.arch,
            devtype=device.devtype,
            chip_ids=device.chip_ids,
            fp32_fallback=device.fp32_fallback,
            optimizer=device.optimizer,
            training=training,
            microbatch_size=device._compiled_graph_state.microbatch,
            microbatch_count=microbatch_count,
            compiled_graph_state=device._compiled_graph_state,
            verify_cfg=verify_cfg,
            compiler_cfg=compiler_cfg,
            module_name_to_metadata={
                module.get_name(): {
                    "module": module.__module__,
                    "class": module.__class__.__name__,
                    "module_file_path": os.path.relpath(inspect.getfile(module.__class__), start=os.curdir)
                } for module in device.modules
            },
            loss_module=device.loss_module.get_name() if device.loss_module else None,
            modules=device.modules,
            grid_size=[grid_size.r, grid_size.c],
            harvested_rows=TTDeviceImage.get_harvested_rows(device, device_cfg),
        )
        return device_image

    @staticmethod
    def load_from_disk(tti_file_path: str, device_id_overrides: Optional[List[int]] = None) -> "TTDeviceImage":
        from .archive import TTIArchive
        return TTIArchive.load_from_disk(tti_file_path, device_id_overrides)

    @staticmethod
    def save_to_disk(
        device_image,
        tti_file_path: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "TTDeviceImage":
        from .archive import TTIArchive
        return TTIArchive.save_to_disk(device_image, tti_file_path, *args, **kwargs)
    
    @staticmethod
    def validate_image_version_compatibility(device_image: "TTDeviceImage"):
        runtime_pybuda_public_version_id = importlib.metadata.version('pybuda')
        image_pybuda_public_version_id = device_image.pybuda_pip_version_id
        assert runtime_pybuda_public_version_id == image_pybuda_public_version_id , (
            "Error: Saved image pybuda version does not match with runtime version"
        )
        expected_version, actual_version = TTDeviceImage.TTI_VERSION, device_image.version
        if actual_version is None:
            raise ValueError(f"TTI Version mismatch: expected {expected_version}, but version not found in TTI. TTI recompilation required.")

        if not hasattr(device_image, "version") or expected_version != actual_version:
            raise ValueError(f"TTI Version mismatch: expected {expected_version}, got {actual_version}. TTI recompilation required.")

        runtime_pybuda_commit_hash = get_pybuda_git_hash()
        image_pybuda_commit_hash = device_image.pybuda_commit_hash
        if runtime_pybuda_commit_hash and image_pybuda_commit_hash and runtime_pybuda_commit_hash != image_pybuda_commit_hash:
            logger.warning(
                f"Warning: runtime pybuda_commit_hash is {runtime_pybuda_commit_hash} but "
                  f"device_image pybuda_commit_hash is {image_pybuda_commit_hash}"
            )

        runtime_budabackend_commit_hash = get_budabackend_git_hash()
        image_budabackend_commit_hash = device_image.budabackend_commit_hash
        if runtime_budabackend_commit_hash and image_budabackend_commit_hash and runtime_budabackend_commit_hash != image_budabackend_commit_hash:
            logger.warning(
                f"Warning: runtime budabackend_commit_hash is {runtime_budabackend_commit_hash} but "
                  f"device_image budabackend_commit_hash is {image_budabackend_commit_hash}"
            )
    
    @staticmethod
    def validate_grid_size(device: "TTDevice", device_image: "TTDeviceImage"):
        compiler_cfg = _get_global_compiler_config()
        detected_device_cfg = device.get_device_config(compiler_cfg)
        detected_grid_size = [detected_device_cfg.grid_size.r, detected_device_cfg.grid_size.c]
        if device.arch == BackendDevice.Wormhole_B0:
            assert detected_grid_size[0] >= device_image.grid_size[0], f"Grid-size in device image do not match this device's grid-size. detected-grid-size: {detected_grid_size}, cached-grid-size: {device_image.grid_size}"
            if detected_grid_size[0] > device_image.grid_size[0]:
                logger.info(f"Detected grid size has more rows than cached grid size, overlays will be recompiled later. detected-grid-size: {detected_grid_size}, cached-grid-size: {device_image.grid_size}") 
        else:
            assert detected_grid_size == device_image.grid_size, f"Grid-size in device image do not match this device's grid-size. detected-grid-size: {detected_grid_size}, cached-grid-size: {device_image.grid_size}"

    @staticmethod
    def create_device_from_image(image: "TTDeviceImage") -> "TTDevice":
        """
        Construct a fully-formed TTDevice back to the user.
        """
        TTDeviceImage.validate_image_version_compatibility(image)
        
        device = TTDevice(
            name=image.device_image_name,
            chip_ids=image.chip_ids,
            arch=image.arch,
            devtype=image.devtype,
            optimizer=image.optimizer,
            fp32_fallback=image.fp32_fallback,
        )
        device._compiled = True
        device._compiled_graph_state = image.compiled_graph_state
        device.modules = image.modules
        device.device_mode = DeviceMode.RunOnly

        TTDeviceImage.validate_grid_size(device, image)

        for module in device.modules:
            module._set_device(device)
            if module.get_name() == image.loss_module:
                device.loss_module = module        

        return device

    def is_compiled_for_training(self):
        return self.training

    def info(self):
        """
        Return summary info for the compiled device image back to the user.
        """
        print(
            f"""
            Image Info...
            - Version Info:
                - pybuda_version: {self.pybuda_pip_version_id}
                - pybuda_commit: {self.pybuda_commit_hash}
                - buda_backend_commit: {self.budabackend_commit_hash}
            - Device Name: {self.device_image_name}

            Device Info...
            - arch: {self.arch}
            - chip_ids: {self.chip_ids}
            - backend device type: {self.devtype}
            - grid size: {self.grid_size}
            - harvested rows: {self.harvested_rows if self.harvested_rows else "None"}

            Compilation Graph State...
            - training: {self.training}
            - ordered input shapes: {self.compiled_graph_state.ordered_input_shapes}
            - ordered targets shapes: {self.compiled_graph_state.ordered_target_shapes}
            """
        )
    
    @staticmethod
    def _get_original_shapes(shapes, microbatch) -> List[List[int]]:
        original_shapes = []
        for input_shape in shapes:
            original_shape = input_shape.copy()
            if original_shape and original_shape[0] == 1:
                original_shape[0] = microbatch
            original_shapes.append(original_shape)
        return original_shapes

    def get_input_shapes(self) -> List[List[int]]:
        return TTDeviceImage._get_original_shapes(self.compiled_graph_state.ordered_input_shapes, self.compiled_graph_state.microbatch)

    def get_target_shapes(self) -> List[List[int]]:
        return TTDeviceImage._get_original_shapes(self.compiled_graph_state.ordered_target_shapes, self.compiled_graph_state.microbatch)
