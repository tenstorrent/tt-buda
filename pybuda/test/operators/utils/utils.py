# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Operator test utilities

import sys
import pybuda

from enum import Enum
from dataclasses import dataclass
from loguru import logger
from typing import Optional, List, Dict

from pybuda import PyBudaModule, VerifyConfig
from pybuda.op_repo import TensorShape
from pybuda.verify import TestKind, verify_module
from pybuda.config import _get_global_compiler_config
from pybuda._C import MathFidelity
from test.conftest import TestDevice

from ..utils import netlist_utils


class ShapeUtils:

    @staticmethod
    def reduce_microbatch_size(shape: TensorShape) -> TensorShape:
        '''
        Reduce microbatch dimension of a shape to 1
        Usually used for calculating shape of a constant tensor
        '''
        return (1, ) + shape[1:]


@dataclass(frozen=True)
class InputSourceFlag:
    '''Dataclass for specifying compiler flags for specific input source'''
    input_queues_on_host: bool
    set_default_dram_parameters: bool
    default_dram_parameters: Optional[bool]


class InputSourceFlags(Enum):
    '''Enums defining input source flags'''
    FROM_HOST = InputSourceFlag(True, False, None)
    FROM_DRAM = InputSourceFlag(False, False, None)
    FROM_DRAM_PROLOGUED = InputSourceFlag(False, True, False)
    FROM_DRAM_NOT_PROLOGUED = InputSourceFlag(False, True, True)
    FROM_DRAM_PROLOGUE_MICROBATCH_SIZE = InputSourceFlag(False, True, None)


class CompilerUtils:
    '''Utility functions for PyBuda compiler configuration'''

    @staticmethod
    def set_input_source(input_source_flag: InputSourceFlag):
        '''Set compiler configuration for input source'''
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.input_queues_on_host = input_source_flag.input_queues_on_host
        if input_source_flag.set_default_dram_parameters:
            compiler_cfg.default_dram_parameters = input_source_flag.default_dram_parameters

    @staticmethod
    def set_math_fidelity(math_fidelity: MathFidelity):
        '''Set compiler configuration for math fidelity'''
        compiler_cfg = _get_global_compiler_config()
        compiler_cfg.default_math_fidelity = math_fidelity


class VerifyUtils:
    '''Utility functions for PyBuda verification'''

    @staticmethod
    def verify(model: PyBudaModule, test_device: TestDevice, input_shapes: List[TensorShape], input_params: List[Dict] = []):
        '''Perform PyBuda verification on the model

        Args:
            model: PyBuda model
            test_device: TestDevice
            input_shapes: List of input shapes
            input_params: List of input parameters
        '''

        verify_module(
            model,
            input_shapes=input_shapes,
            verify_cfg=VerifyConfig(
                test_kind=TestKind.INFERENCE,
                devtype=test_device.devtype,
                arch=test_device.arch,
            ),
            input_params=[input_params],
        )

    @staticmethod
    def get_netlist_filename() -> str:
        '''Get netlist filename of the last compiled model'''
        return pybuda.pybudaglobal.get_devices()[0]._compile_output.netlist_filename


class NetlistValidation:
    '''Utility functions for netlist validation'''

    def __init__(self):
        self.netlist_filename = VerifyUtils.get_netlist_filename()

    def get_value(self, key_path: str):
        """
        Reads a netlist value from a YAML file based on the given key path.

        Args:
            key_path (str): The key path to the desired value in the YAML file.
                            Keys are separated by slashes ("/").

        Returns:
            The value corresponding to the given key path in the YAML file, or None if the key path is not found.
        """
        return netlist_utils.read_netlist_value(self.netlist_filename, key_path)


class LoggerUtils:
    '''Utility functions for logging'''

    @staticmethod
    def set_log_level(package_name: str, level: str):
        ''' Set log level for package_name and its subpackages

        Args:
            package_name (str): package name
            level (str): log level
        '''
        logger.add(sys.stdout, level=level, filter=lambda record: record["name"].startswith(package_name))
