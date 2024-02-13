# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

# Set home directory paths for pybuda and buda
def set_home_paths():
    import sys
    import pathlib
    from loguru import logger
    pybuda_path = pathlib.Path(__file__).parent.parent.resolve()
    if os.path.exists(str(pybuda_path) + "/budabackend"):
        # deployment path
        base_path = str(pybuda_path)
        out_path = "."
    else:
        # DEV path
        pybuda_path = pybuda_path.parent.resolve()
        assert os.path.exists(str(pybuda_path) + "/third_party/budabackend"), "Can't find budabackend"
        base_path = str(pybuda_path) + "/third_party"
        out_path = str(base_path) + "/third_party/budabackend/tt_build"

    if "PYBUDA_HOME" not in os.environ:
        os.environ["PYBUDA_HOME"] = str(pybuda_path)
    if "BUDA_HOME" not in os.environ:
        os.environ["BUDA_HOME"] = str(base_path) + "/budabackend/"
    if "TVM_HOME" not in os.environ:
        os.environ["TVM_HOME"] = str(base_path) + "/tvm"
    if "BUDA_OUT" not in os.environ:
        os.environ["BUDA_OUT"] = str(out_path)
    if "LOGGER_FILE" in os.environ:
        sys.stdout = open(os.environ["LOGGER_FILE"], "w")
        logger.remove()
        logger.add(sys.stdout)

set_home_paths()

# eliminate tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from .module import Module, PyTorchModule, PyBudaModule, TFModule, TFGraphDefModule, OnnxModule, MXNetModule, JaxModule, TFLiteModule
from .tti import TTDeviceImage
from .device import Device 
from .cpudevice import CPUDevice
from .gpudevice import GPUDevice
from .ttdevice import TTDevice
from .ttcluster import TTCluster
from .run import run_inference, run_training, shutdown, initialize_pipeline, run_forward, run_backward, run_optimizer, run_schedulers, run_generate, run_generative_inference, get_parameter_checkpoint, get_parameter_gradients, update_device_parameters, error_raised, get_loss_queue, sync, get_intermediates_queue
from .compile import pybuda_compile
from .torch_compile import compile_torch#, get_default_device, get_available_devices, torch_device
from .compiled_graph_state import CompiledGraphState 
from .config import CompilerConfig, CompileDepth, set_configuration_options, set_epoch_break, set_chip_break, override_op_size, PerfTraceLevel, insert_buffering_nop, insert_nop, _internal_insert_fj_buffering_nop, override_dram_queue_placement, configure_mixed_precision
from .verify import VerifyConfig
from .pybudaglobal import pybuda_reset, set_device_pipeline, is_silicon, get_tenstorrent_device
from .parameter import Parameter
from .tensor import Tensor, SomeTensor, TensorShape
from .optimizers import SGD, Adam, AdamW, LAMB, LARS
from ._C.backend_api import BackendType, BackendDevice
from ._C import DataFormat, MathFidelity
from ._C import k_dim
from .run.api import detect_available_devices

import pybuda.op as op
import pybuda.transformers
