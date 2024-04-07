# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Optional, Set, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os

import torch

import pybuda.optimizers
from pybuda._C.backend_api import BackendType, BackendDevice, DeviceMode
from pybuda._C import DataFormat
from dataclasses_json import dataclass_json
from pybuda.utils import as_json


class TestKind(Enum):
    INFERENCE = 1
    TRAINING = 2
    TRAINING_RECOMPUTE = 3

    def is_training(self) -> bool:
        return self.value in [TestKind.TRAINING_RECOMPUTE.value, TestKind.TRAINING.value]

    def is_recompute(self) -> bool:
        return self.value == TestKind.TRAINING_RECOMPUTE.value

    @classmethod
    def to_json(cls, value):
        return value.name

    @classmethod
    def from_json(cls, value):
        return cls[value.upper()]

    @classmethod
    def get(cls, training: bool, recompute: bool = False) -> "TestKind":
        if not training:
            return cls.INFERENCE

        if recompute:
            return cls.TRAINING_RECOMPUTE

        return cls.TRAINING


class NebulaGalaxy:
    chip_ids = [0, 11, 10, 9, 8, 7, 19, 20, 21, 22, 23, 24, 6, 5, 14, 13, 12, 16, 15, 3, 4, 26, 25, 32, 31, 30, 29, 28, 27, 1, 2, 18, 17] # CI


@dataclass_json
@dataclass
class VerifyConfig:
    graph_name: str = "graph"     # name of the graph/test
    enabled: bool = True
    intermediates: bool = True
    rtol: Dict[Any, Optional[float]] = field(default_factory=lambda: {})  # values per data format
    atol: Dict[Any, Optional[float]] = field(default_factory=lambda: {})  # values per data format
    relative_atol: float = 0.1    # set atol at 10% of the max value in tensor
    pcc: Optional[float] = None   # use Pearson Coefficient Check instead of allclose
    dump_tensors_path: str = ""   # dump nodes at final graph evaluation in a format that can be read in Backend
    run_golden: bool = "PYBUDA_VERIFY_GOLDEN" in os.environ and os.environ["PYBUDA_VERIFY_GOLDEN"] == '1' # run on back-end golden - Legacy, to be replaced by the path below
    run_net2pipe: bool = False    # run netlist through net2pipe
    golden_ignore_df_precision: bool = True # When running golden, run at full FP32 and ignore actual netlist types
    chip_ids: Union[List[int], List[Tuple[int]]] = None  # chip IDs to run on
    num_chips: int = None # number of chips to run on
    verify_each_buda_pass: bool = False # Whether or not to verify tvm outputs after each buda pass
    golden_compare_callback: Optional[Callable[[object, object], bool]] = None # Supply additional golden compare function

    verify_tvm_compile: bool = False         # Should tvm run forward and verify the results
    verify_pipeline_result_vs_framework: bool = False # Compare Framework output on CPU vs module pipline outputs
    verify_pybuda_codegen_vs_framework: bool = False # Compare Framework output on CPU vs pybuda codegen from TVM json graphs

    verify_all: bool = "PYBUDA_FORCE_VERIFY_ALL" in os.environ and os.environ["PYBUDA_FORCE_VERIFY_ALL"] == '1' # Whether or not to verify after every compile stage
    verify_last: bool = True # Whether or not to verify after the final stage (overriden by disabled())
    verify_post_autograd_passes: bool = "PYBUDA_VERIFY_POST_AUTOGRAD_PASSES" in os.environ and os.environ["PYBUDA_VERIFY_POST_AUTOGRAD_PASSES"] == '1'# Whether or not to force verification at post autograd passes (overridden by disabled())
    verify_post_placer: bool = "PYBUDA_VERIFY_POST_PLACER" in os.environ and os.environ["PYBUDA_VERIFY_POST_PLACER"] == '1' # Whether or not to force verification at post placer (overidden by disabled())

    # names of parameters for which gradient error will not fail the test. Some gradients are so small that
    # atol/rtol/pcc will never be good enough to pass
    waive_gradient_errors: Set[str] = field(default_factory=lambda: set()) 

    override_module_outptus = None

    # For auto-testing
    sequential: bool = True
    test_kind: TestKind = field(default=TestKind.INFERENCE, metadata=as_json(TestKind))
    devtype: BackendType = field(default=BackendType.Golden, metadata=as_json(BackendType))
    devmode: DeviceMode = field(default=DeviceMode.CompileAndRun, metadata=as_json(DeviceMode))
    arch: BackendDevice = field(default=BackendDevice.Grayskull if "GOLDEN_WORMHOLE_B0" not in os.environ else BackendDevice.Wormhole_B0, metadata=as_json(BackendDevice))
    scale_loss: float = 50.0  # Loss-scaling to make gradients bigger and easier to verify
    optimizer: Optional[Dict[str, Any]] = field(default_factory=lambda : {"type": "sgd", "params": {"learning_rate": 50.0 } })
    scheduler: Optional[Dict] = None
    epochs: int = 1               # training input counts
    steps: int = 1         
    accumulation_steps: int = 1         
    microbatch_count: int = 1    # this is also the number of inputs for inference
    epoch_breaks: List[str] = field(default_factory=lambda: []) # list of epoch breaks to apply to the model
    fp32_fallback: DataFormat = field(default=DataFormat.Float16_b, metadata=as_json(DataFormat))
    skip_shutdown: bool = False # skip explicit shutdown after verification, leave compiled data around
    tti_path: str = None

    # Optional training queues for verification data
    # Create concrete queues to collect input and weight gradients during training
    checkpoint_interval: int = 1
    enable_input_gradient_checking: bool = True
    enable_parameter_gradient_checking: bool = True
    _input_gradient_queue: Optional[torch.multiprocessing.Queue] = None
    _parameter_gradient_queue: Optional[torch.multiprocessing.Queue] = None

    if "PYBUDA_VERIFY_RESULTS_OFF_BY_DEFAULT" in os.environ and not ("PYBUDA_FORCE_VERIFY_ALL" in os.environ and os.environ["PYBUDA_FORCE_VERIFY_ALL"] == '1'):
        intermediates = False
        run_golden = False
        verify_all = False
        verify_last = False
        verify_post_placer = False
        verify_post_autograd_passes = False
        verify_tvm_compile = False
        verify_pipeline_result_vs_framework = False
        verify_pybuda_codegen_vs_framework = False

    if verify_all:
        verify_pipeline_result_vs_framework = True
        verify_pybuda_codegen_vs_framework = True
        verify_tvm_compile = True
        verify_each_buda_pass = True

    def __post_init__(self):
        # set defaults if not set explicitly by user. Relax under silicon, focus on pcc more.
        if isinstance(self.rtol, (int, float)):
            # User set one value, instead of dict
            self.rtol = {
                torch.float32: self.rtol,
                torch.float16: self.rtol,
                torch.bfloat16: self.rtol}

        if isinstance(self.atol, (int, float)):
            # User set one value, instead of dict
            self.atol = {
                torch.float32: self.atol,
                torch.float16: self.atol,
                torch.bfloat16: self.atol}

        rtol_defaults = {
                torch.float32: None,
                torch.float16: None,
                torch.bfloat16: None,
        }
        atol_defaults = {
                torch.float32: None,
                torch.float16: None,
                torch.bfloat16: None,
        }
        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            if not dt in self.rtol:
                self.rtol[dt] = rtol_defaults[dt]
            if not dt in self.atol:
                self.atol[dt] = atol_defaults[dt]

        if self.pcc is None:
            self.pcc = 0.99

        if "TT_BACKEND_GOLDEN_QUANTIZE" in os.environ:
            self.golden_ignore_df_precision = False

        if self.arch == BackendDevice.Grayskull:
            assert self.fp32_fallback != DataFormat.Float32, "Fallback for f32 cannot be f32 itself"


    @classmethod
    def disabled(cls) -> "VerifyConfig":
        v = VerifyConfig()
        v.enabled = False
        v.verify_last = False
        v.intermediates = False
        v.run_golden = False
        v.scale_loss = 1.0
        v.verify_post_autograd_passes = False
        v.verify_post_placer = False
        return v

    def total_number_of_inputs(self):
        return self.epochs * self.steps * self.accumulation_steps * self.microbatch_count


def should_waive_gradient(param_name, verify_cfg):
    return any([waive_name in param_name for waive_name in verify_cfg.waive_gradient_errors])



# Global verify configutation
g_verify_config: VerifyConfig = VerifyConfig()

def _get_global_verify_config() -> VerifyConfig:
    return g_verify_config

def _clear_global_verify_config():
    global g_verify_config
    g_verify_config = VerifyConfig()

def _set_global_verify_config(config: VerifyConfig):
    global g_compiler_config
    g_compiler_config = config

