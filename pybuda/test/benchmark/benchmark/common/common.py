# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import torch
import functools
from dataclasses import dataclass

import pybuda
from pybuda._C.backend_api import BackendType, BackendDevice

MODELS = {}

# decorator
def benchmark_model(configs=[]):
    def benchmark_decorator(model_func):
        name = model_func.__name__
        global MODELS
        MODELS[name] = {"func": model_func, "configs": configs}
        
        @functools.wraps(model_func)
        def wrapper(*args, **kwargs):
            ret = model_func(*args, **kwargs)

            # Check that the right data has been returned
            err_msg = "benchmark_model function should return 2 value: a list of up to 3 modules to run on 'cpu-pre', 'tt device', 'cpu-post', and list of sample inputs"
            assert len(ret) == 2, err_msg
            assert isinstance(ret[0], list), err_msg
            assert len(ret[0]) > 0 and len(ret[0]) <= 3, err_msg
            for m in ret[0]:
                assert isinstance(m, pybuda.Module), err_msg
            assert isinstance(ret[1], list), err_msg
            assert len(ret[1]) > 0, err_msg
            for s in ret[1]:
                assert isinstance(s, torch.Tensor), err_msg

            return ret

        return wrapper

    return benchmark_decorator

def df_from_str(df: str) -> pybuda.DataFormat:
    if (df == "Fp32"):
        return pybuda.DataFormat.Float32

    if (df == "Fp16"):
        return pybuda.DataFormat.Float16

    if (df == "Fp16_b"):
        return pybuda.DataFormat.Float16_b

    if (df == "Bfp8"):
        return pybuda.DataFormat.Bfp8

    if (df == "Bfp8_b"):
        return pybuda.DataFormat.Bfp8_b

    if (df == "Bfp4"):
        return pybuda.DataFormat.Bfp4

    if (df == "Bfp4_b"):
        return pybuda.DataFormat.Bfp4_b

    raise RuntimeError("Unknown format: " + df)

def mf_from_str(mf: str) -> pybuda.MathFidelity:

    if (mf == "LoFi"):
        return pybuda.MathFidelity.LoFi

    if (mf == "HiFi2"):
        return pybuda.MathFidelity.HiFi2

    if (mf == "HiFi3"):
        return pybuda.MathFidelity.HiFi3

    if (mf == "HiFi4"):
        return pybuda.MathFidelity.HiFi4

    raise RuntimeError("Unknown math fidelity: " + mf)

def trace_from_str(trace: str) -> pybuda.PerfTraceLevel:

    if (trace == "none"):
        return pybuda.PerfTraceLevel.NONE

    if (trace == "light"):
        return pybuda.PerfTraceLevel.LIGHT

    if (trace == "verbose"):
        return pybuda.PerfTraceLevel.VERBOSE

    raise RuntimeError("Unknown trace type: " + trace)

def get_models():
    return MODELS


@dataclass
class TestDevice:
    devtype: BackendType
    arch: BackendDevice


def generate_test_device(devtype: str = "silicon", arch: str = "wormhole_b0") -> TestDevice:
    """
    Generate a test device.

    Args:
        devtype (str, optional): Device type (silicon or golden). Defaults to "silicon".
        arch (str, optional): Device architecture (Wormhole B0 or Grayskull). Defaults to "wh_b0".

    Returns:
        TestDevice: Test device similar to one used for PyTest.
    """
    arch = BackendDevice.from_string(arch) if arch else None
    devtype = BackendType.from_string(devtype) if devtype else None
    
    if not arch:
        arch = BackendDevice.Wormhole_B0
    if not devtype:
        devtype = BackendType.Silicon
    
    return TestDevice(devtype=devtype, arch=arch)
