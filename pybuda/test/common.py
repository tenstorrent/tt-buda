# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import pybuda
from pybuda import is_silicon
from pybuda import PyBudaModule, PyTorchModule, TTDevice, pybuda_compile, Parameter, BackendType
from pybuda.verify import verify_module
from .conftest import TestDevice

from loguru import logger

def pcc(a, b):
    return np.min(
            np.corrcoef(torch.squeeze(a.to(torch.float)).detach().numpy().flatten(), torch.squeeze(b.to(torch.float)).detach().numpy().flatten()
        ))



def create_microbatches(epochs, steps, accumulation_steps, microbatch_count, micro_batch_size, seed=None):
    if seed:
        torch.manual_seed(seed)
    tensor = torch.rand(1, epochs * steps * accumulation_steps * microbatch_count * micro_batch_size, 32, 32)
    return torch.split(tensor, micro_batch_size, 1)


class ModuleBuilder(PyBudaModule):
    def __init__(self, forward_fn, **kwargs):
        super().__init__(self.__class__.__name__ + "." + forward_fn.__name__)
        self.forward_fn = forward_fn
        self.kwargs = {}
        for key, value in kwargs.items():
            if type(value) is Parameter:
                setattr(self, key, value)
                self.kwargs[key] = getattr(self, key)
            else:
                self.kwargs[key] = value

    def forward(self, *inputs):
        return self.forward_fn(*inputs, **self.kwargs)


class PyTorchModuleBuilder(PyTorchModule):
    def __init__(self, forward_fn, **kwargs):
        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.forward_fn = forward_fn
                self.kwargs = {}
                for key, value in kwargs.items():
                    if type(value) is torch.nn.Parameter:
                        setattr(self, key, value)
                        self.kwargs[key] = getattr(self, key)
                    else:
                        self.kwargs[key] = value

            def forward(self, *args):
                return self.forward_fn(*args, **self.kwargs)
        super().__init__(self.__class__.__name__ + "_" + forward_fn.__name__, _M())


def device(**device_kwargs):
    def decorator(compile_fn):
        def wrapper(*args, **kwargs):
            assert "device" not in kwargs, "device already specified in @compile kwargs"
            kwargs["device"] = TTDevice("tt0", **device_kwargs)
            return compile_fn(*args, **kwargs)

        return wrapper

    return decorator


def compile(
    optimizer_create_fn=None,
    device_create_fn=None,
    devtype=BackendType.Golden,
    chip_ids=[0],
    **compile_kwargs
):
    """
    Special compile_kwargs:
        optimizer_create_fn: Function(module: Module) -> Optimizer
            Callback to create an Optimizer

        device_create_fn: Function(optimizer: Optimizer) -> Device
            Callback to create a device

        chip_ids:
            List of chips to run on (optional)

        Everything else gets passed through to pybuda_compile
    """

    def decorator(test_fn):
        def wrapper(*activations, **kwargs):
            create_optimizer = (
                optimizer_create_fn
                if optimizer_create_fn is not None
                else lambda mod: None
            )
            create_device = (
                device_create_fn
                if device_create_fn is not None
                else lambda opt: TTDevice(
                    "tt0", devtype=devtype, optimizer=opt, chip_ids=chip_ids
                )
            )
            module = ModuleBuilder(test_fn, **kwargs)
            optimizer = create_optimizer(module)
            device = create_device(optimizer)
            device.place_module(module)
            return pybuda_compile(
                device, test_fn.__name__, *activations, **compile_kwargs
            )

        return wrapper

    return decorator


def run(verify_cfg, module_cls=ModuleBuilder, num_inputs=1, **compile_kwargs):
    """
    verify_cfg can be one of:
        VerifyConfig
        TestDevice

    num_inputs: int
        By default, we invoke verify_module(..) with a single set of inputs. This will repeat the inputs num_inputs times.

    Special compile_kwargs:
        Everything gets passed through to verify_module
    """
    if isinstance(verify_cfg, TestDevice):
        verify_cfg = pybuda.VerifyConfig(devtype=verify_cfg.devtype, arch=verify_cfg.arch, devmode=verify_cfg.devmode, tti_path=verify_cfg.tti_path)

    def decorator(test_fn):
        def wrapper(*activations, **kwargs):
            module = module_cls(test_fn, **kwargs)
            assert "inputs" not in compile_kwargs, "'inputs' cannot appear in compile_kwargs, automatically populated"
            return verify_module(module, [tuple(t.shape) for  t in activations] * num_inputs, verify_cfg, inputs=[activations]*num_inputs, **compile_kwargs)

        return wrapper

    return decorator


def run_torch(verify_cfg, **compile_kwargs):
    return run(verify_cfg, module_cls=PyTorchModuleBuilder, **compile_kwargs)


def create_sgd_optimizer(learning_rate):
    def create_fn(mod):
        sgd_optimizer = pybuda.optimizers.SGD(
            learning_rate=learning_rate,
            parameters=mod.get_parameters(),
        )
        sgd_optimizer.set_optimizer_parameters()
        return sgd_optimizer

    return create_fn
