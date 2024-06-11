#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pybuda
import pybuda.backend
import torch
import os
import pybuda
import inspect
from pybuda.pybudaglobal import pybuda_reset
import numpy as np
from pybuda.tools.tti_data_parallel import (
    run_tti_data_parallel, 
    RunMode,
    RunResult, 
    ForwardRunInputs,
    GenerativeRunInputs
)
import sys
sys.path.insert(1, "pybuda")
sys.path.insert(1, 'pybuda/test')

from benchmark.benchmark.common import get_models
import benchmark.benchmark.models.resnet
import benchmark.benchmark.models.bert
import benchmark.benchmark.models.mobilenet_v2
import benchmark.benchmark.models.t5

sys.path.remove("pybuda")
sys.path.remove('pybuda/test')

def check_outputs(output1, output2):
    assert type(output1) == type(output2)

    if isinstance(output1, (list, tuple)):
        assert len(output1) == len(output2)
        for item1, item2 in zip(output1, output2):
            check_outputs(item1, item2)

    elif isinstance(output1, torch.Tensor):
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape
        pcc = np.corrcoef(output1.to(torch.float32).detach().numpy().flatten(), output2.to(torch.float32).detach().numpy().flatten())[0, 1]
        assert pcc > 0.9

    else:
        raise TypeError(f"Unsupported type: {type(output1)}")

def get_generative_params(other):
    compile_inputs = None
    num_tokens_to_generate = None
    first_current_index = None
    pad_token_id = None
    write_index = 0
    if bool(other):
        if "compile_inputs" in other:
            compile_inputs = other["compile_inputs"]
        if "max_length" in other:
            num_tokens_to_generate = other["max_length"]
        if "first_current_index" in other:  # TODO
            first_current_index = other["first_current_index"]
        if "pad_token_id" in other:  # TODO
            pad_token_id = other["pad_token_id"]
        if "write_index" in other: # TODO
            write_index = other["write_index"]
        
    return compile_inputs, num_tokens_to_generate, first_current_index, pad_token_id, write_index

if __name__ == "__main__":
    device_list = pybuda.detect_available_devices()
    if not device_list:
        raise RuntimeError("No devices available")
    
    base_kwargs = {
        "training": False, 
        "microbatch": 128, 
        "data_type": 'Fp16_b',
        "math_fidelity": 'HiFi3',
        "arch": "wormhole_b0",
        "devtype": "silicon",
    }
    
    model_to_config = {
        "resnet": "resnet50", 
        "bert": "base", 
        "mobilenet_v2": "224"
    }
    
    mmio_device_ids = list(range(len(device_list)))
    arch = device_list[0]
    
    output_dir="device_images/"
    os.makedirs(output_dir, exist_ok=True)
    
    
    num_loops = 16
    total_microbatch_size = 128
    
    models = get_models()
    os.environ["PYBUDA_FORCE_THREADS"] = "1"
    clean_env = os.environ.copy()
    
    for model, config in model_to_config.items():
        kwargs = base_kwargs.copy()
        func = models[model]["func"]
        available_parameters = inspect.signature(func).parameters
        for p in available_parameters:
            if p == "config":
                kwargs["config"] = config
            elif p == "force_num_layers":
                kwargs["force_num_layers"] = 0
        
        model_config = func(**kwargs)
    
        duts, inputs, targets, other = model_config
        module = duts['tt']
        run_result: RunResult = run_tti_data_parallel(
            module=module,
            run_mode=RunMode.FORWARD,
            inputs=ForwardRunInputs(inputs=inputs),
            arch=arch,
            device_ids=mmio_device_ids,
            num_loops=num_loops,
            output_dir=output_dir,
            sync_at_run_start=True
        )
        outputs = run_result.outputs
        cpu_outputs = [module.cpu_eval_forward(*inputs)] * num_loops
        
        check_outputs(cpu_outputs, outputs)
        
        pybuda_reset()
        os.environ = clean_env