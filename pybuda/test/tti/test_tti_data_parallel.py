#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List
import shutil
import pybuda
import pybuda.backend
import torch
import os
import inspect
from pybuda.pybudaglobal import pybuda_reset
import numpy as np
from pybuda.tools.tti_data_parallel import (
    split_tensor_batch,
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

def compile_and_save_tti(
    module,
    arch: pybuda.BackendDevice,
    tti_output_path: str,
    sample_inputs,
    chip_ids: Optional[List[int]] = None,
    num_chips: Optional[int] = None,
):
    tt0 = pybuda.TTDevice(
        "tt0", 
        module=module,
        chip_ids=chip_ids,
        num_chips=num_chips,
        arch=arch
    )
    tt0.compile_to_image(
        img_path=tti_output_path,
        training=False,
        sample_inputs=sample_inputs,
    )
    pybuda_reset()

def get_model_config(base_kwargs, model, config):
    models = get_models()
    kwargs = base_kwargs.copy()
    func = models[model]["func"]
    available_parameters = inspect.signature(func).parameters
    for p in available_parameters:
        if p == "config":
            kwargs["config"] = config
        elif p == "force_num_layers":
            kwargs["force_num_layers"] = 0

    return func(**kwargs)


def test_tti_mmio_dp_sanity():
    clean_env = os.environ.copy()
    device_list = pybuda.detect_available_devices()
    assert device_list, "No devices available"
    
    mmio_device_ids = [[0]]
    arch = device_list[0]
    num_loops = 16
    total_microbatch_size = 128
    
    base_kwargs = {
        "training": False, 
        "microbatch": total_microbatch_size, 
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

    output_dir = "device_images_multi_mmio/"
    os.makedirs(output_dir, exist_ok=True)
    
    for model, config in model_to_config.items():
        model_config = get_model_config(base_kwargs, model, config)
        duts, inputs, targets, other = model_config
        module = duts['tt']
        image_path = os.path.join(output_dir, f"{model}.tti")
        compile_and_save_tti(
            module=module,
            arch=arch,
            chip_ids=[0],
            tti_output_path=image_path,
            sample_inputs=inputs,
        )
        run_result: RunResult = run_tti_data_parallel(
            precompiled_tti_path=image_path,
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
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

# Sanity test that runs on a single card
def test_tti_n300_dp_sanity():
    clean_env = os.environ.copy()
    device_list = pybuda.detect_available_devices()
    assert device_list, "No devices available"
    assert os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1"
    
    device_ids = [[0, 1]]
    arch = device_list[0]
    num_loops = 16
    total_microbatch_size = 128
        
    base_kwargs = {
        "training": False, 
        "microbatch": total_microbatch_size, 
        "data_type": 'Fp16_b',
        "math_fidelity": 'HiFi3',
        "arch": "wormhole_b0",
        "devtype": "silicon",
    }
    
    model_to_config = {
        "resnet": "resnet50", 
        "bert": "base"
    }
    
    output_dir="device_images_n300_dp/"
    os.makedirs(output_dir, exist_ok=True)
    
    for model, config in model_to_config.items():
        model_config = get_model_config(base_kwargs, model, config)
        duts, inputs, targets, other = model_config
        module = duts['tt']
        image_path = os.path.join(output_dir, f"{model}.tti")
        compile_and_save_tti(
            module=module,
            arch=arch,
            num_chips=1,
            tti_output_path=image_path,
            sample_inputs=inputs,
        )
        run_result: RunResult = run_tti_data_parallel(
            precompiled_tti_path=image_path,
            run_mode=RunMode.FORWARD,
            inputs=ForwardRunInputs(inputs=inputs),
            arch=arch,
            device_ids=device_ids,
            num_loops=num_loops,
            output_dir=output_dir,
            sync_at_run_start=True
        )
        outputs = run_result.outputs
        cpu_outputs = [module.cpu_eval_forward(*inputs)] * num_loops
        check_outputs(cpu_outputs, outputs)
        
        pybuda_reset()
        os.environ = clean_env
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)