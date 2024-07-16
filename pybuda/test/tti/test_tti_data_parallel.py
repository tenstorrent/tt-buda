#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List, Tuple
import shutil
import pybuda
import pybuda.backend
import torch
import os
import inspect
from pybuda.pybudaglobal import pybuda_reset
import numpy as np
from pybuda.tools.tti_data_parallel import (
    RunMode,
    RunResult,
    ForwardInputs,
    GenerativeInputs,
    MultiCardRunner,
    split_tensor_batch,
    initialize_multicard_runner,
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


def generate_random_inputs(sample_inputs, count):
    input_shapes = [t.shape for t in sample_inputs]
    all_inputs = []
    for _ in range(count):
        inputs = []
        for shape in input_shapes:
            inputs.append(torch.randn(*shape))
        
        all_inputs.append(inputs)
        
    return all_inputs

def test_tti_mmio_dp_sanity():
    clean_env = os.environ.copy()
    device_list = pybuda.detect_available_devices()
    assert device_list, "No devices available"
    
    mmio_device_ids = [[i] for i in range(len(device_list))]
    arch = device_list[0]
    num_loops = 2
    num_inputs_per_loop = 2
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
    }

    output_dir = "device_images_multi_mmio/"
    os.makedirs(output_dir, exist_ok=True)
    for model, config in model_to_config.items():
        model_config = get_model_config(base_kwargs, model, config)
        duts, inputs, targets, other = model_config
        multi_inputs = [inputs] * num_inputs_per_loop
        module = duts['tt']
        image_path = os.path.join(output_dir, f"{model}.tti")
        
        single_device_inputs = split_tensor_batch(inputs, len(mmio_device_ids))[0]
        compile_and_save_tti(
            module=module,
            arch=arch,
            num_chips=1,
            tti_output_path=image_path,
            sample_inputs=single_device_inputs,
        )

        # Generate device outputs
        runner = initialize_multicard_runner(
            arch=pybuda.BackendDevice.Wormhole_B0,
            device_ids=mmio_device_ids,
            run_mode=RunMode.FORWARD,
            compile_inputs=inputs,
            precompiled_tti_path=image_path,
            output_dir=output_dir
        )
        
        all_outputs = []
        for _ in range(num_loops):
            run_result: RunResult = runner.run(ForwardInputs(run_inputs=multi_inputs))
            all_outputs.append(run_result.outputs)
            
        runner.shutdown()
        
        # Generate cpu outputs
        all_cpu_outputs = []
        for single_inputs in multi_inputs:
            all_cpu_outputs.append(module.cpu_eval_forward(*single_inputs))
            
        all_cpu_outputs = [all_cpu_outputs] * num_loops
        
        # Compare outputs, check PCC
        check_outputs(all_outputs, all_cpu_outputs)
            
        pybuda_reset()
        os.environ = clean_env
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
