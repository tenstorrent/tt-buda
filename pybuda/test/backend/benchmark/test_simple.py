# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Simple model end-to-end benchmarks
#
import pytest
import pybuda
import torch
from loguru import logger

import time
def run_benchmark(module, microbatch, input_shapes, cycle_range, fidelity, data_format):

    tt0 = pybuda.TTDevice("tt0", module=module, fp32_fallback=data_format)
    pybuda.set_configuration_options(
            math_fidelity=fidelity,
            performance_trace=pybuda.PerfTraceLevel.VERBOSE)

    loop_count = 1
    inputs = tuple(torch.rand(microbatch, *shape) for shape in input_shapes)
    tt0.push_to_inputs(inputs)

    output_q = pybuda.run_inference(input_count = loop_count + 1, _verify_cfg=pybuda.VerifyConfig.disabled())

    # Wait until compile is done, and first input has gone through
    output_q.get()

    # Starting counting time, and keep pushing inputs and pulling outputs
    start_time = time.time()
    tt0.push_to_inputs(inputs)
    for _ in range(loop_count - 1):
        tt0.push_to_inputs(inputs)
        output_q.get()

    output_q.get()
    end_time = time.time()

    total_time = end_time - start_time
    total_samples = loop_count * microbatch

    logger.info(f"Total time for {total_samples} inputs: {total_time}")
    logger.info(f"Samples/s: {total_samples / total_time}")

    time_per_sample = total_time / total_samples
    frequency = 1.2 * 1000000000
    clocks_per_sample = round(time_per_sample * frequency)

    assert clocks_per_sample >= cycle_range[0] and clocks_per_sample <= cycle_range[1], f"Clocks per sample out of range: {clocks_per_sample} not in {cycle_range}"

class MatmulTest(pybuda.PyBudaModule):
    def __init__(self, name, weight_shape):
        super().__init__(name)
        self.weights = pybuda.Parameter(torch.rand(*weight_shape))

    def forward(self, act):
        return pybuda.op.Matmul("matmul", act, self.weights)

@pytest.mark.skip(reason="Perf not close to expected yet")
def test_matmul():
    weight_shape = (768, 768)
    act_shape = (128, 768)
    pybuda.override_op_size("matmul", (1, 4))
    pybuda.override_op_size("matmul_output_nop_0", (1, 4))
    run_benchmark(MatmulTest("benchmark_matmul", weight_shape), 1024, (act_shape,), 
            cycle_range=(15000, 25000), fidelity=pybuda.MathFidelity.LoFi, data_format=pybuda.DataFormat.Bfp8_b)

