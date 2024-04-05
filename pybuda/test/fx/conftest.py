import torch
import pytest

from pybuda.torch_compile import compile_torch
from pybuda.config import remove_cpu_fallback_ops

@pytest.fixture(autouse=True)
def disable_embedding_fallback():
    remove_cpu_fallback_ops("embedding")
    yield

torch._dynamo.reset()
def generic_model_test(src_model, num_inputs = 1, num_outputs = 1, inputs = []):
    # Generic runner for models
    model = torch.compile(src_model.to('tt'), backend=compile_torch)

    for _ in range(3):
        if len(inputs) == 0:
            inputs = [torch.rand(1, 128, 768) for _ in range(num_inputs)]
        device = 'tt'
        tt_inputs = [i.to(device) for i in inputs]
        tt_res = model(*tt_inputs)
        if num_outputs > 0:
            tt_res = tuple([tt_res.to('cpu')]) if isinstance(tt_res, torch.Tensor) else tuple([t.to('cpu') for t in tt_res])

        cpu_res = src_model.to('cpu')(*inputs)
        if isinstance(cpu_res, torch.Tensor):
            cpu_res = tuple([cpu_res])

        for i in range(num_outputs):
            assert torch.allclose(cpu_res[i], tt_res[i], atol=0, rtol=1e-2), f"** MISMATCH **\nCPU:\n{cpu_res[i]}\nTT:\n{tt_res[i]}"


