# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import os
import pybuda
import sys
import torch
import types
import io
import json
from contextlib import redirect_stdout
from pybuda._C.graph import get_constant_input_value, Graph
from pybuda._C.backend_api import translate_addresses
from pybuda._C.torch_device import get_default_device, push_tensor, is_created_on_device, original_shape, PyBudaTensorDesc, CompileRequest, Program 
from loguru import logger
from pybuda.capture_fx_graph import append_to_graph
from pybuda.tensor import const_eval_tensor, do_runtime_transform
from pybuda.compiled_graph_state import CompiledGraphState
_tt0 = None
_compile_cache = None
_compile_cache_dir = os.environ.get("PYBUDA_COMPILE_CACHE_DIR", "tt_build")
_graph = None
_subgraph_index = 0
_module_index = 0
"""
There are dummy enums defined in pytorch, like PrivateUse1 that can be used
for bringing up new device types.  Eventually we should mainline an enum for
Tenstorrent and not have to use the dummy device slot.
"""
# torch2.0 only
torch.utils.rename_privateuse1_backend("tt")

def reset_state():
    global _tt0
    if _tt0 is not None:
        _tt0.graph = None
    _tt0 = None
    global _compile_cache 
    _compile_cache = None
    global _graph
    _graph = None
    global _subgraph_index
    _subgraph_index = 0
    # do not reset module index, we need unique name for bbe compile in case filename cannot be extracted
    logger.debug("Resetting state")

def extract_filename(module):
    code = ""
    with io.StringIO() as buf, redirect_stdout(buf):
        module.print_readable()
        code = buf.getvalue()
    start = code.find("# File: ")
    if start == -1:
        return None
    filename = code[start + 8:]
    end = filename.find(",")
    if end == -1:
        end = filename.find(":")
        if end == -1:
            return None
    filename = filename[:end]

    # take out path
    end = filename.rfind("/")
    if end == -1:
        return filename
    else:
        return filename[end + 1:]

def torch_device(index=0):
    return get_available_devices()[index].torch_device()


def _build_backend_compile_request(device, compiler_cfg, compiled_graph_state, subgraph_idx):
    soc_desc_yaml = (
        compiler_cfg.backend_device_descriptor_path
        if compiler_cfg.backend_device_descriptor_path == ""
        else device.soc_desc_yaml
    )

    cluster_yaml = (
        compiler_cfg.backend_cluster_descriptor_path
        if compiler_cfg.backend_cluster_descriptor_path == ""
        else device.cluster_yaml
    )

    # Backend Compile
    bcfg = pybuda._C.backend_api.BackendConfig(
        device.type,
        device.arch,
        pybuda._C.backend_api.DeviceMode.CompileAndRun,
        compiler_cfg.backend_opt_level,
        compiler_cfg.backend_output_dir,
        soc_desc_yaml,
        cluster_yaml,
    )

    inputs = [
        PyBudaTensorDesc(name, shape)
        for name, shape in zip(
            compiled_graph_state.get_ordered_input_names_for_subgraph(subgraph_idx), compiled_graph_state.get_ordered_input_shapes_for_subgraph(subgraph_idx)
        )
    ]

    input_runtime_transforms = {}
    for i in range(subgraph_idx + 1):
        input_runtime_transforms[i] = [
            json.dumps(transform.to_json()) for transform in compiled_graph_state.get_ordered_input_runtime_transforms_for_subgraph(i)
        ]

    input_tile_bcast_dims = {}
    for i in range(subgraph_idx + 1):
        input_tile_bcast_dims[i] = compiled_graph_state.get_ordered_input_tile_broadcast_dims_for_subgraph(i)

    constants = [
        PyBudaTensorDesc(
            name,
            constant.shape,
            constant=constant,
        )
        for name, constant in compiled_graph_state.post_const_eval_constants.items()
    ]

    parameters = [
        PyBudaTensorDesc(name, param.shape, ptr=0)
        for name, param in compiled_graph_state.post_const_eval_parameters.items()
    ]

    outputs = [
        PyBudaTensorDesc(name, shape)
        for name, shape in zip(
            compiled_graph_state.get_ordered_output_names_for_subgraph(subgraph_idx), compiled_graph_state.get_ordered_output_shapes_for_subgraph(subgraph_idx)
        )
    ]
    output_runtime_transforms = {}
    for i in range(subgraph_idx + 1):
        output_runtime_transforms[i] = [
            json.dumps(transform.to_json()) for transform in compiled_graph_state.get_ordered_output_runtime_transforms_for_subgraph(i)
        ]

    logger.debug("Build CompileRequest")
    return CompileRequest(
        compiled_graph_state.netlist_filename,
        compiler_cfg.backend_output_dir,
        bcfg,
        inputs,
        input_runtime_transforms,
        input_tile_bcast_dims,
        constants,
        parameters,
        outputs,
        output_runtime_transforms,
    )


def _compile(module, aten_module, module_name, sample_inputs, device, compiler_cfg):
    global _tt0
    global _subgraph_index
    global _graph

    if _tt0 is None:
        _tt0 = pybuda.TTDevice("tt0", arch=device.arch)
    else:
        _tt0.remove_modules()

    _tt0.place_module(pybuda.module.PyTorchModule(module_name, module))

    if _graph is None:
        logger.debug("Creating New graph")
        _graph = Graph(module_name)
    
    assert (
        _tt0.arch == device.arch
    ), f"Mismatch in the arch compiling for vs the currently bound device {_tt0.arch} != {device.arch}"
    assert (
        _tt0.devtype == device.type
    ), f"Mismatch in the arch compiling for vs the currently bound device {_tt0.devtype} != {device.type}"

    # Frontend Compile
    logger.debug("Appending to Graph")
    _graph, intermediate_tensors, output_tensors = append_to_graph(_graph, module, aten_module, sample_inputs, _subgraph_index)
    logger.debug(f"Appending to graph done, captured {len(_graph.nodes())} nodes")
    _subgraph_index += 1
    _tt0.graph = _graph.clone()
    _tt0.intermediate_tensors = intermediate_tensors
    _tt0.output_tensors = [pybuda.Tensor.create_from_torch(output_tensor) for output_tensor in output_tensors]
    logger.debug("Frontend Compile")
    module = module.to("cpu")
    fe_compile_result = pybuda.compile.pybuda_compile(
        _tt0,
        module_name,
        *[pybuda.Tensor.create_from_torch(sample_input.to("cpu")) for sample_input in sample_inputs],
        compiler_cfg=compiler_cfg,
        microbatch_size=sample_inputs[0].shape[0],
        # TODO: support all arguments
    )

    # Backend Compile
    logger.debug("Backend Compile")
    compiled_graph_state = CompiledGraphState.from_compiled_graph(_tt0, fe_compile_result)
    workload = device.compile(
        _build_backend_compile_request(device, compiler_cfg, compiled_graph_state, _subgraph_index - 1)
    )

    return workload, compiled_graph_state


def _create_compile_key(module, module_name, sample_inputs, device, compiler_cfg):
    m = hashlib.sha256()
    m.update(id(module).to_bytes(8, "little"))
    for i in sample_inputs:
        for s in i.shape:
            m.update(s.to_bytes(8, "little"))
    # TODO: It'd probably be better to encode the contents of said files
    m.update(device.soc_desc_yaml.encode("utf-8"))
    m.update(device.cluster_yaml.encode("utf-8"))
    m.update(compiler_cfg.to_json().encode("utf-8"))
    return f"{module_name}_{m.hexdigest()}"


def _populate_compile_cache():
    compile_cache = {}
    return compile_cache


def _compile_cached(module, aten_module, module_name, sample_inputs, device, compiler_cfg, cache):
    global _compile_cache
    global _tt0

    key = None

    default_output_dir = compiler_cfg.backend_output_dir == "tt_build/test_out"
    if cache and default_output_dir:
        if _compile_cache is None:
            _compile_cache = _populate_compile_cache()
        key = _create_compile_key(
            module, module_name, sample_inputs, device, compiler_cfg
        )
        logger.debug(f"Created compile key {key}")
        compiler_cfg.backend_output_dir = f"{_compile_cache_dir}/{key}"
        if key in _compile_cache:
            return _compile_cache[key]
    elif cache and not default_output_dir:
        logger.warning(
            "PyBuda compile cache disabled because of user compiler_cfg.backend_output_dir path override"
        )
    else:
        compiler_cfg.backend_output_dir = pybuda.utils.resolve_output_build_directory()

    workload, compiled_graph_state = _compile(module, aten_module, module_name, sample_inputs, device, compiler_cfg)

    if key is not None:
        _compile_cache[key] = workload
    return workload, compiled_graph_state

class compiledModel(torch.nn.Module):
    def __init__(self, module, device, workload, compiled_graph_state, index):
        super().__init__()
        self.module = module
        self.device = device
        self.workload = workload
        self.compiled_graph_state = compiled_graph_state
        self.index = index

    # Submit work to device
    def forward(self, *inputs, **kwargs):
        logger.debug("Invoke Submit")
        assert type(inputs) is tuple

        inputs = tuple([i.to(self.device.torch_device()) for i in inputs])
        for i, input in enumerate(inputs):
            if input.device != self.device.torch_device():
                raise RuntimeError(
                    f"Input tensor[{i}] device[{str(input.device)}] != Compilation device[{str(self.device)}]"
                )

        loop_count = 1
        program_params = {"$p_loop_count": str(loop_count)}
        program = Program(f"run_fwd_{self.index}", program_params)
        logger.info(f"Running run_fwd_{self.index}")

        outputs = self.device.dispatch(self.workload, [program], list(inputs), self.compiled_graph_state.output_host_tms, self.index)
        return outputs
    
    def to(self, dev):
        for desc in self.workload.parameters:
            name = desc.name
            value = self.compiled_graph_state.post_const_eval_parameters[name]
            push_tensor(self.device.backend.get_queue_descriptor(desc.name), desc, value, "")

        for desc in self.workload.constants:
            name = desc.name
            value = self.compiled_graph_state.post_const_eval_constants[name]
            push_tensor(self.device.backend.get_queue_descriptor(desc.name), desc, value, "")
        # self.module.to(dev)

from torch._decomp import core_aten_decompositions, get_decompositions
from torch._functorch.aot_autograd import aot_module_simplified, aot_function, aot_module
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch.fx.interpreter import Interpreter
from torch.fx import subgraph_rewriter

from torch._functorch.compile_utils import strip_overloads

from pybuda.torch_decomp_reconstruct import get_pybuda_decompositions, apply_torch_reconstruct_patterns

def compile_torch(
    module,
    sample_inputs,
    options=None,
):
    torch_device = list(module.parameters())[0].device if len(list(module.parameters())) > 0 else "tt"
    with torch.no_grad():
        pybuda_decompositions = get_pybuda_decompositions()
        decompositions = {**core_aten_decompositions(), **pybuda_decompositions}
        fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(sample_inputs)
        fake_tensor_mode.allow_non_fake_inputs = True
        aten = make_fx(module, tracing_mode="symbolic", decomposition_table=decompositions, _allow_non_fake_inputs=True)(*sample_inputs)
        apply_torch_reconstruct_patterns(aten)
        return _torch_compile(module, sample_inputs, aten, original_torch_device=torch_device)

_device = None
def _torch_compile(
    module,
    sample_inputs,
    aten_module,
    device=None,
    compiler_cfg=None,
    module_name=None,
    cache=False,  # Disabled for now
    original_torch_device=None,
):
    """
    Ideally we can remove having to pass in tt0 (the ttdevice.py) object here,
    but currently it's so coupled to our compile flow that it's too much work to
    remove its dependency for this proof of concept.
    Ideally pybuda.compile.pybuda_compile just takes a device_config dataclass
    which has the device target information to decouple it from the runtime device.
    """
    logger.info("Torch Compile")
    strip_overloads(aten_module)

    # TODO: Remove me, needs to be persistant
    global _device
    if _device is None:
        _device = get_default_device()
    device = _device

    if compiler_cfg is None:
        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.compile_subgraphs = True

    if module_name is None:
        filename = extract_filename(module)
        if filename is not None:
            module_name = f"{filename}_{_subgraph_index}"
        else:
            global _module_index
            module_name = f"{module.__class__.__name__}_{_module_index}"
            _module_index += 1

    cache &= not bool(os.environ.get("PYBUDA_DISABLE_COMPILE_CACHE", "0"))

    rand_inputs = [torch.rand(sample_input.shape).to(sample_input.dtype).to("cpu") for sample_input in sample_inputs]

    workload, compiled_graph_state = _compile_cached(
        module, aten_module, module_name, rand_inputs, device, compiler_cfg, cache
    )

    compiled_model = compiledModel(module, device, workload, compiled_graph_state, _subgraph_index-1)
    # Push parameters and constants to device
    compiled_model.to(device.torch_device())
    logger.info("Done Torch Compile")
    if original_torch_device is not None:
        module = module.to(original_torch_device)
    return compiled_model

# compile_torch = aot_autograd(fw_compiler=_torch_compile, decompositions={**core_aten_decompositions(), **pybuda_decompositions})
