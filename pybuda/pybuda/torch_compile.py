# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import pybuda
import torch
import io
import json
from typing import List
from contextlib import redirect_stdout

from loguru import logger

from pybuda._C.torch_device import get_default_device, push_tensor, unique_id, PyBudaTensorDesc, CompileRequest, Program 
from pybuda.compiled_graph_state import CompiledGraphState
from pybuda.fx.capture import CaptureFX
from pybuda.fx.schedule import TensorSource
from pybuda.fx.mixed_graph import MixedGraph


_tt0 = None
_compile_cache = None
_compile_cache_dir = os.environ.get("PYBUDA_COMPILE_CACHE_DIR", "tt_build")
_capture: CaptureFX = CaptureFX()
_subgraph_index = 0
_module_index = 0
_tensor_to_unique_id = {}
_link_subgraph_unique_tensor_ids = []
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
    global _subgraph_index
    _capture.reset_state()
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


def _build_backend_compile_request(device, compiler_cfg, compiled_graph_state, subgraph_idx: int, program_ids: List[int]):
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

    inputs = {}
    for program_id in program_ids:
        graph_idx = MixedGraph.get_program_subgraph_id(subgraph_idx, program_id)
        program_inputs = [
            PyBudaTensorDesc(name, shape)
            for name, shape in zip(
                compiled_graph_state.get_ordered_input_names_for_subgraph(graph_idx), compiled_graph_state.get_ordered_input_shapes_for_subgraph(graph_idx)
            )
        ]
        inputs[graph_idx] = program_inputs

    #input_runtime_transforms = {}
    #for i in range(subgraph_idx + 1):
    #    input_runtime_transforms[i] = [
    #        json.dumps(transform.to_json()) for transform in compiled_graph_state.get_ordered_input_runtime_transforms_for_subgraph(i)
    #    ]
    input_runtime_transforms = device.input_runtime_transforms # append to existing ones
    for program_id in program_ids:
        graph_idx = MixedGraph.get_program_subgraph_id(subgraph_idx, program_id)
        input_runtime_transforms[graph_idx] = [
            json.dumps(transform.to_json()) for transform in compiled_graph_state.get_ordered_input_runtime_transforms_for_subgraph(graph_idx)
        ]

    input_tile_bcast_dims = device.input_tile_bcast_dims # append to existing ones
    for program_id in program_ids:
        graph_idx = MixedGraph.get_program_subgraph_id(subgraph_idx, program_id)
        input_tile_bcast_dims[graph_idx] = compiled_graph_state.get_ordered_input_tile_broadcast_dims_for_subgraph(graph_idx)
    #for i in range(subgraph_idx + 1):
    #    input_tile_bcast_dims[i] = compiled_graph_state.get_ordered_input_tile_broadcast_dims_for_subgraph(i)

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

    outputs = {}
    for program_id in program_ids:
        graph_idx = MixedGraph.get_program_subgraph_id(subgraph_idx, program_id)
        program_outputs = [
            PyBudaTensorDesc(name, shape)
            for name, shape in zip(
                compiled_graph_state.get_ordered_output_names_for_subgraph(graph_idx), compiled_graph_state.get_ordered_output_shapes_for_subgraph(graph_idx)
            )
        ]
        outputs[graph_idx] = program_outputs

    output_runtime_transforms = device.output_runtime_transforms # append to existing ones
    #for i in range(subgraph_idx + 1):
    #    output_runtime_transforms[i] = [
    #        json.dumps(transform.to_json()) for transform in compiled_graph_state.get_ordered_output_runtime_transforms_for_subgraph(i)
    #    ]
    for program_id in program_ids:
        graph_idx = MixedGraph.get_program_subgraph_id(subgraph_idx, program_id)
        output_runtime_transforms[graph_idx] = [
            json.dumps(transform.to_json()) for transform in compiled_graph_state.get_ordered_output_runtime_transforms_for_subgraph(graph_idx)
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

    if os.environ.get("PRINT_PT2_GRAPH", "0") == "1":
        logger.info("Compiling pt2 graph:")
        aten_module.graph.print_tabular()    

    if _tt0 is None:
        _tt0 = pybuda.TTDevice("tt0", arch=device.arch, devtype=device.type)

    _tt0.place_module(pybuda.module.PyTorchModule(module_name, module))

    assert (
        _tt0.arch == device.arch
    ), f"Mismatch in the arch compiling for vs the currently bound device {_tt0.arch} != {device.arch}"
    assert (
        _tt0.devtype == device.type
    ), f"Mismatch in the arch compiling for vs the currently bound device {_tt0.devtype} != {device.type}"

    # Frontend Compile
    logger.debug("Appending to Graph")
    device_graph_changed, graph_inputs, intermediate_tensors, output_tensors, schedule = _capture.append_to_graph(
        module_name, module, aten_module, sample_inputs, _subgraph_index)
    
    _subgraph_index += 1

    if not device_graph_changed:
        return None, None, schedule

    _tt0.graph = _capture.get_buda_graph().clone()
    _tt0.intermediate_tensors = intermediate_tensors
    _tt0.output_tensors = [pybuda.Tensor.create_from_torch(output_tensor) for output_tensor in output_tensors]
    logger.debug("Frontend Compile")
    module = module.to("cpu")

    fe_compile_result = pybuda.compile.pybuda_compile(
        _tt0,
        module_name,
        #*[pybuda.Tensor.create_from_torch(sample_input.to("cpu")) for sample_input in sample_inputs],
        *[pybuda.Tensor.create_from_torch(sample_input.to("cpu")) for sample_input in [g for gs in graph_inputs for g in gs]],
        compiler_cfg=compiler_cfg,
        microbatch_size=sample_inputs[0].shape[0],
        # TODO: support all arguments
    )

    # Backend Compile
    logger.debug("Backend Compile")
    compiled_graph_state = CompiledGraphState.from_compiled_graph(_tt0, fe_compile_result)
    workload = device.compile(
        _build_backend_compile_request(device, compiler_cfg, compiled_graph_state, _subgraph_index - 1, schedule.get_device_program_ids())
    )
    
    return workload, compiled_graph_state, schedule


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

    workload, compiled_graph_state, schedule = _compile(module, aten_module, module_name, sample_inputs, device, compiler_cfg)

    if key is not None and workload is not None:
        _compile_cache[key] = (workload, compiled_graph_state, schedule)
    return workload, compiled_graph_state, schedule

class compiledModel(torch.nn.Module):
    def __init__(self, module, device, workload, compiled_graph_state, schedule, index):
        super().__init__()
        self.module = module
        self.device = device
        self.workload = workload
        self.compiled_graph_state = compiled_graph_state
        self.schedule = schedule
        self.index = index
        self.is_compile = True

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
        if self.compiled_graph_state is not None and not self.compiled_graph_state.has_cache_buffers:
            program_params = {"$p_loop_count": str(loop_count)}
        else:
            program_params = {
            "$p_cache_write_index": str(0),
            "$p_inner_loop_count": str(1),
            "$p_inner_increment": str(1),
            "$p_outer_loop_count": str(1),
            "$p_outer_increment": str(1),
        }
        output_map = {}
        intermediates = {}

        # Run the schedule
        outputs_generated = set()
        logger.info(f"Running subgraph {self.index}")
        for item in self.schedule:
                
            graph_inputs = []
            for i in item.inputs:
                if i.src == TensorSource.INTERMEDIATE:
                    graph_inputs.append(intermediates[i.index].to('tt'))
                elif i.src == TensorSource.INPUT:
                    graph_inputs.append(inputs[i.index])
                else:
                    graph_inputs.append(output_map[i.index])

            #print("graph inputs:")
            #for i, p in enumerate(graph_inputs):
            #    print(" - ", i, ": ", p.to('cpu'))

            if item.fallback:
                # CPU graph
                logger.trace(f"Running fallback graph on CPU: {item.graph_index}")
                #graph_module = torch.fx.GraphModule({}, item.graph)
                graph_inputs = [i.to('cpu') for i in graph_inputs]
                graph_outputs = item.graph_module(*graph_inputs)
                logger.trace(f"Done, produced {len(graph_outputs)} outputs.")
                graph_outputs = tuple(t.to('tt') for t in graph_outputs)
            else:
                # Device - dispatch to device
                program_index = MixedGraph.get_program_subgraph_id(self.index, item.graph_index)
                program = Program(f"run_fwd_{program_index}", program_params)
                logger.debug(f"Running run_fwd_{program_index} on device")
                
                graph_outputs = self.device.dispatch(
                        self.workload, [program], list(graph_inputs), self.compiled_graph_state.output_host_tms, program_index, self.is_compile)
            
                for i, output in enumerate(graph_outputs):
                    if torch.isnan(output.to('cpu')).any(): # debug
                        logger.error(f"Output {i} has NaNs:")
                        logger.error(output.to('cpu'))
                        raise RuntimeError(f"Output {i} is NaN")
                """
                for i, output in enumerate(graph_outputs):
                    print(f"Graph output {i} of {len(graph_outputs)}: shape={output.to('cpu').shape}, desired shape={self.workload.outputs[program_index][i].shape}, item.outputs={item.outputs}")

                for i, output in enumerate(graph_outputs):
                    narrowed = False
                    for dim in range(len(output.shape)):
                        # TODO: reproduce in a smaller test than StableDiffusion
                        if output.shape[dim] != self.workload.outputs[item.graph_index][i].shape[dim]:
                            narrowed = True
                            graph_outputs[i] = output.narrow(dim, 0, self.workload.outputs[item.graph_index][i].shape[dim]).to('cpu').to('tt')

                    # TODO:
                    # If narrowed, the tensor is now on CPU, and can't be used to link to the next graph.. so that needs disabling somehow
                    if not narrowed:
                        graph_outputs[i] = graph_outputs[i].clone()
                """
            
            #print("graph outputs:")
            #for i, p in enumerate(graph_outputs):
            #    print(" - ", i, ": ", p.to('cpu'))


            # Record outputs
            for i, output in enumerate(item.outputs):
                if output.intermediate:
                    intermediates[output.index] = graph_outputs[i]
                else:
                    assert output.index not in outputs_generated
                    output_map[output.index] = graph_outputs[i]
                    outputs_generated.add(output.index)
                    
        # Flatten output map into list
        outputs = [output_map[i] for i in range(len(output_map))]

        for out in outputs:
            _tensor_to_unique_id[unique_id(out)] = out

        _capture.capture_sample_outputs(outputs, self.index)
        # CHeck previous outputs and push to new param queue

        # TODO: We need to do something to clone this in case backend is going to deallocate.... but we don't yet know it it will :(.
        # Instead, we copy to cpu.
        #outputs = [o.to('cpu') for o in outputs]
        return outputs
    
    def to(self, dev):
        if self.workload is None:
            return 

        for desc in self.workload.parameters:
            name = desc.name
            value = self.compiled_graph_state.post_const_eval_parameters[name]
            push_tensor(self.device.backend.get_queue_descriptor(desc.name), desc, value, "")

        for desc in self.workload.constants:
            name = desc.name
            value = self.compiled_graph_state.post_const_eval_constants[name]
            push_tensor(self.device.backend.get_queue_descriptor(desc.name), desc, value, "")
        # self.module.to(dev)


from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.compile_utils import strip_overloads

from pybuda.fx.torch_decomp_reconstruct import get_pybuda_decompositions, apply_torch_reconstruct_patterns

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
    #global _ordered_inputs_per_subgraph
    #_ordered_inputs_per_subgraph[_subgraph_index] = [unique_id(inp) for inp in sample_inputs]

    strip_overloads(aten_module)

    global _device
    if _device is None:
        _device = get_default_device()
        assert _device is get_default_device()
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

    workload, compiled_graph_state, schedule = _compile_cached(
        module, aten_module, module_name, sample_inputs, device, compiler_cfg, cache
    )

    compiled_model = compiledModel(module, device, workload, compiled_graph_state, schedule, _subgraph_index-1)
    # Push parameters and constants to device
    compiled_model.to(device.torch_device())
    logger.info("Done Torch Compile")
    if original_torch_device is not None:
        module = module.to(original_torch_device)
    return compiled_model
    

# compile_torch = aot_autograd(fw_compiler=_torch_compile, decompositions={**core_aten_decompositions(), **pybuda_decompositions})
