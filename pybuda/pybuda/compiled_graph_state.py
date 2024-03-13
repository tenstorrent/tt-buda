# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import inspect
import os
import json

import importlib
from pybuda.ttdevice import TTDevice

from pybuda.compile import CompileResults

from pybuda._C import DataFormat
from pybuda._C.graph import Graph, get_constant_input_value, get_optimizer_param_info, RuntimeTensorTransform, RuntimeTensorTransformType, Shape
from pybuda._C.balancer import OutputHostTM

import dataclasses
from dataclasses_json import dataclass_json, config
from pybuda.utils import as_json, dict_as_json, list_as_json, detach_tensors
from pybuda.tensor import get_device_constant_and_parameters, get_post_const_eval_tensors 

import torch
def no_encoding(obj):
    return obj # perform json-encoding later
def no_decoding(obj):
    return obj # perform json-encoding later
def optional_no_encoding(obj):
    return None if obj is None else obj
def optional_no_decoding(obj):
    return None if obj is None else obj

@dataclass_json
@dataclass()
class CompiledGraphState:
    microbatch: int
    graph_name: str
    ordered_input_names: List[str]
    ordered_output_names: List[str]
    ordered_input_gradient_names: List[str]
    ordered_output_gradient_names: List[str]
    ordered_target_names: List[str]
    ordered_constant_node_names: List[str]
    ordered_parameter_node_names: List[str]
    ordered_intermediate_activation_names: List[Tuple[str,str]]
    ordered_input_subgraph_indices: List[int]
    ordered_output_subgraph_indices: List[int]
    ordered_target_subgraph_indices: List[int]

    ordered_input_tile_broadcast_dims: List[List[int]]
    ordered_target_tile_broadcast_dims: List[List[int]]
    ordered_bw_input_tile_broadcast_dims: List[List[int]]

    ordered_input_runtime_tensor_transforms: List[RuntimeTensorTransform] = field(
        metadata=list_as_json(RuntimeTensorTransform)
    )
    ordered_output_runtime_tensor_transforms: List[RuntimeTensorTransform] = field(
        metadata=list_as_json(RuntimeTensorTransform)
    )

    input_to_tile_dims: Dict[str, Tuple[int, int]]
    parameter_to_tile_dims: Dict[str, Tuple[int, int]]
    constant_to_tile_dims: Dict[str, Tuple[int, int]]

    # attributes derived based on initial graph
    ordered_input_requires_grad: List[bool]
    ordered_output_requires_grad: List[bool]
    ordered_input_shapes: List[List[int]]
    ordered_output_shapes: List[List[int]]
    ordered_target_shapes: List[List[int]]
    ordered_intermediate_shapes: List[List[int]]
    ordered_output_data_formats: List[DataFormat] = field(metadata=list_as_json(DataFormat))

    netlist_filename: str
    output_host_tms: Dict[str, OutputHostTM] = field(metadata=dict_as_json(OutputHostTM))
    consteval_trace: Dict[str, Dict[str, Any]]
    post_const_eval_constants: Dict[str, torch.Tensor] = field(
        metadata=config( # For serialization of CompiledGraphState cls
            encoder=no_encoding, 
            decoder=no_decoding
        )
    )
    post_const_eval_parameters: Dict[str, torch.Tensor] = field(
        metadata=config( # For serialization of CompiledGraphState cls
            encoder=no_encoding, 
            decoder=no_decoding
        )
    )
    optimizer_param_info: Dict[str, List[Tuple[str, str]]]

    # Hold cpu-evaluated outputs loaded from TTI
    cpueval_outputs: Optional[List[torch.Tensor]] = field(
        metadata=config(
            encoder=optional_no_encoding,
            decoder=optional_no_decoding
        ),
        default=None
    )

    has_cache_buffers: bool = False

    @staticmethod
    def from_compiled_graph(device: "TTDevice", compile_results: CompileResults) -> "CompiledGraphState":
        graph = compile_results.lowered_graph
        ordered_input_names = graph.get_ordered_input_names()
        ordered_output_names = graph.get_ordered_output_names()
        ordered_input_gradient_names = graph.get_ordered_input_gradient_names()
        ordered_output_gradient_names = graph.get_ordered_output_gradient_names()
        ordered_target_names = graph.get_ordered_target_names()
        ordered_input_subgraph_indices = graph.get_ordered_input_subgraph_indices()
        ordered_output_subgraph_indices = graph.get_ordered_output_subgraph_indices()
        ordered_target_subgraph_indices = graph.get_ordered_target_subgraph_indices()
        ordered_constant_node_names=[constant_node.name for constant_node in graph.get_constant_nodes()]
        ordered_parameter_node_names=[parameter_node.name for parameter_node in graph.get_parameter_nodes()]
        ordered_intermediate_activation_names = [(intermediate.rstrip("_intermediate_output"), intermediate) for intermediate in graph.get_ordered_intermediate_names()]

        ordered_input_tile_broadcast_dims = [graph.get_tile_broadcast_dims_for_input(i) for i in range(len(ordered_input_names))]
        ordered_target_tile_broadcast_dims = [graph.get_tile_broadcast_dims_for_target(i) for i in range(len(ordered_target_names))]
        ordered_bw_input_tile_broadcast_dims = [graph.get_tile_broadcast_dims_for_bw_input(i) for i in range(len(ordered_output_gradient_names))]

        # Tile dims
        ordered_input_tile_dims = graph.get_ordered_input_tile_dims()
        ordered_parameter_tile_dims = graph.get_ordered_parameter_tile_dims()
        ordered_constant_tile_dims = graph.get_ordered_constant_tile_dims()
        input_to_tile_dims = {}
        parameter_to_tile_dims = {}
        constant_to_tile_dims = {}
        for name, tile_dim in zip(ordered_input_names, ordered_input_tile_dims):
            input_to_tile_dims[name] = tile_dim

        for name, tile_dim in zip(ordered_parameter_node_names, ordered_parameter_tile_dims):
            parameter_to_tile_dims[name] = tile_dim

        for name, tile_dim in zip(ordered_constant_node_names, ordered_constant_tile_dims):
            constant_to_tile_dims[name] = tile_dim

        # Transforms
        ordered_input_runtime_tensor_transforms = graph.get_input_runtime_tensor_transforms()
        ordered_output_runtime_tensor_transforms = graph.get_output_runtime_tensor_transforms()
        assert len(ordered_input_runtime_tensor_transforms) == len(ordered_input_names)
        assert len(ordered_output_runtime_tensor_transforms) == len(ordered_output_names)

        ordered_input_requires_grad = compile_results.initial_graph.get_ordered_input_requires_grad()
        ordered_output_requires_grad = compile_results.initial_graph.get_ordered_output_requires_grad()
        ordered_input_shapes = compile_results.initial_graph.get_ordered_input_shapes()
        if graph.output_node_redirected():
            ordered_output_shapes = graph.get_ordered_output_shapes()
        else:
            ordered_output_shapes = compile_results.initial_graph.get_ordered_output_shapes()
        ordered_target_shapes = compile_results.initial_graph.get_ordered_target_shapes()
        ordered_intermediate_shapes = graph.get_ordered_intermediate_shapes()

        # Fetching this off the output tensors, but we could also just fetch from graph
        ordered_output_data_formats = [output_tensor.data_format for output_tensor in compile_results.outputs]

        constant_to_tensor = {}
        for name, tensor in graph.get_constant_input_runtime_tensor_transform_constants():
            constant_to_tensor[name] = tensor

        optimizer_param_info = {}
        for param_name, opt_params in device.get_optimizer_params(is_buda=True).items():
            optimizer_param_info[param_name] = []
            for input_node, param_key in get_optimizer_param_info(graph, param_name):
                optimizer_param_info[param_name].append((input_node.name, param_key))

        consteval_trace = compile_results.pass_specific_output_kwargs["consteval_trace"]
        has_cache_buffers = False
        for _, placement in compile_results.pass_specific_output_kwargs["placer_solution"].name_to_queue_placement.items():
            if placement.write_only:
                has_cache_buffers = True
        device_inputs = get_device_constant_and_parameters(
            device, constant_to_tensor=constant_to_tensor
        )
        post_const_eval_constants: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph,
            device_inputs,
            consteval_trace,
            constant_to_tile_dims,
            ordered_constant_node_names
        )
        post_const_eval_parameters: Dict[str, torch.Tensor] = get_post_const_eval_tensors(
            graph,
            device_inputs,
            consteval_trace,
            parameter_to_tile_dims,
            ordered_parameter_node_names
        )

        if os.getenv("PYBUDA_N300_DATA_PARALLEL", 0):
            def replicate_items(items_to_replicate):
                dev0 = { f"{k}.0" : v for k,v in items_to_replicate.items() }
                dev1 = { f"{k}.1" : v for k,v in items_to_replicate.items() }
                return {**dev0, **dev1}

            input_to_tile_dims = replicate_items(input_to_tile_dims)

            post_const_eval_constants = replicate_items(post_const_eval_constants)
            post_const_eval_parameters = replicate_items(post_const_eval_parameters)

            ordered_constant_node_names=[f"{name}.0" for name in ordered_constant_node_names] + [f"{name}.1" for name in ordered_constant_node_names]
            ordered_parameter_node_names=[f"{name}.0" for name in ordered_parameter_node_names] + [f"{name}.1" for name in ordered_parameter_node_names]

            ordered_input_names=[f"{name}.0" for name in ordered_input_names] + [f"{name}.1" for name in ordered_input_names]
            ordered_output_names=[f"{name}.0" for name in ordered_output_names] + [f"{name}.1" for name in ordered_output_names]
            ordered_input_shapes=ordered_input_shapes + ordered_input_shapes
            ordered_output_shapes=ordered_output_shapes + ordered_output_shapes
            ordered_input_requires_grad = ordered_input_requires_grad + ordered_input_requires_grad
            ordered_output_requires_grad = ordered_output_requires_grad + ordered_output_requires_grad
            ordered_input_runtime_tensor_transforms = ordered_input_runtime_tensor_transforms + ordered_input_runtime_tensor_transforms
            ordered_output_runtime_tensor_transforms = ordered_output_runtime_tensor_transforms + ordered_output_runtime_tensor_transforms
            ordered_input_tile_broadcast_dims = ordered_input_tile_broadcast_dims + ordered_input_tile_broadcast_dims
            ordered_target_tile_broadcast_dims = ordered_target_tile_broadcast_dims + ordered_target_tile_broadcast_dims
            ordered_bw_input_tile_broadcast_dims = ordered_bw_input_tile_broadcast_dims + ordered_bw_input_tile_broadcast_dims

            print(f"ordered_output_names = {ordered_output_names}")
            print(f"ordered_output_shapes = {ordered_output_shapes}") #TODO: probably double here

        return CompiledGraphState(
            microbatch=graph.get_microbatch(),
            graph_name=graph.get_name(),
            ordered_input_names=ordered_input_names,
            ordered_output_names=ordered_output_names,
            ordered_input_gradient_names=ordered_input_gradient_names,
            ordered_output_gradient_names=ordered_output_gradient_names,
            ordered_target_names=ordered_target_names,
            ordered_constant_node_names=ordered_constant_node_names,
            ordered_parameter_node_names=ordered_parameter_node_names,
            ordered_intermediate_activation_names=ordered_intermediate_activation_names,
            ordered_input_tile_broadcast_dims=ordered_input_tile_broadcast_dims,
            ordered_target_tile_broadcast_dims=ordered_target_tile_broadcast_dims,
            ordered_bw_input_tile_broadcast_dims=ordered_bw_input_tile_broadcast_dims,
            ordered_input_runtime_tensor_transforms=ordered_input_runtime_tensor_transforms,
            ordered_output_runtime_tensor_transforms=ordered_output_runtime_tensor_transforms,
            ordered_input_requires_grad=ordered_input_requires_grad,
            ordered_output_requires_grad=ordered_output_requires_grad,
            ordered_input_shapes=ordered_input_shapes,
            ordered_output_shapes=ordered_output_shapes,
            ordered_target_shapes=ordered_target_shapes,
            ordered_intermediate_shapes=ordered_intermediate_shapes,
            ordered_output_data_formats=ordered_output_data_formats,
            netlist_filename=compile_results.netlist_filename,
            output_host_tms=compile_results.pass_specific_output_kwargs["output_host_tms"],
            consteval_trace=consteval_trace,
            optimizer_param_info=optimizer_param_info,
            ordered_input_subgraph_indices=ordered_input_subgraph_indices,
            ordered_output_subgraph_indices=ordered_output_subgraph_indices,
            ordered_target_subgraph_indices=ordered_target_subgraph_indices,
            input_to_tile_dims=input_to_tile_dims,
            parameter_to_tile_dims=parameter_to_tile_dims,
            constant_to_tile_dims=constant_to_tile_dims,
            post_const_eval_constants=post_const_eval_constants,
            post_const_eval_parameters=post_const_eval_parameters,
            has_cache_buffers=has_cache_buffers,
        )

    def get_tensor(self, name_to_tensor, name):
        assert name in name_to_tensor
        value = name_to_tensor[name]

        # If mapped value is callable, we call it to get the tensor.
        # This is useful for the case where we want to lazily evaluate
        if callable(value):
            tensor = value()
            name_to_tensor[name] = tensor
        else:
            tensor = value
        return tensor

    def get_constant_tensor(self, name):
        return self.get_tensor(self.post_const_eval_constants, name)

    def get_parameter_tensor(self, name):
        return self.get_tensor(self.post_const_eval_parameters, name)

    def get_ordered_input_names_for_subgraph(self, subgraph_idx):
        return [name for i, name in enumerate(self.ordered_input_names) if self.ordered_input_subgraph_indices[i] == subgraph_idx]

    def get_ordered_input_shapes_for_subgraph(self, subgraph_idx):
        return [shape for i, shape in enumerate(self.ordered_input_shapes) if self.ordered_input_subgraph_indices[i] == subgraph_idx]

    def get_ordered_input_runtime_transforms_for_subgraph(self, subgraph_idx):
        return [transform for i, transform in enumerate(self.ordered_input_runtime_tensor_transforms) if self.ordered_input_subgraph_indices[i] == subgraph_idx]

    def get_ordered_input_tile_broadcast_dims_for_subgraph(self, subgraph_idx):
        return [tile_dims for i, tile_dims in enumerate(self.ordered_input_tile_broadcast_dims) if self.ordered_input_subgraph_indices[i] == subgraph_idx]

    def get_ordered_output_names_for_subgraph(self, subgraph_idx):
        return [name for i, name in enumerate(self.ordered_output_names) if self.ordered_output_subgraph_indices[i] == subgraph_idx]

    def get_ordered_output_shapes_for_subgraph(self, subgraph_idx):
        return [shape for i, shape in enumerate(self.ordered_output_shapes) if self.ordered_output_subgraph_indices[i] == subgraph_idx]

    def get_ordered_output_runtime_transforms_for_subgraph(self, subgraph_idx):
        return [transform for i, transform in enumerate(self.ordered_output_runtime_tensor_transforms) if self.ordered_output_subgraph_indices[i] == subgraph_idx]
