# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from pybuda._C.graph import (
    Graph,
    create_op_node,
    create_data_edge,
    create_parameter_input,
    create_activation_input,
    create_output,
    create_constant_input,
    remove_node,
)
from .pybudaglobal import TILE_DIM
from pybuda.module import PyBudaModule
import pybuda
from pybuda.tensor import pytorch_dtype_to_buda_dataformat
from pybuda._C import DataFormat
from pybuda._C.graph import OpType
from pybuda.op.resize import RESIZE2d_METHOD_TO_INT
from tvm.contrib.pybuda_compile import load_tvm_graph

from collections import deque
from loguru import logger

import torch
import numpy as np

import sys
import json


def populate_binary_stack_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    input_shape = graph["nodes"][node["inputs"][0][0]]["buda_shape"]
    node_shape = node["buda_shape"]

    for dim, (i, o) in enumerate(zip(input_shape, node_shape)):
        if i != o:
            attrs.append(dim - len(input_shape))
            break

def populate_conv2d_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    kernel_size = [int(kernel) for kernel in node["attrs"]["kernel_size"][0]]
    assert all([k == kernel_size[0] for k in kernel_size])
    assert all([stride == strides[0] for stride in strides])
    attrs.append(strides[0])

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    attrs.append(dilation[0])

    groups = int(node["attrs"]["groups"][0][0])
    attrs.append(groups)

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    attrs.append(padding[1])
    attrs.append(padding[3])
    attrs.append(padding[0])
    attrs.append(padding[2])

def populate_maxpool2d_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    assert all([dim == kernel_size[0] for dim in kernel_size])
    attrs.append(kernel_size[0])

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert all([stride == strides[0] for stride in strides])
    attrs.append(strides[0])

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    attrs.append(dilation[0])

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0]) # 1 for True
    attrs.append(ceil_mode)

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    attrs.append(padding[1])
    attrs.append(padding[3])
    attrs.append(padding[0])
    attrs.append(padding[2])

def populate_clip_transpose_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    min = float(node["attrs"]["a_min"][0][0])
    max = float(node["attrs"]["a_max"][0][0])
    attrs.extend([min, max])


def populate_argmax_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]

    dim = int(node["attrs"]["axis"][0][0])
    attrs.append(dim)

def populate_avgpool2d_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]

    kernel_size = [int(pool_size) for pool_size in node["attrs"]["pool_size"][0]]
    assert all([dim == kernel_size[0] for dim in kernel_size])
    attrs.append(kernel_size[0])

    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert all([stride == strides[0] for stride in strides])
    attrs.append(strides[0])

    dilation = [int(dilate) for dilate in node["attrs"]["dilation"][0]]
    attrs.append(dilation[0])

    ceil_mode = int(node["attrs"]["ceil_mode"][0][0]) # 1 for True
    attrs.append(ceil_mode)

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    attrs.append(padding[1])
    attrs.append(padding[3])
    attrs.append(padding[0])
    attrs.append(padding[2])

def populate_vslice_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]
    output_shape = node["attrs"]["shape"][0][0]
    slice_size = output_shape[-3] // input_shape[-3]
    attrs.append(slice_size)
    

def populate_vstack_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]
    output_shape = node["attrs"]["shape"][0][0]
    
    slice_size = input_shape[-3] // output_shape[-3]
    attrs.append(slice_size)
    

def populate_hslice_attrs(graph, nid, attrs):
    attrs.append(graph["nodes"][nid]["buda_shape"][-3])


def populate_hstack_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]

    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    attrs.append(input_shape[-3])


def populate_reduce_avg_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    axis = int(node["attrs"]["axis"][0][0])
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]
    if axis > 0:
        axis -= len(input_shape)
    attrs.append(axis)

def populate_reduce_max_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    attrs.append(int(node["attrs"]["axis"][0][0]))

def populate_reduce_sum_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    axis = int(node["attrs"]["axis"][0][0])
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]
    if axis > 0:
        axis -= len(input_shape)
    attrs.append(axis)


def populate_index_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    strides = [int(strides) for strides in node["attrs"]["strides"][0]]
    begin = [int(begin) for begin in node["attrs"]["begin"][0]]
    end = [int(e) for e in node["attrs"]["end"][0]]

    assert len(strides) == 1 and len(begin) == 1 and len(end) == 1, "Stridedslice should be on 1 axis"
    assert int(node["attrs"]["num_inputs"]) == 1

    assert len(list(node["attrs"]["axes"][0])) == 1, "Select can only have 1 axis"
    dim = int(node["attrs"]["axes"][0][0])

    attrs.append(dim)
    attrs.append(begin[0])
    attrs.append(end[0])
    attrs.append(strides[0])


def populate_softmax_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    attrs.append(int(node["attrs"]["axis"][0][0]))


def populate_layernorm_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]

    attrs.append(int(node["attrs"]["axis"][0][0]))
    attrs.append(float(node["attrs"]["epsilon"][0][0]))


def populate_conv2d_transpose_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    strides = [int(stride) for stride in node["attrs"]["strides"][0]]
    assert all([stride == strides[0] for stride in strides])
    attrs.append(strides[0])

    dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
    assert all([dim == dilation[0] for dim in dilation])
    assert all([x == 1 for x in dilation]), "Only supports dilation of 1"
    attrs.append(dilation[0])

    assert int(node["attrs"]["groups"][0][0]) == 1, "Only supports group of 1"
    kernel_size = [int(kernel) for kernel in node["attrs"]["kernel_size"][0]]

    groups = int(node["attrs"]["groups"][0][0])
    attrs.append(groups)

    padding = [int(padding) for padding in node["attrs"]["padding"][0]]
    assert all([p == padding[0] for p in padding]), "Pybuda only supports same padding on all sides"
    # TVM has padding [top, left, bottom, right]
    # Convert to [left right top bottom]
    attrs.append(padding[1])
    attrs.append(padding[3])
    attrs.append(padding[0])
    attrs.append(padding[2])

def populate_pad_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    pad_width = [int(x) for x in node["attrs"]["pad_width"][0]]
    shape = node["attrs"]["shape"][0][0]

    if len(shape) > 2: 
        # Pybuda Pad only supports padding on last 2 dims
        assert len(pad_width) == len(shape) * 2
        assert all([x == 0 for x in pad_width[0:-4]]), "Pybuda Pad only supports padding on last 2 dims"
        pad_width = pad_width[-4:]

    # TVM nn.pad axis start from the last axis, need to swap 
    pad_width_by_axis = [pad_width[x : x + 2] for x in range(0, len(pad_width), 2)]
    pad_width_by_axis.reverse()
    pad_width_final = [item for axis in pad_width_by_axis for item in axis]
    attrs.extend(pad_width_final)


def populate_transpose_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]
    axes = [int(axis) for axis in node["attrs"]["axes"][0]]
    transpose_shape = list(graph["nodes"][nid]["buda_shape"])

    assert int(node["attrs"]["num_inputs"]) == 1

    for i, axis in enumerate(axes):
        if axis < 0:
            axes[i] += len(transpose_shape)

    node["attrs"]["axes"] = axes

    transpose_axes = []
    for idx, axis in enumerate(axes):
        if axis != idx:
            transpose_axes.insert(0, axis)

    assert (
        len(transpose_axes) == 2
    ), "only single axis transpose supported at this time, decompose in tvm"

    [attrs.append(axis - len(transpose_shape)) for axis in transpose_axes]

    # If transpose unpadded Z/W dim, record the original shape
    if (attrs[0] == -3 and attrs[1] != -4) or (attrs[0] == -4 and attrs[1] != -3):
        attrs.append(transpose_shape[attrs[0]])
    elif (attrs[1] == -3 and attrs[0] != -4) or (attrs[1] == -4 and attrs[0] != -3):
        attrs.append(transpose_shape[attrs[1]])
    else:
        attrs.append(-1)

    dim0, dim1, z_dim_slice = attrs
    return {
        "dim0": dim0,
        "dim1": dim1,
        "z_dim_slice": z_dim_slice,
    }
    

def populate_reshape_attrs(graph, nid, attrs):
    output_shape = graph["nodes"][nid]["buda_shape"]
    for x in output_shape:
        attrs.append(x)

def populate_concatenate_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]

    buda_shape = node["buda_shape"]
    concat_axis = int(node["attrs"]["axis"][0][0])
    if concat_axis >= 0:
        concat_axis -= len(buda_shape)
    attrs.append(concat_axis)

def populate_broadcast_attrs(graph, nid, attrs):

    node = graph["nodes"][nid]
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]
    output_shape = node["attrs"]["shape"][0][0]

    for i, (inp_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
        if inp_dim == out_dim:
            continue

        attrs.append(i)
        attrs.append(out_dim)
        attrs.append(True)
        input_shape[i] = out_dim
        assert input_shape == output_shape, "Pybuda broadcast only support 1 dim"


def populate_resize2d_attrs(graph, nid, attrs):
    node = graph["nodes"][nid]

    sizes = [int(x) for x in node["attrs"]["size"][0]]
    assert len(sizes) == 2
    method = node["attrs"]["method"][0][0]

    assert method == "nearest_neighbor" or method == "linear", "Only support nearest neighbor and linear for now"
    assert int(node["attrs"]["num_inputs"]) == 1
    input_nid = node["inputs"][0][0]
    input_shape = graph["nodes"][input_nid]["attrs"]["shape"][0][0]

    attrs.extend(sizes)
    attrs.append(RESIZE2d_METHOD_TO_INT[method])

    coordinate_transform_mode = node["attrs"]["coordinate_transformation_mode"][0][0]
    if coordinate_transform_mode == "align_corners":
        attrs.append(1)
    else:
        attrs.append(0)


def expand_compound_op(graph, nid, attrs, buda_graph, intermediate):
    class SubGraph(PyBudaModule):
        def __init__(self, fn_name, attributes):
            super().__init__("subgraph")
            self.fn_name = fn_name
            self.attributes = attributes

        def forward(self, *act):
            import pybuda.op
            import pybuda.op.nn

            if self.fn_name == "softmax":
                return pybuda.op.nn.Softmax("softmax", act[0], dim=self.attributes[0])

            if self.fn_name == "layernorm":
                return pybuda.op.nn.Layernorm(
                    "layernorm",
                    act[0],
                    act[1],
                    act[2],
                    dim=self.attributes[0],
                    epsilon=self.attributes[1],
                )

    node = graph["nodes"][nid]
    op_type = tvm_to_buda_op_map[node["name"]]

    subgraph = SubGraph(fn_name=op_type, attributes=attrs)

    input_nids = [
        node["inputs"][input_port][0]
        for input_port in range(int(node["attrs"]["num_inputs"]))
    ]
    inputs = tuple(
        [
            pybuda.Tensor.create_from_torch(graph["nodes"][input_nid]["tensor"])
            for input_nid in input_nids
        ]
    )

    output = subgraph.forward(*inputs)

    # TODO: Handle subgraph with multiple outputs
    assert isinstance(output, pybuda.Tensor)

    pending_tensors = deque()
    visited_tensors = {}
    module_io_tensor_to_node: Dict[str, Tensor] = {}

    pending_tensors.append((output, None, 0, output.src_op.operand_broadcast))

    def set_as_subgraph_output(op, tensor):
        logger.debug("Setting {} as output of subgraph", op)
        node["bid"] = op
        node["tensor"] = tensor.value()

    while pending_tensors:

        tensor, output, port_index, operand_broadcast = pending_tensors.popleft()

        if tensor in visited_tensors:
            if output is not None:
                create_data_edge(
                    buda_graph,
                    visited_tensors[tensor],
                    0,
                    output,
                    port_index,
                    operand_broadcast,
                )
                logger.debug(
                    f"Edge from: {visited_tensors[tensor]}:0 to: {output}:{port_index}"
                )
            else:
                set_as_subgraph_output(visited_tensors[tensor], tensor)
            continue

        if isinstance(tensor, pybuda.Parameter):
            # parameter tensor
            if tensor.get_name() is not None:
                name = tensor.get_name()
            else:
                name = "parameter_" + buda_graph.get_node_name(output)

            inq = create_parameter_input(
                graph, name, tensor.shape.get_pytorch_shape(), tensor.requires_grad
            )
            if output is not None:
                create_data_edge(
                    buda_graph, inq, 0, output, port_index, operand_broadcast
                )
                logger.debug("Edge from: {}:0 to: {}:{}", inq, output, port_index)
            else:
                set_as_subgraph_output(inq, tensor)
            visited_tensors[tensor] = inq
            continue

        if tensor.src_op is None:
            # input tensor from main graph
            input_node = graph["nodes"][input_nids[inputs.index(tensor)]]
            buda_id = input_node["bid"]

            logger.debug("Setting {} as input to subgraph", buda_id)
            create_data_edge(
                buda_graph, buda_id, 0, output, port_index, operand_broadcast
            )
            logger.debug("Edge from: {}:0 to: {}:{}", buda_id, output, port_index)
            visited_tensors[tensor] = buda_id
            continue

        elif tensor.src_op.op_type == "constant":
            constant_value = tensor.src_op.attrs[0]
            constant = create_constant_input(
                buda_graph,
                "constant_" + str(port_index) + "_" + buda_graph.get_node_name(output),
                constant_value,
                tensor.data_format,
            )

            if output is not None:
                create_data_edge(
                    buda_graph, constant, 0, output, port_index, operand_broadcast
                )
                logger.debug("Edge from: {}:0 to: {}:{}", constant, output, port_index)
            else:
                set_as_subgraph_output(constant, tensor)
            visited_tensors[tensor] = constant
            continue

        op = create_op_node(
            buda_graph,
            f"{tensor.src_op.name}_{nid}",
            tensor.src_op.cpp_op_type,
            tensor.shape.get_pytorch_shape(),
            tensor.data_format,
            {},
        )
        logger.debug(
            f"Node: {op} shape: {tensor.shape.get_pytorch_shape()} name: {tensor.src_op.name}_{nid}"
        )
        visited_tensors[tensor] = op
        intermediate[op] = tensor.value()

        if output is not None:
            create_data_edge(buda_graph, op, 0, output, port_index, operand_broadcast)
            logger.debug("Edge from: {}:0 to: {}:{}", op, output, port_index)
        else:
            set_as_subgraph_output(op, tensor)

        for i, t in enumerate(tensor.src_op.operands):
            pending_tensors.append((t, op, i, tensor.src_op.operand_broadcast))


# keep sorted
tvm_to_buda_op_map = {
    "abs"                           : "abs",
    "add"                           : "add",
    "argmax"                        : "argmax",
    "broadcast_to"                  : "broadcast",
    "pybuda.binary_stack"           : "binary_stack",
    "pybuda.buda_conv2d_with_bias"  : "conv2d",
    "pybuda.concatenate"            : "concatenate",
    "pybuda.hslice"                 : "hslice",
    "pybuda.hstack"                 : "hstack",
    "pybuda.matmul"                 : "matmul",
    "pybuda.vslice"                 : "vslice",
    "pybuda.vstack"                 : "vstack",
    "clip"                          : "clip",
    "cos"                           : "cos",
    "exp"                           : "exp",
    "gelu"                          : "gelu",
    "image.resize2d"                : "resize2d",
    "layernorm"                     : "layernorm",
    "log"                           : "log",
    "max"                           : "reduce_max",
    "mean"                          : "reduce_avg",
    "multiply"                      : "multiply",
    "nn.avg_pool2d"                 : "avg_pool2d",
    "nn.batch_matmul"               : "matmul",
    "nn.conv2d_transpose"           : "conv2d_transpose",
    "nn.conv2d"                     : "conv2d",
    "nn.matmul"                     : "matmul",
    "nn.max_pool2d"                 : "max_pool2d",
    "nn.pad"                        : "pad",
    "nn.relu"                       : "relu",
    "nn.softmax"                    : "softmax",
    "nop"                           : "nop",
    "power"                         : "power",
    "reciprocal"                    : "reciprocal",
    "reshape"                       : "reshape",
    "sigmoid"                       : "sigmoid",
    "sigmoid"                       : "sigmoid",
    "sin"                           : "sin",
    "sqrt"                          : "sqrt",
    "strided_slice"                 : "index",
    "subtract"                      : "subtract",
    "sum"                           : "reduce_sum",
    "transpose"                     : "transpose",
    "where"                         : "where",
}

compound_buda_ops = [
    "layernorm",
    "softmax",
]

ops_needing_attributes = {
    "argmax"            : populate_argmax_attrs,
    "avg_pool2d"        : populate_avgpool2d_attrs,
    "binary_stack"      : populate_binary_stack_attrs,
    "broadcast"         : populate_broadcast_attrs,
    "clip"              : populate_clip_transpose_attrs,
    "concatenate"       : populate_concatenate_attrs,
    "conv2d_transpose"  : populate_conv2d_transpose_attrs,
    "conv2d"            : populate_conv2d_attrs,
    "hslice"            : populate_hslice_attrs,
    "hstack"            : populate_hstack_attrs,
    "index"             : populate_index_attrs,
    "layernorm"         : populate_layernorm_attrs,
    "max_pool2d"        : populate_maxpool2d_attrs,
    "pad"               : populate_pad_attrs,
    "reduce_avg"        : populate_reduce_avg_attrs,
    "reduce_max"        : populate_reduce_max_attrs,
    "reduce_sum"        : populate_reduce_sum_attrs,
    "reshape"           : populate_reshape_attrs,
    "resize2d"          : populate_resize2d_attrs,
    "softmax"           : populate_softmax_attrs,
    "transpose"         : populate_transpose_attrs,
    "vslice"            : populate_vslice_attrs,
    "vstack"            : populate_vstack_attrs,
}


class ModuleWrapper(PyBudaModule):
    def __init__(self, torchmod, name):
        super().__init__(name=name)
        self.torchmod = torchmod

    def forward(self, *acts):
        return self.torchmod(*acts)


def str_to_dataformat(t: str) -> DataFormat:

    if t == "float32":
        return DataFormat.Float32

    if t == "float16":
        return DataFormat.Float16

    if t == "bfloat16":
        return DataFormat.Float16_b

    if t == "uint8":
        return DataFormat.Int8

    if t == "int8":
        return DataFormat.Int8

    if t == "int32":
        logger.warning("Op type is 'int32'. Setting to Int8 for now.")
        return DataFormat.Int8

    if t == "uint1":
        logger.warning("Op type is 'uint1'. Setting to Int8 for now.")
        return DataFormat.Int8

    raise RuntimeError("Unsupported format: " + t)

def compile_tvm_for_pybuda(buda_graph, torchmod, inputs, compiler_cfg, graph_name, verify_cfg=None):
    from pybuda.op.eval.pybuda import get_f_pybuda_shape  # avoid circular import
    from pybuda.op.eval.pybuda import get_f_pybuda_eval  # avoid circular import
    from pybuda._C.graph import OpType

    framework = pybuda.tvm_to_python.get_framework(module)
    module = torchmod.module
    json_graph, pytorch_inputs, weights = load_tvm_graph(inputs, module, compiler_cfg, graph_name, framework, verify_cfg=verify_cfg)

    buda_module = ModuleWrapper(module, torchmod.name + "_tvm")

    nid_to_bid = {}
    intermediate = {}
    graph = json.loads(json_graph["graph"])

    output_nodes = [head[0] for head in graph["heads"]]
    # reshape nops are added in tvm to passthrough nodes, prune them
    def is_nop_reshape(nid):
        node = graph["nodes"][nid]
        if node["name"] != "reshape":
            return False

        input_shape = graph["nodes"][node["inputs"][0][0]]["attrs"]["shape"]
        node_shape = node["attrs"]["shape"]
        return input_shape == node_shape

    for output_node in output_nodes:
        if is_nop_reshape(output_node):
            # we want to replace the reshape with a nop
            graph["nodes"][output_node]["name"] = "nop"

    input_nodes = graph["arg_nodes"]

    ordered_output_bids = [None] * len(output_nodes)

    def create_output_if_needed(nid, buda_graph, graph):
        node = graph["nodes"][nid]

        if nid in output_nodes:
            output_name = "output_" + node["name"] + "_" + str(nid)
            outq = create_output(
                buda_graph, output_name, node["buda_shape"],
                str_to_dataformat(node["attrs"]["dtype"][0][0]),
                False, # TODO: loss output?
            )
            logger.debug(
                f"Ouput: {outq} name: + {output_name}"
            )
            ordered_output_bids[output_nodes.index(nid)] = outq
            create_data_edge(buda_graph, node["bid"], 0, outq, 0, [])
    
    for nid, node in enumerate(graph["nodes"]):
        shape = node["attrs"]["shape"][0][0]
        node["buda_shape"] = tuple(shape)
        if node["op"] == "input":
            if node["name"] not in weights:
                if "nid_to_input_idx" in json_graph.keys() and len(json_graph["nid_to_input_idx"]) != 0:
                    inp_idx = json_graph["nid_to_input_idx"][nid]
                    requires_grad = pytorch_inputs[inp_idx].requires_grad
                else:
                    requires_grad = pytorch_inputs[input_nodes.index(nid)].requires_grad

                inq = create_activation_input(
                    buda_graph, "input_" + node["name"], node["buda_shape"], requires_grad,
                    str_to_dataformat(node["attrs"]["dtype"][0][0])
                )

                if "nid_to_input_idx" in json_graph.keys() and len(json_graph["nid_to_input_idx"]) != 0:
                    inp_idx = json_graph["nid_to_input_idx"][nid]
                    tensor = pytorch_inputs[inp_idx]
                else:
                    tensor = pytorch_inputs[input_nodes.index(nid)]

                node["bid"] = inq
                node["tensor"] = tensor
                logger.debug(
                    f"Node: {node['bid']} shape: {node['buda_shape']} name: {buda_graph.get_node_name(node['bid'])} type: input"
                )
            else:
                input_nodes.remove(nid)
                tensor, requires_grad = weights[node["name"]]
                tensor.requires_grad = requires_grad
                if len(shape) == 0:
                    constant_value = tensor.item()
                    inq = create_constant_input(
                            buda_graph,
                            "constant_" + node["name"],
                            constant_value,
                            str_to_dataformat(node["attrs"]["dtype"][0][0]))
                    node["tensor"] = tensor
                    logger.debug(
                        f"Node: {inq} shape: {node['buda_shape']} name: {buda_graph.get_node_name(inq)} type: constant"
                    )
                else:
                    param = pybuda.Parameter(
                        tensor,
                        requires_grad=tensor.requires_grad,
                        name=node["name"],
                    )
                    buda_module._parameters[node["name"]] = param
                    inq = create_parameter_input(
                        buda_graph,
                        node["name"],
                        param.shape.get_pytorch_shape(),
                        param.requires_grad,
                        pytorch_dtype_to_buda_dataformat(tensor.dtype)
                    )
                    node["tensor"] = tensor
                    logger.debug(
                        f"Node: {inq} shape: {node['buda_shape']} name: {buda_graph.get_node_name(inq)} type: parameter, requires_grad: {param.requires_grad}"
                    )
                node["bid"] = inq
        elif node["op"] == "const":
            input_nodes.remove(nid)
            if isinstance(json_graph["params"][node["name"]], np.ndarray):
                tensor = torch.from_numpy(json_graph["params"][node["name"]])
            else:
                tensor = torch.tensor(json_graph["params"][node["name"]])
            tensor.requires_grad = False
            node["tensor"] = tensor

            if len(shape) == 0:
                constant_value = tensor.item()
                inq = create_constant_input(
                        buda_graph,
                        "constant_" + node["name"],
                        constant_value,
                        str_to_dataformat(node["attrs"]["dtype"][0][0]))
            else:
                # if tvm is folding constants, then params are const type, and 'requires_grad' from framework is not preserved
                # so assume we do
                requires_grad = compiler_cfg.enable_tvm_constant_prop
                tensor.requires_grad = requires_grad
                param = pybuda.Parameter(
                    tensor,
                    requires_grad=requires_grad,
                    name=node["name"],
                )
                buda_module._parameters[node["name"]] = param
                inq = create_parameter_input(
                    buda_graph,
                    node["name"],
                    param.shape.get_pytorch_shape(),
                    param.requires_grad,
                    pytorch_dtype_to_buda_dataformat(tensor.dtype),
                )

            node["bid"] = inq
            logger.debug(
                f"Node: {node['bid']} shape: {node['buda_shape']} name: {buda_graph.get_node_name(node['bid'])} type: constant"
            )
        elif node["op"] == "kernel":

            op_type = tvm_to_buda_op_map[node["name"]]
            name = node["name"] + f"_{nid}"

            attrs = []
            named_attrs = None

            if op_type in ops_needing_attributes:
                named_attrs = ops_needing_attributes[op_type](graph=graph, nid=nid, attrs=attrs)

            if op_type in compound_buda_ops:
                expand_compound_op(
                    graph=graph,
                    nid=nid,
                    attrs=attrs,
                    buda_graph=buda_graph,
                    intermediate=intermediate,
                )
                create_output_if_needed(nid, buda_graph, graph)
                continue

            cpp_op_type = OpType(op_type, attrs) if named_attrs is None else OpType(op_type, named_attrs=named_attrs)
            op = create_op_node(buda_graph, name, cpp_op_type, node["buda_shape"], str_to_dataformat(node["attrs"]["dtype"][0][0]), {})
            node["bid"] = op

            # TVM nn.pad has 2 inputs [Data, pad_value]
            # We need to assert pad_value being zero, then remove the constant
            if node["name"] == "nn.pad" and int(node["attrs"]["num_inputs"]) == 2:
                pad_value_node = graph["nodes"][node["inputs"][1][0]]
                assert pad_value_node["tensor"].ndim == 0, "Pad value should be a single element"
                assert pad_value_node["tensor"].numpy().item() == 0, "Pybuda only support padding with 0"
                remove_node(buda_graph, pad_value_node["bid"])
                # Remove from json
                node["attrs"]["num_inputs"] = '1'


            logger.debug(
                f"Node: {node['bid']} shape: {node['buda_shape']} name: {buda_graph.get_node_name(node['bid'])} type: op"
            )

            assert "num_inputs" in node["attrs"]

            shapes = []
            forward_inputs = []
            for input_port in range(int(node["attrs"]["num_inputs"])):
                input_node = graph["nodes"][node["inputs"][input_port][0]]
                shapes.append(input_node["buda_shape"])
                forward_inputs.append(input_node["tensor"])

            shape, operand_broadcast = get_f_pybuda_shape(OpType(op_type, attrs))(shapes)
            node["tensor"] = get_f_pybuda_eval(OpType(op_type, attrs))(forward_inputs)
            if node["name"] != "nop":
                intermediate[op] = node["tensor"]

            for input_port in range(int(node["attrs"]["num_inputs"])):
                input_node = graph["nodes"][node["inputs"][input_port][0]]
                buda_id = input_node["bid"]

                create_data_edge(
                    buda_graph, buda_id, 0, op, input_port, operand_broadcast
                )
                logger.debug(
                    f"Edge from: {input_node['bid']}:0 to: {node['bid']}:{input_port}"
                )

        create_output_if_needed(nid, buda_graph, graph)

    assert None not in ordered_output_bids
    ordered_input_bids = [graph["nodes"][nid]["bid"] for nid in input_nodes]

    # TODO: figure out how to get requires_grad for output nodes?
    buda_graph.register_module_inputs(ordered_input_bids)
    buda_graph.register_module_outputs(ordered_output_bids, [True] * len(ordered_output_bids))

    buda_inputs = []
    for buda_input in input_nodes:
        buda_inputs.append(
            pybuda.Tensor.create_from_torch(graph["nodes"][buda_input]["tensor"])
        )

    buda_outputs = []
    for output in output_nodes:
        buda_outputs.append(
            pybuda.Tensor.create_from_torch(graph["nodes"][output]["tensor"])
        )

    return buda_graph, buda_module, buda_inputs, buda_outputs, intermediate
