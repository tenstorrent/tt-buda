# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# 
# Functions that convert FX nodes to PyBuda
#

import sys
import math
from typing import List, Set, Tuple

import torch
from loguru import logger

from pybuda._C.graph import OpType
from pybuda.tensor import pytorch_dtype_to_buda_dataformat
from pybuda.config import CompilerConfig, _get_global_compiler_config

class PyBudaNode:
    def __init__(self, op: OpType, args: List[torch.fx.node.Node]):
        self.op = op
        self.args = args
        self.shape = None
        self.dtype = None
        self.wrap_tuple = False

def process_dummy_no_attr(node, pybuda_op_name):
    return PyBudaNode(OpType(pybuda_op_name, []), node.args)

def process_dummy_attr_in_args(node, pybuda_op_name):
    attrs = node.args[1] if len(node.args) == 2 else node.args[1:]
    if not isinstance(attrs, (list, tuple)):
        attrs = [attrs, ]
    return PyBudaNode(OpType(pybuda_op_name, attrs), [node.args[0], ])

def process_expand(node, pybuda_op_name):
    return PyBudaNode(OpType(pybuda_op_name, []), [node.args[0], ])

def process_clamp(node, pybuda_op_name):
    assert len(node.args) == 3
    inputs = [node.args[0],]
    min_ = node.args[1]
    max_ = node.args[2]

    if min_ is None:
        assert max_ is not None, "Both min and max attributes for clmap are empty"
        return PyBudaNode(OpType("relu", [max_, "max"]), inputs)
    elif max_ is None:
        assert min_ is not None, "Both min and max attributes for clmap are empty"
        return PyBudaNode(OpType("relu", [min_, "min"]), inputs)
    else:
        return PyBudaNode(OpType(pybuda_op_name, named_attrs = {"min": min_, "max": max_}), inputs)

def process_flatten(node, pybuda_op_name):
    return PyBudaNode(OpType(pybuda_op_name, [-1, ]), [node.args[0], ])

def process_gelu(node, pybuda_op_name):
    return PyBudaNode(OpType(pybuda_op_name, ["none", ]), node.args)

def process_getitem(node, pybuda_op_name):
    num_dims = sum([(isinstance(dim, slice) and (dim.start is not None or dim.stop is not None)) or (not isinstance(dim, slice) and dim is not None) for dim in node.args[1]])
    if num_dims == 0:
        return PyBudaNode(OpType("nop", []), [node.args[0], ])
    assert num_dims <= 1, "TODO: Support multi axis getitem"
    for dim, slice_index in enumerate(node.args[1]):
        if isinstance(slice_index, slice) and slice_index.start is None and slice_index.stop is None:
            continue
        if isinstance(slice_index, int):
            start = slice_index
            stop = None
            stride = 1
        else:
            start = slice_index.start
            stop = slice_index.stop
            if slice_index.step is not None:
                stride = slice_index.step
            else:
                stride = 1

    if stop is None:
        stop = start + 1
    if stop < 0:
        stop += node.args[0].meta['tensor_meta'].shape[dim]
    
    return PyBudaNode(OpType(pybuda_op_name, [dim, start, stop, stride]), [node.args[0], ])

def process_interpolate(node, pybuda_op_name):
    assert all([arg in node.kwargs for arg in ["size", "mode", "align_corners"]])

    output_size = node.kwargs["size"]
    align_corners = int(node.kwargs["align_corners"])
    mode_str = node.kwargs["mode"]
    if mode_str == "bilinear":
        mode = 1
    elif mode_str == "nearest":
        mode = 0
    else:
        assert False, f"Unsupported interpolate mode: {mode_str}"

    attrs = [output_size, output_size, mode, align_corners, 0] # channel-last is false for pt
    return PyBudaNode(OpType(pybuda_op_name, attrs), [node.args[0], ])

def process_transpose(node, pybuda_op_name):
    torch_op_name = node.target.__name__
    if torch_op_name == "permute":
        dim0 = None
        dim1 = None
        for i, arg in enumerate(node.args[1]):
            if arg != i:
                if dim0 is None:
                    dim0 = i
                elif dim1 is None:
                    dim1 = i
                else:
                    assert False, "Multi axis permute needs to be added to pybuda"

    elif torch_op_name == "transpose":
        dim0 = node.args[1]
        dim1 = node.args[2]
    
    dims = len(node.args[0].meta['tensor_meta'].shape)
    if dim0 > 0:
        dim0 -= dims
    if dim1 > 0:
        dim1 -= dims
    if dim0 > dim1:
        dim0, dim1 = dim1, dim0

    named_attrs = {"dim0": dim0, "dim1": dim1, "z_dim_slice": -1}

    return PyBudaNode(OpType(pybuda_op_name, named_attrs=named_attrs), [node.args[0], ])

def process_softmax(node, pybuda_op_name):
    if len(node.args) == 1:
        assert "dim" in node.kwargs, "dim must be specified"
        dim = node.kwargs["dim"]
    else:
        dim = node.args[1]
    
    if dim >= 0:
        dim -= len(node.args[0].meta['tensor_meta'].shape)
    stable = 1
    attrs = [dim, stable]
    return PyBudaNode(OpType(pybuda_op_name, attrs), [node.args[0], ])

def process_conv2d(node, pybuda_op_name):
    assert len(node.args) == 9

    inputs = [node.args[0], node.args[1]]
    if node.args[2]: # bias
        inputs.append(node.args[2])

    strides = node.args[3]
    if isinstance(node.args[4], list):
        if len(node.args[4]) == 2:
            padding = [node.args[4][1], node.args[4][1], node.args[4][0], node.args[4][0]]
        else:
            padding = node.args[4]
    else:
        padding = [node.args[4]] * 4 
    dilation = node.args[5]
    group = node.args[8]
    assert all([d == dilation[0] for d in dilation]), "Dilation is not same for all-dim, not supported"
    attrs = strides + [dilation[0], group] + padding + [False, 0, 0, 0, False] # channel-last = false for pt 

    return PyBudaNode(OpType(pybuda_op_name, attrs), inputs)

def process_maxpool2d(node, pybuda_op_name):
    assert len(node.args) >= 2 and len(node.args) <= 7, f"Maxpool-2d supposed to have 2~7 args: #args = {len(node.args)}" 
    inputs = [node.args[0],] 
    kernel_size = node.args[1]
    strides = node.args[1]
    padding = [0] * 4
    dilation = 1
    ceil_mode = False

    if len(node.args) >= 3:
        strides = node.args[2]

    if len(node.args) >= 4:
        if isinstance(node.args[3], list):
            if len(node.args[3]) == 2:
                padding = [node.args[3][1], node.args[3][1], node.args[3][0], node.args[3][0]]
            else:
                padding = node.args[3]
        else:
            padding = [node.args[3]] * 4

    if len(node.args) >= 5:
        dilation = node.args[4]

    if len(node.args) >= 6:
        ceil_mode = node.args[5]

    compiler_cfg = _get_global_compiler_config()
    add_sub_surround = compiler_cfg.max_pool_add_sub_surround
    add_sub_surround_value = compiler_cfg.max_pool_add_sub_surround_value
    attrs = kernel_size + strides + [dilation, ceil_mode] + padding + [add_sub_surround, add_sub_surround_value, False] # channel-last = False for pt

    pybuda_node = PyBudaNode(OpType(pybuda_op_name, attrs), inputs)
    pybuda_node.shape = node.meta['tensor_meta'][0].shape
    pybuda_node.dtype = pytorch_dtype_to_buda_dataformat(node.meta['tensor_meta'][0].dtype)
    pybuda_node.wrap_tuple = True
    return pybuda_node

def process_matmul(node, pybuda_op_name):
    assert len(node.args) == 2 or len(node.args) == 3
    if len(node.args) == 3:
        # Torch addmm inputs are bias, LHS, RHS
        args = [node.args[1], node.args[2], node.args[0]]
    else:
        args = node.args
    
    return PyBudaNode(OpType(pybuda_op_name, []), args)

def process_embedding(node, pybuda_op_name):
    assert len(node.args) == 2 or len(node.args) == 3

    #TODO Handle padding index (arg 2)
    args = [node.args[0], node.args[1]]
    return PyBudaNode(OpType(pybuda_op_name, []), args)

def process_mean(node, pybuda_op_name):
    assert len(node.args) >= 2
    dim = node.args[1]
    attrs = [dim,]
    args = [node.args[0],]
    return PyBudaNode(OpType(pybuda_op_name, attrs), args)

def process_layernorm(node, pybuda_op_name):
    assert len(node.args) == 5
    dim = -1
    epsilon = node.args[4]
    attrs = [dim, epsilon]

    args = [node.args[0], node.args[2], node.args[3]]
    pybuda_node = PyBudaNode(OpType(pybuda_op_name, attrs), args)
    pybuda_node.shape = node.meta['tensor_meta'][0].shape
    pybuda_node.dtype = pytorch_dtype_to_buda_dataformat(node.meta['tensor_meta'][0].dtype)
    pybuda_node.wrap_tuple = True
    return pybuda_node

def process_batchnorm(node, pybuda_op_name):
    assert len(node.args) == 7
    epsilon = node.args[-1]
    attrs = [epsilon]
    args = [node.args[0], node.args[1], node.args[2], node.args[3], node.args[4]] 
    pybuda_node = PyBudaNode(OpType(pybuda_op_name, attrs), args)

    pybuda_node.shape = node.meta['tensor_meta'][0].shape
    pybuda_node.dtype = pytorch_dtype_to_buda_dataformat(node.meta['tensor_meta'][0].dtype)
    pybuda_node.wrap_tuple = True
    return pybuda_node

def process_select(node, pybuda_op_name):
    assert len(node.args) == 3

    dim = node.args[1]
    if dim >= 0:
        dim -= len(node.args[0].meta['tensor_meta'].shape)
    index = node.args[2]
    attrs = [dim, index, index+1, 1]
    args = [node.args[0], ]
    return PyBudaNode(OpType(pybuda_op_name, attrs), args)

def process_slice(node, pybuda_op_name):
    assert len(node.args) == 4

    dim = node.args[1]
    start = node.args[2]
    end = node.args[3]
    if dim >= 0:
        dim -= len(node.args[0].meta['tensor_meta'].shape)
    if start == 0 and end == sys.maxsize:
        pybuda_node = PyBudaNode(OpType("nop", []), [node.args[0], ])
    else:
        stride = 1
        attrs = [dim, start, end, stride]
        args = [node.args[0], ]
        pybuda_node = PyBudaNode(OpType(pybuda_op_name, attrs), args)
    return pybuda_node

def process_unsqueeze(node, pybuda_op_name):
    assert len(node.args) == 2
    dim = node.args[1]
    input_ndim = len(node.meta['tensor_meta'].shape) - 1 # supopsed to feed input ndim

    if dim >= 0:
        dim -= len(node.meta['tensor_meta'].shape)
    
    attrs = [dim, input_ndim]
    return PyBudaNode(OpType(pybuda_op_name, attrs), [node.args[0], ])

def process_reshape(node, pybuda_op_name):
    attrs = node.args[1].copy() if len(node.args) == 2 else node.args[1:].copy()
    if not isinstance(attrs, (list, tuple)):
        attrs = [attrs, ]

    input_volume = 1
    for dim in node.args[0].meta['tensor_meta'].shape:
        input_volume *= dim

    blank_index = None
    reshape_volume = 1
    for i, dim in enumerate(attrs):
        if dim == -1:
            assert blank_index is None, "Only one dimension can be -1"
            blank_index = i
        else:
            reshape_volume *= dim
    
    if blank_index is not None:
        attrs[blank_index] = input_volume//reshape_volume

    input_volume = node.args[0].meta['tensor_meta'].shape[0]
    return PyBudaNode(OpType(pybuda_op_name, attrs), [node.args[0], ])

def process_power(node, pybuda_op_name):
    if isinstance(node.args[1], int) or isinstance(node.args[1], float) and math.isclose(node.args[1] / int(node.args[1]), 1.0):
        attrs = [int(node.args[1]), ]
        pybuda_node = PyBudaNode(OpType("pow", attrs), [node.args[0], ])
    else:
        pybuda_node = PyBudaNode(OpType("power", []), node.args)
    return pybuda_node

def process_cat(node, pybuda_op_name):
    dim = node.args[1]
    if dim >= 0:
        dim -= len(node.meta['tensor_meta'].shape)
    pybuda_node = PyBudaNode(OpType(pybuda_op_name, [dim, ]), node.args[0])
    return pybuda_node

def process_constant_pad_nd(node, pybuda_op_name):
    padding = node.args[1]
    value = node.args[2]
    if value != 0.0:
        raise ValueError("Buda only supports zero padding") # TODO: add to cpu fallback if padding is not 0
    pybuda_node = PyBudaNode(OpType(pybuda_op_name, [*padding, 0, False]), [node.args[0], ]) # mode index 0 = constant
    return pybuda_node

dynamo_to_pybuda_function = {
    "_softmax"                             : (process_softmax, "softmax"),
    "add"                                  : (process_dummy_no_attr, "add"),
    "add_"                                 : (process_dummy_no_attr, "add"),
    "addmm"                                : (process_matmul, "matmul"),
    "_native_batch_norm_legit_no_training" : (process_batchnorm, "batchnorm"), 
    "bmm"                                  : (process_matmul, "matmul"),
    "cat"                                  : (process_cat, "concatenate"),
    "clamp"                                : (process_clamp, "clip"),
    "clone"                                : (process_dummy_no_attr, "nop"),
    "contiguous"                           : (process_dummy_no_attr, "nop"),
    "constant_pad_nd"                      : (process_constant_pad_nd, "pad"),
    "convolution"                          : (process_conv2d, "conv2d"), #TODO: check if conv3d is also mapped to 'convolution'
    "div"                                  : (process_matmul, "divide"),
    "embedding"                            : (process_embedding, "embedding"),
    "eq"                                   : (process_dummy_no_attr, "equal"),
    "expand"                               : (process_expand, "nop"),
    "flatten"                              : (process_flatten, "reshape"),
    "gelu"                                 : (process_gelu, "gelu"),
    "getitem"                              : (process_getitem, "index"),
    "gt"                                   : (process_dummy_no_attr, "greater"),
    "gte"                                  : (process_dummy_no_attr, "greater_equal"),
    "hardtanh"                             : (process_clamp, "clip"),
    "iadd"                                 : (process_dummy_no_attr, "add"),
    "interpolate"                          : (process_interpolate, "resize2d"),
    "lt"                                   : (process_dummy_no_attr, "less"),
    "lte"                                  : (process_dummy_no_attr, "less_equal"),
    "matmul"                               : (process_dummy_no_attr, "matmul"),
    "max_pool2d_with_indices"              : (process_maxpool2d, "max_pool2d"),
    "mean"                                 : (process_mean, "reduce_avg"),
    "mm"                                   : (process_matmul, "matmul"),
    "mul"                                  : (process_dummy_no_attr, "multiply"),
    "native_layer_norm"                    : (process_layernorm, "layernorm"),
    "permute"                              : (process_transpose, "transpose"),
    "relu"                                 : (process_dummy_no_attr, "relu"),
    "relu_"                                : (process_dummy_no_attr, "relu"),
    "select"                               : (process_select, "index"),
    "sigmoid"                              : (process_dummy_no_attr, "sigmoid"),
    "slice"                                : (process_slice, "index"),
    "softmax"                              : (process_softmax, "softmax"),
    "sub"                                  : (process_dummy_no_attr, "subtract"),
    "tanh"                                 : (process_dummy_no_attr, "tanh"),
    "to"                                   : (process_dummy_no_attr, "nop"), #TODO
    "_to_copy"                             : (process_dummy_no_attr, "nop"), #TODO
    "copy_"                                : (process_dummy_no_attr, "nop"), #TODO
    "lift_fresh_copy"                      : (process_dummy_no_attr, "nop"), #TODO
    "alias"                                : (process_dummy_no_attr, "nop"), #TODO
    "transpose"                            : (process_transpose, "transpose"),
    "truediv"                              : (process_dummy_no_attr, "divide"),
    "unsqueeze"                            : (process_unsqueeze, "unsqueeze"),
    "view"                                 : (process_reshape, "reshape"),
    "_unsafe_view"                         : (process_reshape, "reshape"),
    "where"                                : (process_dummy_no_attr, "where"),
    "pow"                                  : (process_power, ""),
}

torch_constant_ops = {
    "ones"                           : torch.ones,
    "zeros"                          : torch.zeros,
    "arange"                         : torch.arange,
    "full"                           : torch.full,
    "empty"                          : torch.empty,
    "scalar_tensor"                  : torch.scalar_tensor,
}


def is_supported_op(torch_op_name, node: torch.fx.Node):
    if torch_op_name not in dynamo_to_pybuda_function:
        return False

    # Check for special cases
    if torch_op_name == "cat":
        if len(node.args) == 1:
            return False # We currently need explicit dim specificed in second arg

    return True


def get_pybuda_node(torch_op_name, node):
    if not is_supported_op(torch_op_name, node):
        print(f"Unsupported op {torch_op_name}")
        breakpoint()
        assert False, f"Unsupported op {torch_op_name}"
    
    return dynamo_to_pybuda_function[torch_op_name][0](node, dynamo_to_pybuda_function[torch_op_name][1])

# Check to see if subgraph is already on device
def is_on_device(subgraph_idx: int):
    pass

# Remove all nodes associated with subgraph
def remove_subgraph(subgraph_idx: int):
    pass

def add_op(graph, node, name, pybuda_node, subgraph_idx):
    global node_to_id
    shape = node.meta['tensor_meta'].shape if pybuda_node.shape is None else pybuda_node.shape
    dtype = pytorch_dtype_to_buda_dataformat(node.meta['tensor_meta'].dtype) if pybuda_node.dtype is None else pybuda_node.dtype

    add_constants_if_necessary(graph, pybuda_node.args, subgraph_idx)
    if "nn_module_stack" in node.meta:
        tags = {
            "layer": list(node.meta["nn_module_stack"].values())[-1][0],
            "stack_trace": "-->".join([str(v) for v in node.meta["nn_module_stack"].values()])
        }
    else:
        tags = {}
    if len(shape) == 0:
        shape = [1]
    nid = create_op_node(
            graph,
            f"{name}_{subgraph_idx}",
            pybuda_node.op,
            [int(dim) for dim in shape],
            pytorch_dtype_to_buda_dataformat(dtype),
            subgraph_idx,
            tags)
    
    for i, input_node in enumerate(pybuda_node.args):
        create_data_edge(graph, node_to_id[input_node], 0, nid, i, [])

    eval_args = [id_to_intermed[node_to_id[arg]] if isinstance(arg, torch.fx.node.Node) else arg for arg in node.args]
    for idx, arg in enumerate(eval_args):
        if isinstance(arg, (list, tuple)):
            eval_args[idx] = [id_to_intermed[node_to_id[a]] if isinstance(a, torch.fx.node.Node) else a for a in arg]
    kwargs = {k:v for k, v in node.kwargs.items() if k != "device"}

    if isinstance(node.target, torch._ops.OpOverloadPacket):
        # We will add NOP in cases where input to current subgraph is left on device
        # For input nodes, node.target is str
        id_to_intermed[nid] = node.target(*eval_args, **kwargs)
    if (pybuda_node.wrap_tuple):
        nid = (nid,)
    return nid

def add_input(graph, node, subgraph_idx, module_inputs):
    nid = create_activation_input(
            graph,
            f"{node.name}_{subgraph_idx}",
            [int(dim) for dim in node.meta['tensor_meta'].shape],
            node.meta["tensor_meta"].requires_grad,
            pytorch_dtype_to_buda_dataformat(node.meta["tensor_meta"].dtype),
            subgraph_idx)
    module_inputs.append(nid)
    return nid
    

def add_constant(graph, name, tensor, subgraph_idx):
    if tensor in const_to_id:
        return const_to_id[tensor]
    nid = create_constant_input(
            graph, 
            f"{name}_{subgraph_idx}",
            tensor,
            [int(dim) for dim in tensor.shape],
            pytorch_dtype_to_buda_dataformat(tensor.dtype),
            subgraph_idx)
    const_to_id[tensor] = nid
    return nid

def add_param(graph, name, torch_param, subgraph_idx):
    if name in param_to_id:
        return param_to_id[name]
    nid = create_parameter_input(
            graph, 
            name,
            [int(dim) for dim in torch_param.shape],
            torch_param.requires_grad,
            pytorch_dtype_to_buda_dataformat(torch_param.dtype),
            subgraph_idx)
    param_to_id[name] = nid
    return nid

def add_outputs(graph, node, subgraph_idx, output_nids, output_requires_grad, output_tensors):
    global node_to_id
    for index, meta in enumerate(node.meta['tensor_meta']):
        arg = node.args[0][index]
        nid = create_output(
                graph, 
                node.name + "_" + arg.name + "_" + str(subgraph_idx),
                [int(dim) for dim in meta.shape],
                pytorch_dtype_to_buda_dataformat(meta.dtype),
                False,  #TODO Loss output
                subgraph_idx)
        create_data_edge(graph, node_to_id[arg], 0, nid, index, [])
        output_nids.append(nid)
        output_requires_grad.append(meta.requires_grad)
        output_tensors.append(id_to_intermed[node_to_id[arg]])

def add_constants_if_necessary(graph, ops, subgraph_idx):
    global node_to_id
    for op in ops:
        if isinstance(op, (float, int)):
            if op in node_to_id:
                continue
            tensor = torch.ones([1]) * op
            node_to_id[op] = add_constant(graph, f"{op}", tensor, subgraph_idx)
            id_to_intermed[node_to_id[op]] = tensor


def map_node_name_to_org_name(module, aten_module):
    ret = dict()

    # param nodes
    aten_params = dict()
    for itm in aten_module.named_parameters():
        aten_name = itm[0]
        aten_tensor = itm[1]
        aten_params[id(aten_tensor)] = aten_name
    module_params = dict()
    for itm in module.named_parameters():
        module_name = itm[0]
        mod = itm[1]
        module_params[id(mod)] = module_name
    if len(module_params) == len(aten_params):
        for tensor_id in module_params.keys():
            ret[aten_params[tensor_id]] = module_params[tensor_id]

    # buffers
    aten_buffers = dict()
    for itm in aten_module.named_buffers():
        aten_name = itm[0]
        aten_tensor = itm[1]
        if len(aten_tensor.shape) == 0:
            continue
        aten_buffers[id(aten_tensor)] = aten_name
    module_buffers = dict()
    for itm in module.named_buffers():
        mod_name = itm[0]
        mod_tensor = itm[1]
        if len(mod_tensor.shape) == 0:
            continue
        module_buffers[id(mod_tensor)] = mod_name
    if len(module_buffers) == len(aten_buffers):
        for tensor_id in module_buffers.keys():
            ret[aten_buffers[tensor_id]] = module_buffers[tensor_id]

    return ret


def append_to_graph(graph, module, aten_module, activations, subgraph_idx, inputs_per_subgraph, outputs_per_subgraph):
    param_name_map = map_node_name_to_org_name(module, aten_module)

    tt_act = [a.to("tt") for a in activations]

    # Run static shape propagation on aten module
    shape_prop = torch.fx.passes.shape_prop.ShapeProp(aten_module)
    if shape_prop.fake_mode is not None:
        fake_args = [shape_prop.fake_mode.from_tensor(t, static_shapes=True) if isinstance(t, torch.Tensor) else t for t in tt_act]
    else:
        fake_args = tt_act
    shape_prop.run(*fake_args)

    aten_module = aten_module.to("cpu")

    module_inputs = []
    output_nids = []
    output_requires_grad = []
    output_tensors = []

    def process_function(node):
        global node_to_id
        op_name = node.target.__name__

        if op_name in torch_constant_ops:
            kwargs = {k:v for k, v in node.kwargs.items() if k != "device"}
            tensor = torch_constant_ops[op_name](*node.args, **kwargs)
            if len(tensor.shape) == 0:
                tensor = tensor.unsqueeze(0)
            node_to_id[node] = add_constant(graph, node.name, tensor.float(), subgraph_idx)
            id_to_intermed[node_to_id[node]] = tensor
        elif op_name == "getitem":
            assert isinstance(node_to_id[node.args[0]], (list, tuple))
            assert node.args[1] == 0, "currently getitem only supported for index = 0"
            node_to_id[node] = node_to_id[node.args[0]][node.args[1]]
            id_to_intermed[node_to_id[node]] = id_to_intermed[node_to_id[node]][node.args[1]]
        else:
            pybuda_node = get_pybuda_node(op_name, node)
            node_to_id[node] = add_op(graph, node, node.name, pybuda_node, subgraph_idx)

    # Traverse up the graph from output nodes to populate consumed nodes set
    consumed = set()
    working_nodes = []
    for node in aten_module.graph.nodes:
        if node.op == "output":
            working_nodes.append(node)
            consumed.add(node)

    while len(working_nodes) > 0:
        node = working_nodes.pop(0)
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node) and arg not in consumed:
                consumed.add(arg)
                working_nodes.append(arg)
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.fx.node.Node) and item not in consumed:
                        consumed.add(item)
                        working_nodes.append(item)


    input_index = 0
    for index, node in enumerate(aten_module.graph.nodes):
        if node not in consumed:
            logger.debug(f"Skipping {node} because it was not consumed")
            continue

        if node.op == "placeholder":
            uid = inputs_per_subgraph[subgraph_idx][input_index]
            if uid != -1:
                # this input is on device, don't create input node, add edge to corresponding output
                node_to_id[node] = add_input(graph, node, subgraph_idx, module_inputs)

                for idx in range(subgraph_idx):
                    if uid not in outputs_per_subgraph[idx]:
                        continue
                    output_index = outputs_per_subgraph[idx].index(uid)
                    add_subgraph_io_link_edge(graph, output_nodes_per_subgraph[idx][output_index], 0, node_to_id[node], 0)
            else:
                node_to_id[node] = add_input(graph, node, subgraph_idx, module_inputs)
            id_to_intermed[node_to_id[node]] = activations[index]
            input_index +=1
        elif node.op == "get_attr":
            assert node.target in param_name_map, f"Weight node is not mapped to original names: {node.target}"
            node_to_id[node] = add_param(graph, param_name_map[node.target], aten_module.state_dict()[node.target], subgraph_idx)
            id_to_intermed[node_to_id[node]] = aten_module.state_dict()[node.target]
        elif node.op == "call_function":
            process_function(node)
        elif node.op == "output":
            add_outputs(graph, node, subgraph_idx, output_nids, output_requires_grad, output_tensors)
        else:
            assert False, f"Unsupported op {node.op}"

    graph.register_module_inputs(module_inputs, append=True)
    graph.register_module_outputs(output_nids, output_requires_grad, append=True)

    output_nodes_per_subgraph[subgraph_idx] = output_nids
    return graph, id_to_intermed, output_tensors


def call_function_is_nop(node):
    assert node.op == "call_function"
    op_name = node.target.__name__
    if op_name in dynamo_to_pybuda_function:
        return dynamo_to_pybuda_function[op_name][1] == "nop"
    else:
        return False

def call_function_is_reshape(node):
    assert node.op == "call_function"
    op_name = node.target.__name__
    if op_name in dynamo_to_pybuda_function:
        return dynamo_to_pybuda_function[op_name][1] == "reshape"
    else:
        return False

def unsupported_shared_embedding_input(graph: torch.fx.GraphModule, unsupported_nodes: Set[torch.fx.Node], unsupported_outputs: Set[torch.fx.Node]):
    # Embedding input is untilized integer input. No other op can handle it, other than a "tilize" op, which currently is not implemented. So, we'll mark it as unsupported.

    def search_up(node: torch.fx.Node, visited: Set[torch.fx.Node]):
        if node in visited:
            return 

        if not isinstance(node, torch.fx.Node):
            return

        visited.add(node)
                
        for user in node.users:
            if user in visited:
                continue
            if user.op == "call_function" and user.target.__name__ == "embedding":
                continue
            if user.op == "output":
                unsupported_outputs.add(raw_input)
                continue
            unsupported_nodes.add(user)

        for arg in node.all_input_nodes:
            search_up(arg, visited)


    for node in graph.nodes:
        if node.op == "call_function" and node.target.__name__ == "embedding":
            raw_input = node.args[1]
            visited = set()
            search_up(raw_input, visited)

def get_unsupported_nodes(graph: torch.fx.Graph, config: CompilerConfig) -> Tuple[Set[torch.fx.Node], Set[torch.fx.Node]]:
    # Traverse the FX graph and find all the nodes that are not supported and should fall back to CPU
    # Returns a set of unsupported nodes, and a set of unsupported outputs - since there's only one output node,
    # we represent those by nodes that drive the output, and have to be in a separate set
    unsupported_nodes = set()
    unsupported_outputs = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue

        op_name = node.target.__name__

        if op_name in torch_constant_ops:
            continue
        
        if op_name == "getitem":
            continue

        if op_name in config.cpu_fallback_ops:
            unsuppored_nodes.add(node)
            continue

        if is_supported_op(op_name, node):
            continue

        unsupported_nodes.add(node)

    # Additional passes to find unsupported patterns
    unsupported_shared_embedding_input(graph, unsupported_nodes, unsupported_outputs)

    if len(unsupported_outputs) > 0 or len(unsupported_nodes) > 0:
        logger.trace("Unsupported nodes: " + str(unsupported_nodes) + " Unsupported outputs: " + str(unsupported_outputs))
        
    return unsupported_nodes, unsupported_outputs
