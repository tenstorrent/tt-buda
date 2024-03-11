# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#
# Capture the FX graph and convert to MixedGraph of PyBuda and CPU graphs
#

from typing import Dict, List, Optional
import torch
from loguru import logger

from .mixed_graph import MixedGraph, reduce_graph
from pybuda._C.graph import create_op_node, create_data_edge, create_parameter_input, create_activation_input, create_output, create_constant_input, OpType, add_subgraph_io_link_edge
from pybuda.tensor import pytorch_dtype_to_buda_dataformat
from pybuda.fx.nodes import get_pybuda_node, torch_constant_ops, is_supported_op, get_unsupported_nodes
from pybuda.config import _get_global_compiler_config
import pybuda

class CaptureFX:
    def __init__(self):
        self.graph : Optional[MixedGraph] = None
        self.node_to_id : Dict[torch.fx.Node, int] = {}
        self.param_to_id : Dict[str, int] = {}
        self.const_to_id : Dict[torch.Tensor, int] = {}
        self.id_to_intermed : Dict[int, torch.Tensor] = {}
        self.output_nodes_per_subgraph : Dict[int, List] = {}

    def reset_state(self):
        self.graph = None
    
    def capture_sample_outputs(self, outputs: List[torch.Tensor], subgraph_id: int):
        assert self.graph is not None
        self.graph.capture_sample_outputs(outputs, subgraph_id)

    def get_buda_graph(self) -> pybuda._C.graph.Graph:
        assert self.graph is not None
        return self.graph.graph

    def append_to_graph(
            self, 
            module_name: str,
            module: torch.nn.Module, 
            aten_module: torch.nn.Module, 
            sample_inputs: List[torch.Tensor],
            subgraph_id: int):

        if self.graph is None:
            self.graph = MixedGraph(module_name)

        self.graph.capture_sample_inputs(inputs=sample_inputs, subgraph_id=subgraph_id)

        activations = [torch.rand(sample_input.shape).to(sample_input.dtype).to("cpu") for sample_input in sample_inputs]
        _, intermediate_tensors, output_tensors = self._append_to_graph(module, aten_module, activations, subgraph_id)
        logger.debug(f"Appending to graph done, captured {len(self.get_buda_graph().nodes())} nodes")
        return intermediate_tensors, output_tensors

    def eval_node(self, node):
        assert isinstance(node.target, torch._ops.OpOverloadPacket)
    
        eval_args = [self.id_to_intermed[self.node_to_id[arg]] if isinstance(arg, torch.fx.node.Node) else arg for arg in node.args]
        for idx, arg in enumerate(eval_args):
            if isinstance(arg, (list, tuple)):
                eval_args[idx] = [self.id_to_intermed[self.node_to_id[a]] if isinstance(a, torch.fx.node.Node) else a for a in arg]
        kwargs = {k:v for k, v in node.kwargs.items() if k != "device"}
    
        return node.target(*eval_args, **kwargs)
    
    
    def add_op(self, node, name, pybuda_node, subgraph_idx):
        shape = node.meta['tensor_meta'].shape if pybuda_node.shape is None else pybuda_node.shape
        dtype = pytorch_dtype_to_buda_dataformat(node.meta['tensor_meta'].dtype) if pybuda_node.dtype is None else pybuda_node.dtype
    
        self.add_constants_if_necessary(pybuda_node.args, subgraph_idx)
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
                self.get_buda_graph(),
                f"{name}_{subgraph_idx}",
                pybuda_node.op,
                [int(dim) for dim in shape],
                pytorch_dtype_to_buda_dataformat(dtype),
                subgraph_idx,
                tags)
        
        for i, input_node in enumerate(pybuda_node.args):
            create_data_edge(self.get_buda_graph(), self.node_to_id[input_node], 0, nid, i, [])
    
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            # We will add NOP in cases where input to current subgraph is left on device
            # For input nodes, node.target is str
            self.id_to_intermed[nid] = self.eval_node(node)
    
        if (pybuda_node.wrap_tuple):
            nid = (nid,)
        return nid
    
    def add_input(self, node, subgraph_idx, module_inputs):
        nid = create_activation_input(
                self.get_buda_graph(),
                f"{node.name}_{subgraph_idx}",
                [int(dim) for dim in node.meta['tensor_meta'].shape],
                node.meta["tensor_meta"].requires_grad,
                pytorch_dtype_to_buda_dataformat(node.meta["tensor_meta"].dtype),
                subgraph_idx)
        module_inputs.append(nid)
        return nid
        
    
    def add_constant(self, name, tensor, subgraph_idx):
        if tensor in self.const_to_id:
            return self.const_to_id[tensor]
        nid = create_constant_input(
                self.get_buda_graph(), 
                f"{name}_{subgraph_idx}",
                tensor,
                [int(dim) for dim in tensor.shape],
                pytorch_dtype_to_buda_dataformat(tensor.dtype),
                subgraph_idx)
        self.const_to_id[tensor] = nid
        return nid
    
    def add_param(self, name, torch_param, subgraph_idx):
        if name in self.param_to_id:
            return self.param_to_id[name]
        nid = create_parameter_input(
                self.get_buda_graph(), 
                name,
                [int(dim) for dim in torch_param.shape],
                torch_param.requires_grad,
                pytorch_dtype_to_buda_dataformat(torch_param.dtype),
                subgraph_idx)
        self.param_to_id[name] = nid
        return nid
    
    def add_outputs(self, node, subgraph_idx, output_nids, output_requires_grad, output_tensors):
        for index, meta in enumerate(node.meta['tensor_meta']):
            arg = node.args[0][index]
            nid = create_output(
                    self.get_buda_graph(), 
                    node.name + "_" + arg.name + "_" + str(subgraph_idx),
                    [int(dim) for dim in meta.shape],
                    pytorch_dtype_to_buda_dataformat(meta.dtype),
                    False,  #TODO Loss output
                    subgraph_idx)
            create_data_edge(self.get_buda_graph(), self.node_to_id[arg], 0, nid, index, [])
            output_nids.append(nid)
            output_requires_grad.append(meta.requires_grad)
            output_tensors.append(self.id_to_intermed[self.node_to_id[arg]])
    
    def add_constants_if_necessary(self, ops, subgraph_idx):
        for op in ops:
            if isinstance(op, (float, int)):
                if op in self.node_to_id:
                    continue
                tensor = torch.ones([1]) * op
                self.node_to_id[op] = self.add_constant(f"{op}", tensor, subgraph_idx)
                self.id_to_intermed[self.node_to_id[op]] = tensor
    
    
    def map_node_name_to_org_name(self, module, aten_module):
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

    def process_function(self, node, subgraph_idx, module_inputs):
        op_name = node.target.__name__

        if op_name in torch_constant_ops:
            kwargs = {k:v for k, v in node.kwargs.items() if k != "device"}
            tensor = torch_constant_ops[op_name](*node.args, **kwargs)
            if len(tensor.shape) == 0:
                tensor = tensor.unsqueeze(0)
            self.node_to_id[node] = self.add_constant(node.name, tensor.float(), subgraph_idx)
            self.id_to_intermed[self.node_to_id[node]] = tensor
        elif op_name == "getitem":
            assert isinstance(self.node_to_id[node.args[0]], (list, tuple))
            assert node.args[1] == 0, "currently getitem only supported for index = 0"
            self.node_to_id[node] = self.node_to_id[node.args[0]][node.args[1]]
            self.id_to_intermed[self.node_to_id[node]] = self.id_to_intermed[self.node_to_id[node]][node.args[1]]
        elif is_supported_op(op_name):
            pybuda_node = get_pybuda_node(op_name, node)
            self.node_to_id[node] = self.add_op(node, node.name, pybuda_node, subgraph_idx)
        else:
            # Unsupported function, fall back to CPU
            assert False, f"Unsupported function {op_name}"

            #logger.warning(f"Unsupported function {op_name}, falling back to CPU")
            #fg = torch.fx.Graph()
            #arg_remap = {}
            #self.node_to_id[node] = fg.node_copy(node, lambda n : self.node_to_id[n])

            # Create an input from the fallback
            #self.node_to_id[node] = self.add_input(node, subgraph_idx, module_inputs)

            # Add to fallback list, which we're going to use to create fallback graphs later
            #fallback_ops.append(node)

            # Record the intermed value
            #self.id_to_intermed[self.node_to_id[node]] = self.eval_node(node)

    
    def _append_to_graph(self, module, aten_module, activations, subgraph_idx):
    
        param_name_map = self.map_node_name_to_org_name(module, aten_module)
    
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

        # Remove unused nodes
        reduce_graph(aten_module)

        # Find unsupported nodes
        fallback_ops = get_unsupported_nodes(aten_module)

        # Filter out unsupported nodes into separate FX graphs
        self.graph.filter_unsupported_nodes(aten_module, fallback_ops, subgraph_idx)

        if len(aten_module.graph.nodes) == 0:
            # Nothing left in the graph
            logger.debug("No nodes left in the device graph after fallback, skipping")
            return self.get_buda_graph(), self.id_to_intermed, output_tensors

        # Now convert whatever is left
        input_index = 0
        for index, node in enumerate(aten_module.graph.nodes):
    
            if node.op == "placeholder":
                assert self.graph
                uid = self.graph.get_subgraph_input(subgraph_idx, input_index)
                if uid != -1 and _get_global_compiler_config().enable_pt2_fx_graph_link:
                    # this input is on device, don't create input node, add edge to corresponding output
                    self.node_to_id[node] = self.add_input(node, subgraph_idx, module_inputs)
    
                    idx, output_index = self.graph.get_output_index(uid)
                    add_subgraph_io_link_edge(self.get_buda_graph(), self.output_nodes_per_subgraph[idx][output_index], 0, self.node_to_id[node], 0)
                else:
                    self.node_to_id[node] = self.add_input(node, subgraph_idx, module_inputs)
                self.id_to_intermed[self.node_to_id[node]] = activations[index]
                input_index +=1
            elif node.op == "get_attr":
                assert node.target in param_name_map, f"Weight node is not mapped to original names: {node.target}"
                self.node_to_id[node] = self.add_param(param_name_map[node.target], aten_module.state_dict()[node.target], subgraph_idx)
                self.id_to_intermed[self.node_to_id[node]] = aten_module.state_dict()[node.target]
            elif node.op == "call_function":
                self.process_function(node, subgraph_idx, module_inputs)
            elif node.op == "output":
                self.add_outputs(node, subgraph_idx, output_nids, output_requires_grad, output_tensors)
            else:
                assert False, f"Unsupported op {node.op}"
    
        self.get_buda_graph().register_module_inputs(module_inputs, append=True)
        self.get_buda_graph().register_module_outputs(output_nids, output_requires_grad, append=True)
    
        self.output_nodes_per_subgraph[subgraph_idx] = output_nids
        return self.get_buda_graph(), self.id_to_intermed, output_tensors
    
