# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Verify by evaluating the pybuda graph
"""

import os
from typing import Tuple, Dict, List, Any

from loguru import logger
from pybuda.pybudaglobal import align_up_tile
import torch

from ..tensor import Tensor, TensorShape, pad_pytorch_tensor_to_buda, narrow_buda_tensor_to_pytorch
from .config import VerifyConfig, should_waive_gradient
from ..config import PerfTraceLevel
import pybuda._C.graph as pygraph
from pybuda._C.backend_api import BackendType, DeviceMode
from pybuda.tensor import pytorch_tensor_to_tensor_desc
from ..backend import BackendAPI
from pybuda.tools.run_net2pipe import net2pipe

def _generate_random_losses(outputs, is_buda):
    losses = []
    for out in outputs:
        if out.requires_grad:
            shape = list(out.shape.get_pytorch_shape())
            if is_buda:
                while len(shape) < 4:
                    shape.insert(0, 1)
                while len(shape) > 4:
                    shape.pop(0)
                
                shape[-1] = align_up_tile(shape[-1])
                shape[-2] = align_up_tile(shape[-2])

            losses.append(torch.rand(shape, dtype=out.pt_data_format))
    return losses

def _run_pytorch_backward(outputs, device, losses):
    retain_graph = True
    for i, o in enumerate(outputs):
        if o.requires_grad:
            if device.loss_module is None:
                loss = narrow_buda_tensor_to_pytorch(losses[i], o.value().shape)
                o.value().backward(loss, retain_graph=retain_graph)
            else:
                o.value().backward(retain_graph=True) # this is loss

def get_intermediate_tensors(
        graph: pygraph.Graph,
        inputs: Tuple[Tensor, ...],
        parameters: Dict[str, torch.Tensor],
        device: "TTDevice",
        is_buda: bool,
):
    torch_inputs: List[torch.Tensor] = [i.value() for i in inputs]

    if is_buda:
        torch_inputs = [pad_pytorch_tensor_to_buda(t, graph.get_tile_broadcast_dims_for_input(i)) for i, t in enumerate(torch_inputs)]
    intermediates = pygraph.get_intermediate_tensors(graph, torch_inputs, parameters, device, relative_atol=1.0, pcc = 0.0)
    return intermediates

def do_verify(
        stage_name: str,
        training: bool,
        graph: pygraph.Graph,
        inputs: Tuple[Tensor, ...],
        parameters: Dict[str, torch.Tensor],
        golden_input_grads: Tuple[torch.Tensor, ...],
        outputs: Tuple[Tensor, ...],
        device: "TTDevice",
        intermediate_golden_tensors: Dict,
        verify_cfg: VerifyConfig,
        is_buda: bool,
        losses=None,
        targets: List[Tensor] = [],
        balancer_solution=None):
    """
    Verify graph vs. pytorch golden
    """
    from pybuda.op.eval import compare_tensor_to_golden # avoid circular import

    torch_inputs: List[torch.Tensor] = [i.value() for i in inputs]
    torch_targets: List[torch.Tensor] = [i.value() for i in targets]

    if is_buda:
        torch_inputs = [pad_pytorch_tensor_to_buda(
                tensor=t,
                tile_broadcast_dims=graph.get_tile_broadcast_dims_for_input(i),
                squeeze=False,
                microbatch=1,
                tile_r=graph.get_ordered_input_tile_dims()[i][0],
                tile_c=graph.get_ordered_input_tile_dims()[i][1],)
            for i, t in enumerate(torch_inputs)
        ]
        
    if device.loss_module is not None:
        assert len(targets) > 0, f"No target provided, but device {device} has a loss module"

    logger.info("Verifying stage {}", stage_name)
    if not training:

        pcc = 0.0 if verify_cfg.pcc is None else verify_cfg.pcc
        trace_outputs, *_ = pygraph.eval(graph, torch_inputs, parameters, device, verify_cfg.relative_atol, pcc, intermediate_golden_tensors, balancer_solution=balancer_solution, dump_tensors_path=verify_cfg.dump_tensors_path, targets=torch_targets)

        # Verify forward pass results
        ok = True
        for i, result in enumerate(zip(outputs, trace_outputs)):
            evaled = result[1]
            golden = result[0].value()
            ok &= compare_tensor_to_golden(f"Output {i}", golden, evaled, is_buda=is_buda, verify_cfg=verify_cfg)

    else:
        if losses is None and device.loss_module is None:
            losses = _generate_random_losses(outputs, is_buda)
        elif losses is None:
            losses = []

        # retain intermediate gradients for verification
        for t in intermediate_golden_tensors.values():
            if t.requires_grad == True:
                t.retain_grad()

        # Calculate pytorch gradients
        run_backward = False
        for i, o in enumerate(outputs):
            # Check if we need to run, or if gradients have been calculated already
            if o.value().grad is None and o.requires_grad:
                run_backward = True
                break 
        if run_backward:
            _run_pytorch_backward(outputs, device, losses)

        pcc = 0.0 if verify_cfg.pcc is None else verify_cfg.pcc
        trace_outputs, parameter_to_gradients, bwd_gradients, parameter_to_updated_parameter = pygraph.eval(
            graph,
            torch_inputs,
            parameters,
            tt_device=device,
            relative_atol=verify_cfg.relative_atol,
            pcc=pcc,
            intermediate_golden_tensors=intermediate_golden_tensors,
            losses=losses,
            targets=torch_targets,
            balancer_solution=balancer_solution,
            dump_tensors_path=verify_cfg.dump_tensors_path
        )

        # Verify forward pass results
        ok = True
        for i, result in enumerate(zip(outputs, trace_outputs)):
            evaled = result[1]
            golden = result[0].value()
            ok &= compare_tensor_to_golden(f"Output {i}", golden, evaled, is_buda=is_buda, verify_cfg=verify_cfg)

        # Verify bwd gradients
        # allow 0 on golden below because on the first post-autograd pass we don't have golden input grads yet
        assert len(golden_input_grads) == 0 or (len(golden_input_grads) == len(bwd_gradients)), f"Golden has {len(golden_input_grads)} input gradients, but graph eval returned {len(bwd_gradients)}"
        for bwd_index, golden_input_grad in enumerate(golden_input_grads):
            evaled = bwd_gradients[bwd_index]
            ok &= compare_tensor_to_golden(f"Bwd gradient {bwd_index}", golden_input_grad, evaled, is_buda=is_buda, verify_cfg=verify_cfg)

        # Verify parameter gradients:
        device_parameters = device.get_parameters()
        for parameter in device_parameters:
            if parameter.requires_grad:
                parameter_name = parameter.get_name()
                if not parameter_name in parameter_to_gradients:
                    logger.warning("Parameter {} not used.", parameter_name)
                    continue

                golden = parameter.value().grad
                assert golden is not None
                evaled = parameter_to_gradients[parameter_name]
                warning_only = should_waive_gradient(parameter_name, verify_cfg)
                ok &= compare_tensor_to_golden(f"Gradient for {parameter_name}", golden, evaled, is_buda=is_buda, verify_cfg=verify_cfg, warning_only=warning_only)

        # Verify parameter updates:
        optimizer = device.get_optimizer()
        if optimizer:
            for parameter in device_parameters:
                if parameter.requires_grad:
                    parameter_name = parameter.get_name()
                    if not parameter_name in parameter_to_updated_parameter:
                        logger.warning("Parameter {} not used.", parameter_name)
                        continue

                    golden = optimizer.torch_parameter_update(
                        parameter_name=parameter_name,
                        parameter=parameter.value(),
                        gradient=parameter.value().grad
                    )
                    evaled = parameter_to_updated_parameter[parameter_name]
                    warning_only = should_waive_gradient(parameter_name, verify_cfg)
                    ok &= compare_tensor_to_golden(f"Parameter Update for {parameter_name}", golden, evaled, is_buda=is_buda, verify_cfg=verify_cfg, warning_only=warning_only)

    msg = f"Stage {stage_name}: Data mismatch detected"
    if not ok:
        logger.error(msg)

    continue_on_mismatch = bool(int(os.environ.get("PYBUDA_CONTINUE_ON_MISMATCH", "0")))
    if not continue_on_mismatch:
        assert ok, msg
    return losses

# Make sure to clean up after ourselves, even if an abort happens
def atexit_handler(backend_api):
    backend_api.shutdown()

def verify_golden(
        netlist_filename: str, 
        training: bool,
        compile_results: Any,
        device: "TTDevice",
        inputs: Tuple[Tensor], 
        outputs: Tuple[torch.Tensor],
        verify_cfg: VerifyConfig):

    logger.info("Running golden backend verify")

    backend_api = None
    try:
        from pybuda.compiled_graph_state import CompiledGraphState
        compiled_graph_state = CompiledGraphState.from_compiled_graph(device, compile_results)
        backend_api = BackendAPI(
            BackendType.Golden,
            device.arch,
            device,
            netlist_filename,
            compiled_graph_state,
            False,
            None,
            None,
            PerfTraceLevel.NONE,
            DeviceMode.CompileAndRun)

        backend_api.push_constants_and_parameters()
        backend_api.push_optimizer_parameters()

        iqs = backend_api.get_ordered_input_queues()
        assert len(inputs) == len(iqs)

        padded_inputs = []
        for i, t in enumerate(inputs):
            padded_tensor = pad_pytorch_tensor_to_buda(t.value(), compiled_graph_state.ordered_input_tile_broadcast_dims[i])
            padded_inputs.append(pytorch_tensor_to_tensor_desc(padded_tensor))
        BackendAPI.push_to_queues(iqs, padded_inputs, single_input=True)

        # Run fwd program
        backend_api.schedule_run_forward(loop_count=1)

        # Get outputs, and check them
        from pybuda.op.eval import compare_tensor_to_golden  # avoid circular import
        oq = backend_api.get_ordered_output_queues()
        assert len(oq) == len(outputs)
        calculated_outputs = BackendAPI.read_queues(oq, [g.value().shape for g in outputs], None, [False] * len(oq), single_output=True, has_microbatch_dim=False)

        ok = True
        for i, (golden, calculated) in enumerate(zip(outputs, calculated_outputs)):
            output_tensor = calculated.value()
            golden = golden.value().type(output_tensor.dtype)
            output_tensor = narrow_buda_tensor_to_pytorch(output_tensor, golden.shape)
            ok &= compare_tensor_to_golden(f"Output {i}", golden, output_tensor, verify_cfg=verify_cfg)

        BackendAPI.pop_queues(oq, single_output=True)

    finally:
        # Make sure to clean up
        if backend_api is not None:
            backend_api.shutdown()

    assert ok, "Verify Golden: Data mismatch detected"


def verify_net2pipe(netlist, device_yaml, cluster_cfg_yaml):
    level = int(os.environ.get("PYBUDA_VERIFY_NET2PIPE", "0"))
    returncode, error_message = net2pipe(netlist, device_yaml=device_yaml, cluster_cfg_yaml=cluster_cfg_yaml, stats=(level > 3), run_pipegen=(level > 1), run_blobgen=(level > 2))
    ok = returncode == 0
    return ok, error_message
