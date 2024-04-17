# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from sys import intern
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field

import torch
from loguru import logger

from .ttdevice import TTDevice
from .tensor import Tensor, pad_pytorch_tensor_to_buda
from pybuda._C import (
    BudaNetlist,
    link_past_cache_ios,
    move_index_to_mm_weights,
    run_post_initial_graph_passes,
    run_optimization_graph_passes,
    run_post_optimize_decompose_graph_passes,
    run_consteval_graph_pass,
    run_post_autograd_graph_passes,
    run_pre_placer_buda_passes,
    run_post_placer_buda_passes,
    run_pre_netlist_generation_buda_passes,
    run_placer_buda_passes,
    run_pre_lowering_passes,
    lower_to_buda_netlist,
    merge_netlists,
    dump_graph,
    dump_epoch_type_graphs,
    dump_epoch_id_graphs,
    is_subset_of_instructions,
    PostPlacerConfig,
    UnsupportedHWOpsError,
    NopInsertionInstruction,
    PostPlacerResults
)
import pybuda
from .parameter import Parameter
from pybuda._C.backend_api import BackendCompileFailure, BackendType, BackendDevice, DeviceConfig
import pybuda._C.autograd as pyautograd
import pybuda._C.balancer as pybalancer
import pybuda._C.pattern_matcher as pypattern_matcher
import pybuda._C.scheduler as pyscheduler
from pybuda._C.placer import match_op_names_to_placer_overrides, PlacerConfigUpdate, PlacerSolution
from pybuda._C.graph import Graph
import pybuda.query as query
from .verify import VerifyConfig, do_verify, verify_golden, verify_net2pipe, _generate_random_losses, _run_pytorch_backward, get_intermediate_tensors
import pybuda._C.graph as pygraph
from .config import (
    CompilerConfig,
    CompileDepth,
    _get_global_compiler_config,
)
from .pybudaglobal import state_changed, clear_state_changed
from pybuda import PyBudaModule
from .tensor import Tensor, to_pt_tensors, to_buda_tensors
from . import ci, utils
from pybuda.tools.net2reportify import net2placement
from .backend import BackendCompileException

LAST_SUCCESSFUL_STAGE = None
def init_log_last_successful_compile_stage():
    global LAST_SUCCESSFUL_STAGE
    LAST_SUCCESSFUL_STAGE = None


def dump_compiler_cfg(backend_output_directory, compiler_cfg, graph_name):
    import yaml
    try:
        int(os.environ["PYBUDA_DUMP_CONFIG"])
        path = f"{graph_name}_config.yaml"
    except ValueError:
        path = os.environ["PYBUDA_DUMP_CONFIG"]
    with open(os.path.join(backend_output_directory, path), "w") as fd:
        yaml.dump(compiler_cfg.to_dict(), fd, indent=2)


def load_compiler_cfg(compiler_cfg, clobber=False):
    import yaml
    import json
    path = os.environ["PYBUDA_LOAD_CONFIG"]
    loader = json.load if os.path.splitext(path)[1] == ".json" else lambda f: yaml.load(f, yaml.SafeLoader)
    with open(path) as fd:
        d = compiler_cfg.to_dict()
        overrides = loader(fd)
        for k, v in overrides.items():
            d[k] = v
        return CompilerConfig.from_dict(d)


def generate_override_config(graph, balancer_solution, placer_solution, nop_instructions, graph_name):
    import yaml
    try:
        int(os.environ["PYBUDA_GENERATE_OVERRIDE_CONFIG"])
        path = f"{graph_name}_override_config.yaml"
    except ValueError:
        path = os.environ["PYBUDA_GENERATE_OVERRIDE_CONFIG"]

    overrides = {}
    overrides["balancer_op_overrides"] = {k: {
        "grid_shape": [v.grid_shape.r, v.grid_shape.c],
        "t_stream_dir": str(v.t_stream_factor.dir).split(".")[1],
        "t_stream_shape": [v.t_stream_factor.r, v.t_stream_factor.c],
        "fracture_factor": v.fracture_factor,
    } for k, v in balancer_solution.op_models.items()}

    overrides["buffering_nops_to_insert"] = [NopInsertionInstruction.to_json(n) for n in nop_instructions]

    overrides["insert_queues"] = list(list(v) for v in balancer_solution.cut_edges_as_override(graph))

    with open(path, "w") as fd:
        yaml.dump(overrides, fd, indent=2)

class CompileResults:
    """
    Wrapper for result from the graph compiler. Contains initial and final graphs, output tensors,
    and, optionally golden results for final output and intermediates, if desired.
    """
    outputs: List[Tensor]
    golden_outputs: List[torch.Tensor]
    golden_intermediates: Dict[str, torch.Tensor]
    initial_graph: Graph
    lowered_graph: Graph
    netlist_filename: str
    perf_model_results: Dict[str, float]

    pass_specific_output_kwargs: Dict[str, Any] = {}

@dataclass
class CompileContext:
    dev: TTDevice
    graph_name: str
    inputs: Tuple[Union[Tensor, List[Any], Dict[str, Any]],...]
    compiler_cfg: CompilerConfig
    verify_cfg: VerifyConfig
    device_cfg: DeviceConfig
    microbatch_size: int
    microbatch_count: int
    graph: Optional[Graph] = None
    scheduler_config: Optional[pyscheduler.SchedulerConfig] = None
    losses: Optional[List[Tensor]] = None
    output_kwargs: Dict[str, Any] = field(default_factory=dict)
    targets: List[Tensor] = field(default_factory=list)
    initial_graph: Optional[Graph] = None
    lowered_graph: Optional[Graph] = None
    stage: CompileDepth = CompileDepth.INIT_COMPILE
    initial_graph_copy: Optional[Graph] = None
    outputs: Tuple[Tensor, ...] = field(default_factory=tuple)
    intermediate_tensors: Dict[int, Tensor] = field(default_factory=dict)
    parameter_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    input_grads: List[torch.Tensor] = field(default_factory=list)
    netlist_filename: Optional[str] = None
    perf_model_results: Optional[Dict[str, float]] = None
    use_interactive_placer: bool = False
    post_placer_results: Optional[PostPlacerResults] = None
    fracture_chip_id_assignments: Dict[str, int] = field(default_factory=dict)
    policy_type: Optional[pybalancer.PolicyType] = None
    placer_config_update: Optional[PlacerConfigUpdate] = None
    post_placer_config: Optional[PostPlacerConfig] = None
    placer_solution: Optional[PlacerSolution] = None
    balancer_solution: Optional[pybalancer.BalancerSolution] = None
    buda_targets: List[Tensor] = field(default_factory=list)
    buda_losses: List[Tensor] = field(default_factory=list)
    placer_retry_count: int = 0
    backend_output_directory: str = ""
    in_recompile: bool = False
    recompile_count: int = 0
    target_cycles_offset: int = 0

def calculate_grads(
        outputs: Tuple[Tensor, ...],
        device: "TTDevice",
        intermediate_golden_tensors: Dict,
        is_buda: bool,
        losses=None):
    """
    Verify graph vs. pytorch golden
    """

    # retain intermediate gradients for verification
    for t in intermediate_golden_tensors.values():
        if t.requires_grad == True:
            t.retain_grad()

    # Calculate pytorch gradients
    run_backward = False
    for o in outputs:
        # Check if we need to run, or if gradients have been calculated already
        if o.value().grad is None and o.requires_grad:
            run_backward = True
            break

    if not losses or run_backward:

        if losses is None and device.loss_module is None:
            losses = _generate_random_losses(outputs, is_buda)

        if run_backward:
            _run_pytorch_backward(outputs, device, losses)

    return losses

def pybuda_compile_from_context(context: CompileContext) -> CompileResults:
    """
    Run front-end compile passes and generate a Buda netlist, with a given compile context.

    Parameters
    ----------
    context: CompileContext
        Contains all needed info to run compile passes.

    Returns
    -------
    CompileResults

    """

    # Map stages to functions which execute them.
    stage_to_func = {
        CompileDepth.INIT_COMPILE: init_compile,
        CompileDepth.GENERATE_INITIAL_GRAPH: generate_initial_graph,
        CompileDepth.POST_INITIAL_GRAPH_PASS: run_post_initial_graph_pass,
        CompileDepth.CONSTEVAL_GRAPH: run_consteval_pass,
        CompileDepth.POST_PATTERN_MATCHER: run_post_pattern_matcher,
        CompileDepth.OPTIMIZED_GRAPH: run_optimization_pass,
        CompileDepth.AUTOGRAD: run_autograd_pass,
        CompileDepth.POST_AUTOGRAD_PASS: run_post_autograd_pass,
        CompileDepth.PRE_LOWERING_PASS: run_pre_lowering_pass,
        CompileDepth.BUDA_GRAPH_PRE_PLACER: run_pre_placer_pass,
        CompileDepth.BALANCER_PASS: run_balancer_and_placer,
        CompileDepth.PRE_NETLIST_PASS: run_pre_netlist_pass,
        CompileDepth.GENERATE_NETLIST: generate_netlist,
        CompileDepth.BACKEND_GOLDEN_VERIFY: run_backend_golden_verify,
    }

    while context.stage != CompileDepth.FULL:
        logger.info("Running compile stage {}", context.stage.name.lower())

        current_stage = context.stage
        verify_cfg = context.verify_cfg
        compiler_cfg = context.compiler_cfg
        dev = context.dev

        # Execute the current stage.
        next_stage = stage_to_func[current_stage](context)

        # Check if we need to stop compilation or perform verifications in the current stage.
        should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, current_stage)

        can_verify = current_stage != CompileDepth.INIT_COMPILE and current_stage != CompileDepth.PRE_LOWERING_PASS and current_stage != CompileDepth.POST_PATTERN_MATCHER
        should_verify = (current_stage == CompileDepth.PRE_NETLIST_PASS and verify_cfg.verify_post_placer) or (current_stage == CompileDepth.POST_AUTOGRAD_PASS and verify_cfg.verify_post_autograd_passes)

        if ci.capture_tensors() and current_stage == CompileDepth.PRE_NETLIST_PASS:
            # We need to dump tensors during PRE_NETLIST_PASS (POST_PLACER)
            verify_cfg.dump_tensors_path = ci.get_netlist_dir()
            should_verify = True

        if (verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation) or should_verify) and can_verify:
            in_training = context.compiler_cfg.enable_training and current_stage.value >= CompileDepth.AUTOGRAD.value
            is_buda = current_stage.value >= CompileDepth.BUDA_GRAPH_PRE_PLACER.value

            if not is_buda:
                do_verify(current_stage.name.lower(), in_training, context.graph, context.inputs, context.parameter_dict, context.input_grads, context.outputs, dev, context.intermediate_tensors, verify_cfg, is_buda, losses=context.losses, targets=context.targets)
            else:
                do_verify(current_stage.name.lower(), in_training, context.lowered_graph, context.inputs, context.parameter_dict, context.input_grads, context.outputs, dev, context.intermediate_tensors, verify_cfg, is_buda, losses=context.buda_losses, targets=context.buda_targets, balancer_solution=context.balancer_solution)

        if should_early_stop_compilation:
            logger.info("Early stopping compilation at stage {}", current_stage.name.lower())
            return generate_compile_results(context.verify_cfg, context.initial_graph_copy, context.outputs, context.intermediate_tensors, context.lowered_graph, context.netlist_filename, context.perf_model_results, pass_specific_output_kwargs=context.output_kwargs)

        context.stage = next_stage

    return generate_compile_results(context.verify_cfg, context.initial_graph_copy, context.outputs, context.intermediate_tensors, context.lowered_graph, context.netlist_filename, context.perf_model_results, pass_specific_output_kwargs=context.output_kwargs)

def pybuda_compile(
        dev: TTDevice,
        graph_name: str,
        *inputs: Union[Tensor, List[Any], Dict[str, Any]],
        targets: List[Tensor] = [],
        compiler_cfg: Optional[CompilerConfig] = None,
        verify_cfg: Optional[VerifyConfig] = None,
        losses: Optional[List[Tensor]] = None,
        microbatch_size: int = 1,
        microbatch_count: int = 1) -> CompileResults:
    """
    Run front-end compile passes and generate a Buda netlist for given input tensors. Optionally verify
    against PyTorch model.

    This version has significant amount of verification built-in, and is primarily used for testing. A "deliverable"
    version that does only the compile will be written in the future.

    Parameters
    ----------
    dev: TTDevice
        Device to compile modules for. Modules should already be placed on the device.

    graph_name: str
        Name to be used in the netlist

    *inputs: Tuple[Tensor, ...]
        Input tensors to compile for. Tensors must have set shapes, but values are only needed for
        automatic verification.

    targets: List[Tensor], optional
        Optional list of target tensors, if this device has a loss module

    verify_cfg: Optional[VerifyConfig]
        If set, automatic verification vs. pytorch golden result will be performed, with given parameters
        must contain data.


    Returns
    -------
    CompileResults

    """

    inputs = list(inputs)
    if verify_cfg is None:
        verify_cfg = VerifyConfig.disabled() # no verification config provided, disable by default

    if compiler_cfg is None:
        compiler_cfg = _get_global_compiler_config()

    compiler_cfg.apply_env_config_overrides()

    compile_context: CompileContext = CompileContext(
        dev=dev,
        graph_name=graph_name,
        inputs=inputs,
        compiler_cfg=compiler_cfg,
        verify_cfg=verify_cfg,
        device_cfg=dev.get_device_config(compiler_cfg=compiler_cfg),
        microbatch_size=microbatch_size,
        microbatch_count=microbatch_count,
        targets=targets,
        losses=losses,
    )

    return pybuda_compile_from_context(compile_context)

def check_for_compilation_early_stop(desired_stage, current_stage):
    """
    Determines should current compilation process stop or not based on desired
    and current phase of execution.

    Parameters
    ----------
    desired_stage: CompileDepth
        Desired phase for compiler early stopping.

    current_stage: CompileDepth
        Current phase for compiler early stopping.

    Returns
    -------
    Boolean
    """
    # update global compile stage variable
    global LAST_SUCCESSFUL_STAGE
    LAST_SUCCESSFUL_STAGE = str(CompileDepth(current_stage.value).name)

    if not CompileDepth.has_value(desired_stage.value):
        raise Exception("Invalid compilation depth flag: {}".format(desired_stage.name))

    if desired_stage.value == current_stage.value:
        logger.info("Compilation early stopping after {}".format(current_stage.name))

        return True

    return False

def placer_breaks_eval(value):
    if type(value) is query.NodePredicateBuilder:
        return value.eval()
    elif type(value) is list:
        return [placer_breaks_eval(v) for v in value]
    else:
        assert type(value) is str
        return value

def placer_op_overrides_eval(value):
    assert type(value) is tuple
    if type(value[0]) is query.NodePredicateBuilder:
        return (value[0].eval(), value[1])
    else:
        return value

def validate_override_names(graph, compiler_cfg):
    """
    Checks whether name in per_op overrides uses depracated naming scheme and warns user.

    Parameters
    ----------
    graph: Graph
        PyBuda Graph

    compiler_cfg: CompilerCfg
        Compiler configuration options

    Returns
    -------
    None
    """
    from pybuda.op.common import depracated_name_dict

    keys = list(compiler_cfg.balancer_op_overrides.keys())
    keys.extend([key[0] for key in compiler_cfg.op_names_to_epoch_break if type(key) is list and type(key[0]) is str])
    keys.extend([key[0] for key in compiler_cfg.op_names_to_chip_break if type(key) is list and type(key[0]) is str])
    for key in keys:
        for depracated_name in depracated_name_dict.keys():
            if key.startswith(depracated_name):
                new_name = key.replace(depracated_name, depracated_name_dict[depracated_name])
                if key in compiler_cfg.balancer_op_overrides:
                    compiler_cfg.balancer_op_overrides[new_name] = compiler_cfg.balancer_op_overrides.pop(key)
                elif [key] in compiler_cfg.op_names_to_epoch_break:
                    compiler_cfg.op_names_to_epoch_break.remove([key])
                    compiler_cfg.op_names_to_epoch_break.append([new_name])
                elif [key] in compiler_cfg.op_names_to_chip_break:
                    compiler_cfg.op_names_to_chip_break.remove([key])
                    compiler_cfg.op_names_to_chip_break.append([new_name])

                logger.warning("Using depracated node name: {}, being replaced by: {}. Please update your test files. ", key, new_name)



def generate_compile_results(
    verify_cfg = None,
    initial_graph = None,
    outputs = None,
    intermediate_tensors = None,
    lowered_graph = None,
    netlist_filename = None,
    perf_model_results = None,
    *,
    pass_specific_output_kwargs = None,
):
    """
    Wrapper for generating result from the graph compiler. Contains initial and final graphs, output tensors,
    and, optionally golden results for final output and intermediates, if desired.

    Parameters
    ----------
    verify_cfg: VerifyConfig
        Value verification config

    initial_graph: Graph
        Initial graph, immediately after conversion from the input framework

    outputs: Tuple[Tensor, ...]
        Output tensors

    intermediate_tensors: Dict[str, Tensor]
        Intermediated tensors

    lowered_graph: Graph
        Buda graph

    netlist_filename: str
        Netlist file name

    Returns
    -------
    CompileResults
    """
    ret = CompileResults()

    ret.initial_graph = initial_graph
    ret.outputs = outputs
    if verify_cfg and verify_cfg.intermediates:
        ret.golden_intermediates = {
            initial_graph.get_node_name(node_id): tensor
            for node_id, tensor in intermediate_tensors.items() if initial_graph.has_node_with_id(node_id)
        }
    ret.lowered_graph = lowered_graph
    ret.netlist_filename = netlist_filename
    ret.perf_model_results = perf_model_results

    if outputs:
        ret.golden_outputs = [out.value() if out.has_value() else None for out in outputs]

    if pass_specific_output_kwargs:
        ret.pass_specific_output_kwargs = pass_specific_output_kwargs

    return ret

def init_compile(context: CompileContext) -> CompileDepth:
    
    dev = context.dev
    compiler_cfg = context.compiler_cfg
    device_cfg = context.device_cfg
    graph_name = context.graph_name
    targets = context.targets

    force_full = bool(int(os.environ.get("PYBUDA_FORCE_FULL_COMPILE_DEPTH", "0")))
    if force_full:
        compiler_cfg.compile_depth = CompileDepth.FULL

    if len(targets) > 0:
        assert dev.loss_module is not None, "Target provided for compilation, but this device has no loss module"

    if dev.loss_module is not None:
        assert len(targets) > 0, f"Device {dev} has a loss module, but no targets were provided for compilation"

    context.backend_output_directory = compiler_cfg.backend_output_dir
    ci.initialize_output_build_directory(context.backend_output_directory)

    device_cfg = dev.get_device_config(compiler_cfg=compiler_cfg)
    logger.info("Device grid size: r = {}, c = {}", device_cfg.grid_size.r, device_cfg.grid_size.c)

    # Set global cluster descriptor file path if not provided by user (it was obtained from backend when getting device config)
    if compiler_cfg.backend_cluster_descriptor_path == "":
        compiler_cfg.backend_cluster_descriptor_path = device_cfg.cluster_config_yaml

    if compiler_cfg.backend_device_descriptor_path == "":
        compiler_cfg.backend_device_descriptor_path = device_cfg.device_yaml

    assert len(device_cfg.chip_ids) > 0, "Trying to compile onto zero chips."
    logger.info("Using chips: {}", device_cfg.chip_ids)

    # compiler_cfg is fully formed
    if "PYBUDA_LOAD_CONFIG" in os.environ:
        compiler_cfg = load_compiler_cfg(compiler_cfg)
    elif "PYBUDA_DUMP_CONFIG" in os.environ:
        dump_compiler_cfg(context.backend_output_directory, compiler_cfg, graph_name)

    init_log_last_successful_compile_stage()

    return CompileDepth.GENERATE_INITIAL_GRAPH

def generate_initial_graph(context: CompileContext) -> CompileDepth:
    """
    Generates initial graph from the input framework.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """

    if context.compiler_cfg.compile_tvm_to_python and context.dev.graph is None:
        module_inputs = context.inputs
        for index, module in enumerate(context.dev.modules):
            if not isinstance(module, PyBudaModule):
                from .tvm_to_python import generate_pybuda_module
                prev_state = state_changed()
                modules, dev_types, module_inputs = generate_pybuda_module(module, to_pt_tensors(module_inputs), context.compiler_cfg, module.name, context.verify_cfg,)
                assert len(modules) == 1, "Attemping to load split model onto single devices"
                context.dev.modules[index] = modules[0]

                if index == 0:
                    context.inputs = module_inputs

                if not(prev_state):
                    clear_state_changed()

            if index < len(context.dev.modules) - 1 and not context.compiler_cfg.compile_subgraphs:
                if module is context.dev.loss_module:
                    if len(module_inputs) == 1:
                        module_inputs = context.dev.modules[index].forward(module_inputs[0], context.targets[0])
                    else:
                        module_inputs = context.dev.modules[index].forward(tuple(module_inputs), tuple(context.targets))
                else:
                    module_inputs = context.dev.modules[index].forward(*module_inputs)

                if isinstance(module_inputs, Tensor):
                    module_inputs = (module_inputs,) # Force a tuple

    if context.dev.graph is None:
        context.graph, context.outputs, context.intermediate_tensors, context.inputs, _ = context.dev.generate_graph(*context.inputs, return_intermediate=context.verify_cfg.intermediates, graph_name=context.graph_name, compiler_cfg=context.compiler_cfg, target_tensors=context.targets, verify_cfg=context.verify_cfg)
    else:
        context.graph = context.dev.graph
        context.intermediate_tensors = context.dev.intermediate_tensors
        context.outputs = context.dev.output_tensors

    context.graph.set_microbatch(context.microbatch_size)
    dump_graph(context.graph, context.graph_name, "initial_graph")
    validate_override_names(context.graph, context.compiler_cfg)
    if context.compiler_cfg.enable_link_past_cache_ios:
        # move index ops to weights if applicable
        move_index_to_mm_weights(context.graph)

        # link past cache ios will change the number on inputs / outputs, so it is called before we clone the initial graph
        new_params = link_past_cache_ios(context.graph)
        inputs_to_remove = []
        for k, v in new_params.items():
            context.dev.modules[-1].add_parameter(k, Parameter(context.inputs[v].value(), requires_grad=False, name=k))
            inputs_to_remove.append(context.inputs[v])
        for i in inputs_to_remove:
            context.inputs.remove(i)

    context.initial_graph_copy = context.graph.clone() # save the original graph for verification and analysis
    context.input_grads = []
    context.parameter_dict = {p.get_name() : p.value(is_buda=False) for p in context.dev.get_parameters()}

    return CompileDepth.POST_INITIAL_GRAPH_PASS

def run_post_initial_graph_pass(context: CompileContext) -> CompileDepth:
    """
    Runs post initial graph passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph_name = context.graph_name
    graph, intermediate_tensors = context.graph, context.intermediate_tensors

    if compiler_cfg.enable_consteval:
        run_consteval_graph_pass(graph)
    inserted_node_id_mapping, context.fracture_chip_id_assignments = run_post_initial_graph_passes(graph, compiler_cfg, compiler_cfg.fracture_groups)

    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        # If we have multi-level of decomposition, some node id might not in the original
        # intermediate tensor dict.
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    dump_graph(graph, graph_name, "decomposed_graph")

    next_stage = CompileDepth.OPTIMIZED_GRAPH
    if compiler_cfg.enable_consteval:
        next_stage = CompileDepth.CONSTEVAL_GRAPH
    elif compiler_cfg.match_subgraph_patterns:
        next_stage = CompileDepth.POST_PATTERN_MATCHER

    return next_stage

def run_consteval_pass(context: CompileContext) -> CompileDepth:
    """
    Runs consteval pass.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph = context.graph
    graph_name = context.graph_name

    run_consteval_graph_pass(graph)
    dump_graph(graph, graph_name, "consteval_graph")

    next_stage = CompileDepth.OPTIMIZED_GRAPH
    if compiler_cfg.match_subgraph_patterns:
        next_stage = CompileDepth.POST_PATTERN_MATCHER

    return next_stage

def run_post_pattern_matcher(context: CompileContext) -> CompileDepth:
    """
    Runs post pattern matcher passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    graph = context.graph
    graph_name = context.graph_name

    graph, match_result = pypattern_matcher.lower_pybuda_to_pattern_matcher(graph, compiler_cfg.match_subgraph_patterns)
    context.output_kwargs["match_result"] = match_result

    if match_result.is_subgraph_loopable:
        dump_graph(graph, graph_name, "looped_graph")

    return CompileDepth.OPTIMIZED_GRAPH

def run_optimization_pass(context: CompileContext) -> CompileDepth:
    """
    Runs optimization passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    device_cfg = context.device_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph, intermediate_tensors = context.graph, context.intermediate_tensors

    run_optimization_graph_passes(graph, device_cfg)
    dump_graph(graph, graph_name, "optimized_graph")

    inserted_node_id_mapping = run_post_optimize_decompose_graph_passes(graph, compiler_cfg)
    dump_graph(graph, graph_name, "decomposed_optimized_graph")
    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    # Workaround for TVM and lack of parameters at the time optimizer is created
    if dev.optimizer:
        if dev.optimizer.device_params:
            dev.optimizer.set_parameters_to_optimize(dev.modules[0].get_parameters())
        dev.optimizer.set_optimizer_parameters()

    next_stage = CompileDepth.POST_AUTOGRAD_PASS
    if context.compiler_cfg.enable_training:
        next_stage = CompileDepth.AUTOGRAD

    return next_stage

def run_autograd_pass(context: CompileContext) -> CompileDepth:
    """
    Runs autograd pass.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph, intermediate_tensors, outputs = context.graph, context.intermediate_tensors, context.outputs

    autograd_config = pyautograd.AutogradConfig(recompute=compiler_cfg.enable_recompute, optimizer=dev.optimizer)
    autograd_engine = pyautograd.AutogradEngine(graph, autograd_config)

    graph = autograd_engine.run()
    dump_graph(graph, graph_name, "post_autograd")

    context.losses = calculate_grads(outputs, dev, intermediate_tensors, False, context.losses)

    # Record calculated input grads from the previous do_verify call and save so that we don't keep re-calculating and
    # accumulating on each verification call
    context.input_grads = [i.value().grad for i in context.inputs if i.value().requires_grad and i.value().grad is not None]

    return CompileDepth.POST_AUTOGRAD_PASS

def run_post_autograd_pass(context: CompileContext) -> CompileDepth:
    """
    Runs post autograd passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph, intermediate_tensors, losses, outputs = context.graph, context.intermediate_tensors, context.losses, context.outputs

    inserted_node_id_mapping = run_post_autograd_graph_passes(graph, compiler_cfg)
    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    dump_graph(graph, graph_name, "post_autograd_passes")
    if compiler_cfg.enable_training:
        calculate_grads(outputs, dev, intermediate_tensors, False, losses)

    return CompileDepth.PRE_LOWERING_PASS

def run_pre_lowering_pass(context: CompileContext) -> CompileDepth:
    """
    Runs pre lowering passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph = context.graph

    run_pre_lowering_passes(graph)
    dump_graph(graph, graph_name, "pre_lowering")

    for parameter in dev.get_parameters():
        parameter._set_fp32_fallback(dev.fp32_fallback)

    context.scheduler_config = pyscheduler.SchedulerConfig(
        scheduler_policy=pyscheduler.policy_from_string(os.environ.get("PYBUDA_SCHEDULER_POLICY", compiler_cfg.scheduler_policy)),
        scheduler_constraints=compiler_cfg.scheduler_constraints,
    )

    context.policy_type = pybalancer.policy_from_string(os.environ.get("PYBUDA_BALANCER_POLICY_TYPE", compiler_cfg.balancer_policy))
    context.use_interactive_placer = (
        compiler_cfg.use_interactive_placer and
        not (bool(int(os.environ.get("PYBUDA_DISABLE_INTERACTIVE_PLACER", "0")))) and
        pybalancer.can_use_interactive_placer(context.policy_type)
    )

    return CompileDepth.BUDA_GRAPH_PRE_PLACER

def run_pre_placer_pass(context: CompileContext) -> CompileDepth:
    """
    Runs pre placer passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    device_cfg = context.device_cfg
    dev = context.dev
    graph_name = context.graph_name
    graph = context.graph
    scheduler_config = context.scheduler_config
    intermediate_tensors = context.intermediate_tensors
    outputs = context.outputs
    losses = context.losses
    targets = context.targets

    instructions = {} if context.post_placer_results is None else context.post_placer_results.ins_instructions
    temp_dict = {}; temp_dict.update(compiler_cfg.buffering_nops_to_insert); temp_dict.update(instructions)
    context.lowered_graph, context.placer_config_update = run_pre_placer_buda_passes(
            graph,
            scheduler_config,
            device_cfg,
            device_cfg.chip_ids,
            list(map(placer_breaks_eval, compiler_cfg.op_names_to_chip_break)),
            list(map(placer_breaks_eval, compiler_cfg.op_names_to_epoch_break)),
            compiler_cfg.op_names_dont_fuse,
            compiler_cfg.op_names_manual_fuse,
            context.fracture_chip_id_assignments,
            compiler_cfg.default_df_override,
            compiler_cfg.default_accumulate_df,
            compiler_cfg.enable_broadcast_splitting or bool(int(os.environ.get("PYBUDA_ENABLE_BROADCAST_SPLITTING", "0"))),
            dev.fp32_fallback,
            compiler_cfg.default_math_fidelity,
            compiler_cfg.enable_auto_fusing,
            compiler_cfg.amp_level or int(os.environ.get("PYBUDA_AMP_LEVEL", "0")),
            compiler_cfg.enable_recompute,
            (bool(int(os.environ.get("PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST", "1"))) and compiler_cfg.output_queues_on_host),
            (bool(int(os.environ.get("PYBUDA_ENABLE_INPUT_QUEUES_ON_HOST", "1"))) and compiler_cfg.input_queues_on_host),
            temp_dict,
            compiler_cfg.insert_queues,
            compiler_cfg.amp_properties,
            compiler_cfg.op_intermediates_to_save,
            context.use_interactive_placer,
            compiler_cfg.enable_device_tilize)
    dump_graph(context.lowered_graph, graph_name, "pre_placer")

    assert(context.lowered_graph is not None)

    # Convert to buda tensors - i.e. 4d / tile-snapped dims
    def to_buda_shapes(tensors):
        if tensors is None or not tensors:
            return tensors

        if isinstance(tensors[0], torch.Tensor):
            return [pad_pytorch_tensor_to_buda(t, context.lowered_graph.get_tile_broadcast_dims_for_bw_input(i)) for i, t in enumerate(tensors)]

        return [t.to_buda_shape(tile_broadcast_dims=context.lowered_graph.get_tile_broadcast_dims_for_target(i)) for i, t in enumerate(tensors)]

    context.buda_losses = to_buda_shapes(losses)
    context.buda_targets = to_buda_shapes(targets)

    if compiler_cfg.enable_training:
        calculate_grads(outputs, dev, intermediate_tensors, True, losses)

    return CompileDepth.BALANCER_PASS

def run_balancer_and_placer(context: CompileContext) -> CompileDepth:
    """
    Runs balancer and placer passes.
    
    Parameters
    ----------
    context: CompileContext

    Returns
    -------
    CompileDepth - next compile stage
    """

    instructions = {} if context.post_placer_results is None else context.post_placer_results.ins_instructions
    op_name_to_placer_overrides = match_op_names_to_placer_overrides(context.lowered_graph, list(map(placer_op_overrides_eval, context.compiler_cfg.placer_op_overrides)))
    balancer_config = pybalancer.BalancerConfig(
        device_config=context.device_cfg,
        scheduler_config=context.scheduler_config,
        policy_type=context.policy_type,
        random_policy_seed=int(os.environ.get("PYBUDA_BALANCER_RANDOM_POLICY_SEED", 0)),
        chip_ids=context.device_cfg.chip_ids,
        chip_placement_policy=pybalancer.chip_placement_policy_from_string(context.compiler_cfg.chip_placement_policy),
        enable_t_streaming = (bool(int(os.environ.get("PYBUDA_ENABLE_T_STREAMING", "0"))) or context.compiler_cfg.enable_t_streaming),
        manual_t_streaming = context.compiler_cfg.manual_t_streaming,
        skip_l1_usage_validation=bool(int(os.environ.get("PYBUDA_SKIP_L1_USAGE_VALIDATION", "0"))),
        input_queues_on_host=context.compiler_cfg.input_queues_on_host,
        output_queues_on_host=context.compiler_cfg.output_queues_on_host,
        default_dram_parameters=(not context.verify_cfg.enabled and (context.microbatch_size == 1)) if context.compiler_cfg.default_dram_parameters is None else context.compiler_cfg.default_dram_parameters,
        op_names_to_epoch_break=context.placer_config_update.op_names_to_epoch_break,
        op_names_to_chip_break=context.placer_config_update.op_names_to_chip_break,
        op_overrides=context.compiler_cfg.balancer_op_overrides,
        op_names_to_chip_id_assignment=context.placer_config_update.op_to_chip_id_assignment,
        op_name_to_placer_overrides=op_name_to_placer_overrides,
        enable_auto_transposing_placement = context.compiler_cfg.enable_auto_transposing_placement,
        graph_solver_self_cut_type = pybalancer.graph_solver_self_cut_type_from_string(os.environ.get("PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE", context.compiler_cfg.graph_solver_self_cut_type)),
        use_interactive_placer = context.use_interactive_placer,
        enable_enumerate_u_kt = context.compiler_cfg.enable_enumerate_u_kt,
        enable_single_buffer_fallback = context.compiler_cfg.enable_single_buffer_fallback,
    )
    balancer_config.target_cycles_offset = context.target_cycles_offset

    try:
        context.balancer_solution, had_balancer_attempts = run_placer_buda_passes(context.lowered_graph, balancer_config, context.fracture_chip_id_assignments, context.compiler_cfg.paddings)
    except UnsupportedHWOpsError as e:
        logger.warning("Found unsupported HW ops, stopping compilation early:\n{}", e)
        assert not bool(int(os.environ.get("PYBUDA_ASSERT_UNSUPPORTED_HW_OP", "0")))
        return CompileDepth.FULL # should be FATAL_ERROR or smth like that

    context.placer_solution = context.balancer_solution.placer_solution
    context.output_kwargs["placer_solution"] = context.placer_solution
    context.output_kwargs["output_host_tms"] = context.balancer_solution.output_host_tms

    if had_balancer_attempts:
        dump_graph(context.lowered_graph, context.graph_name, "post_balancer_error")
        if context.compiler_cfg.enable_training:
            calculate_grads(context.outputs, context.dev, context.intermediate_tensors, True, context.losses)

    # Golden back-end requires input and output queues to be large enough to store all inputs/outputs
    io_queue_multiplier = context.microbatch_count if context.dev.devtype == BackendType.Golden else 2  # double-buffer on silicon
    input_queue_multiplier = io_queue_multiplier
    output_queue_multiplier = io_queue_multiplier

    # For training, queues used across fwd & bwd must be large enough to store all microbatches
    if context.compiler_cfg.enable_training and context.microbatch_count > 1:
        input_queue_multiplier = max(input_queue_multiplier, context.microbatch_count)
        # output_queue_multiplier = max(output_queue_multiplier, microbatch_count)

    cross_chip_buffering = context.dev.arch == BackendDevice.Grayskull or bool(
        int(os.environ.get("PYBUDA_WORMHOLE_PIPELINED_PLACER", "0"))
    )

    context.post_placer_config = PostPlacerConfig(
            device_config=context.device_cfg,
            input_queue_multiplier=input_queue_multiplier,
            output_queue_multiplier=output_queue_multiplier,
            microbatch_size=context.microbatch_size,
            microbatch_count=context.microbatch_count,
            enable_t_streaming=balancer_config.enable_t_streaming,
            input_queues_on_host=balancer_config.input_queues_on_host,
            output_queues_on_host=balancer_config.output_queues_on_host,
            manual_dram_queue_placement=context.compiler_cfg.manual_dram_queue_placement,
            fork_join_tiles_treshold=balancer_config.fork_join_tiles_treshold,
            enable_cross_chip_buffering=cross_chip_buffering,
            placement_algorithm=context.compiler_cfg.dram_placement_algorithm)

    allocated_blocks = context.dev.allocated_blocks
    current_host_address = context.dev.current_host_address
    context.post_placer_results = run_post_placer_buda_passes(context.lowered_graph, context.graph_name, context.device_cfg, context.placer_solution, context.post_placer_config, context.balancer_solution, instructions, allocated_blocks, current_host_address)
    dump_graph(context.lowered_graph, context.graph_name, "post_placer", context.placer_solution, context.balancer_solution)

    # placer_done = len(post_placer_results.ins_instructions) == len(instructions) # no new instructions
    placer_done, _, _ = is_subset_of_instructions(context.post_placer_results.ins_instructions, instructions)

    if not placer_done:
        context.placer_retry_count += 1
        logger.debug(f"Previous  instructions: {len(instructions)}, new instructions: {len(context.post_placer_results.ins_instructions)}")
        logger.info(f"Placer failed, retrying loop count {context.placer_retry_count}")
        assert context.placer_retry_count < 20, " 20 loops of placer failed - aborting compile"
        return CompileDepth.BUDA_GRAPH_PRE_PLACER

    return CompileDepth.PRE_NETLIST_PASS

def run_pre_netlist_pass(context: CompileContext) -> CompileDepth:
    """
    Runs pre netlist passes.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    device_cfg = context.device_cfg
    verify_cfg = context.verify_cfg
    dev = context.dev
    graph_name = context.graph_name
    lowered_graph = context.lowered_graph
    balancer_solution = context.balancer_solution
    placer_solution = context.placer_solution
    post_placer_config = context.post_placer_config
    post_placer_results = context.post_placer_results
    inputs = context.inputs
    parameter_dict = context.parameter_dict

    if bool(int(os.environ.get("PYBUDA_REPRODUCE_SUBGRAPH", "0"))):
        intermediates = get_intermediate_tensors(lowered_graph, inputs, parameter_dict, dev, True)
        assert len(context.outputs) == 1, "Only single output supported for cut_graph"
        golden_output = intermediates[os.environ.get("PYBUDA_REPRODUCE_SUBGRAPH_OUTPUT")]
        verify_cfg.override_module_outptus = [golden_output]
    else:
        intermediates = {}

    run_pre_netlist_generation_buda_passes(lowered_graph, graph_name, device_cfg, intermediates, placer_solution, post_placer_config, balancer_solution, post_placer_results.allocated_blocks, post_placer_results.current_host_address)
    dump_graph(lowered_graph, graph_name, "pre_netlist")

    context.output_kwargs["consteval_trace"] = pygraph.record_consteval_operations(lowered_graph)

    if compiler_cfg.enable_training:
        calculate_grads(context.outputs, dev, context.intermediate_tensors, True, context.losses)

    return CompileDepth.GENERATE_NETLIST

def generate_netlist(context: CompileContext) -> CompileDepth:
    """
    Generates netlist.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    compiler_cfg = context.compiler_cfg
    device_cfg = context.device_cfg
    verify_cfg = context.verify_cfg
    dev = context.dev
    graph_name = context.graph_name
    lowered_graph = context.lowered_graph
    balancer_solution = context.balancer_solution
    placer_solution = context.placer_solution
    post_placer_results = context.post_placer_results
    backend_output_directory = context.backend_output_directory

    logger.info("Generating Netlist")
    net : BudaNetlist = lower_to_buda_netlist(lowered_graph, graph_name, placer_solution, balancer_solution, device_cfg.chip_ids, device_cfg, compiler_cfg.enable_forked_dram_inputs)
    dev.compiled_netlists.append(net)

    dump_epoch_type_graphs(lowered_graph, graph_name, "post_placer", placer_solution, balancer_solution)
    dump_epoch_id_graphs(lowered_graph, graph_name, "post_placer", placer_solution, balancer_solution)

    context.netlist_filename = ci.write_netlist_and_buda_envs_config(net, graph_name, backend_output_directory)

    netlist_override = os.environ.get("PYBUDA_NETLIST_OVERRIDE", None)
    if netlist_override is not None:
        logger.info("PYBUDA_NETLIST_OVERRIDE={}", netlist_override)
        context.netlist_filename = netlist_override

    postfix = os.environ.get("PYBUDA_REPORTIFY_POSTFIX", "")
    if len(postfix) > 0:
        postfix = "." + postfix
    net2placement(graph_name + postfix, context.netlist_filename, device_yaml=device_cfg.device_yaml)
    if "PYBUDA_GENERATE_OVERRIDE_CONFIG" in os.environ:
        generate_override_config(lowered_graph, balancer_solution, placer_solution, post_placer_results.nop_instructions, graph_name)

    if verify_cfg.run_net2pipe or bool(int(os.environ.get("PYBUDA_VERIFY_NET2PIPE", "0"))):
        logger.info("Verifying net2pipe.")
        ok, error = verify_net2pipe(context.netlist_filename, device_cfg.device_yaml, device_cfg.cluster_config_yaml)
        if not ok:
            logger.error("net2pipe failed: {}", error)
            is_error_handled = handle_backend_error(context, None)
            assert is_error_handled, "Net2Pipe verification failed"

            return context.stage

        logger.info("net2pipe completed successfully!")

    return CompileDepth.BACKEND_GOLDEN_VERIFY

def run_backend_golden_verify(context: CompileContext) -> CompileDepth:
    """
    Runs backend golden verify.

    Parameters
    ----------
    context: CompileContext
        Compile context

    Returns
    -------
    CompileDepth - next compile stage
    """
    verify_cfg = context.verify_cfg
    compiler_cfg = context.compiler_cfg
    dev = context.dev

    compile_results = generate_compile_results(
        verify_cfg,
        context.initial_graph_copy, context.outputs,
        context.intermediate_tensors,
        context.lowered_graph,
        context.netlist_filename,
        context.post_placer_results.perf_model_results,
        pass_specific_output_kwargs = context.output_kwargs
    )

    # Verify on backend golden
    if verify_cfg.run_golden:
        verify_golden(context.netlist_filename, compiler_cfg.enable_training, compile_results, dev, context.inputs, context.outputs, verify_cfg)

    return CompileDepth.FULL

def handle_backend_error(context: CompileContext, ex: Optional[BackendCompileException]) -> bool:
    """
    If pybuda recompile is enabled, tries to handle error raised by the backend.

    Parameters
    ----------
    context: CompileContext
        Compile context

    e: Exception
        Exception

    Returns
    -------
    bool - True if the error was handled and we should recompile, false otherwise
    """

    assert context is not None
    recompile_enabled = bool(int(os.environ.get("PYBUDA_AUTO_RECOMPILE", "1")))
    recompile_retry_limit = int(os.environ.get("PYBUDA_AUTO_RECOMPILE_RETRY_LIMIT", "10"))

    if recompile_enabled and context.recompile_count < recompile_retry_limit:
        context.in_recompile = True
        context.recompile_count += 1
        logger.warning("Compile failed, retrying compilation with different parameters. Retry count: {}", context.recompile_count)

        if ex is not None and ex.compile_result.failure_type == BackendCompileFailure.OverlaySize:
            # Add extra overlay blob size.
            overlay_max_extra_blob_size = int(os.environ.get("TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE", "0"))
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = str(overlay_max_extra_blob_size + ex.compile_result.extra_size_bytes)
            logger.warning("Setting TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE to {}", os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"])
        elif bool(int(os.environ.get("PYBUDA_AUTO_RECOMPILE_TARGET_CYCLES", "0"))):
            # Currently only NLP and Ribbon policies are supported for recompilation.
            # Because the only handling we do is to change the target cycles and recompile - which other policies don't use.
            if context.policy_type not in [pybalancer.PolicyType.NLP, pybalancer.PolicyType.Ribbon]:
                return False

            # Offset target cycles for the recompile.
            context.target_cycles_offset += int(os.environ.get("PYBUDA_TARGET_CYCLES_OFFSET", "50000"))
            logger.warning("Setting PYBUDA_TARGET_CYCLES_OFFSET to {}", os.environ["PYBUDA_TARGET_CYCLES_OFFSET"])
        else:
            return False

        # Set the compile context to execute from pre placer stage.
        context.stage = CompileDepth.BUDA_GRAPH_PRE_PLACER
        return True

    return False
