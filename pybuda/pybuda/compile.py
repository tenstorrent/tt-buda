# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional, List, Dict, Any, Tuple, Union

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
)
import pybuda
from .parameter import Parameter
from pybuda._C.backend_api import BackendType, BackendDevice
import pybuda._C.autograd as pyautograd
import pybuda._C.balancer as pybalancer
import pybuda._C.pattern_matcher as pypattern_matcher
import pybuda._C.scheduler as pyscheduler
from pybuda._C.placer import match_op_names_to_placer_overrides
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

    if verify_cfg is None:
        verify_cfg = VerifyConfig.disabled() # no verification config provided, disable by default
    
    if compiler_cfg is None:
        compiler_cfg = _get_global_compiler_config()

    force_full = bool(int(os.environ.get("PYBUDA_FORCE_FULL_COMPILE_DEPTH", "0")))
    if force_full:
        compiler_cfg.compile_depth = CompileDepth.FULL

    if len(targets) > 0:
        assert dev.loss_module is not None, "Target provided for compilation, but this device has no loss module"

    if dev.loss_module is not None:
        assert len(targets) > 0, f"Device {dev} has a loss module, but no targets were provided for compilation"

    backend_output_directory = compiler_cfg.backend_output_dir
    ci.initialize_output_build_directory(backend_output_directory)

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
        dump_compiler_cfg(backend_output_directory, compiler_cfg, graph_name)

    init_log_last_successful_compile_stage()

    should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.START_COMPILE)
    if should_early_stop_compilation:
        return generate_compile_results(
            verify_cfg, 
        )

    logger.info("Generating initial graph")
    should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.GENERATE_INITIAL_GRAPH)

    if compiler_cfg.compile_tvm_to_python and dev.graph is None:
        module_inputs = inputs
        for index, module in enumerate(dev.modules):
            if not isinstance(module, PyBudaModule):
                from .tvm_to_python import generate_pybuda_module
                prev_state = state_changed()
                modules, dev_types, module_inputs = generate_pybuda_module(module, to_pt_tensors(module_inputs), compiler_cfg, module.name, verify_cfg,)
                assert len(modules) == 1, "Attemping to load split model onto single devices"
                dev.modules[index] = modules[0]

                if index == 0:
                    inputs = module_inputs

                if not(prev_state):
                    clear_state_changed()

            if index < len(dev.modules) - 1 and not compiler_cfg.compile_subgraphs:
                if module is dev.loss_module:
                    if len(module_inputs) == 1:
                        module_inputs = dev.modules[index].forward(module_inputs[0], targets[0])
                    else:
                        module_inputs = dev.modules[index].forward(tuple(module_inputs), tuple(targets))
                else:
                    module_inputs = dev.modules[index].forward(*module_inputs)

                if isinstance(module_inputs, Tensor):
                    module_inputs = (module_inputs,) # Force a tuple

    if dev.graph is None:
        graph, outputs, intermediate_tensors, inputs, _ = dev.generate_graph(*inputs, return_intermediate=verify_cfg.intermediates, graph_name=graph_name, compiler_cfg=compiler_cfg, target_tensors=targets, verify_cfg=verify_cfg)
    else:
        graph = dev.graph
        intermediate_tensors = dev.intermediate_tensors
        outputs = dev.output_tensors

    graph.set_microbatch(microbatch_size)
    dump_graph(graph, graph_name, "initial_graph")
    validate_override_names(graph, compiler_cfg)
    if compiler_cfg.enable_link_past_cache_ios:
        # move index ops to weights if applicable
        move_index_to_mm_weights(graph)

        # link past cache ios will change the number on inputs / outputs, so it is called bfore we clone the initial graph
        new_params = link_past_cache_ios(graph)
        inputs_to_remove = []
        for k, v in new_params.items():
            module.add_parameter(k, Parameter(inputs[v].value(), requires_grad=False, name=k))
            inputs_to_remove.append(inputs[v])
        for i in inputs_to_remove:
            inputs.remove(i)

    initial_graph_copy = graph.clone() # save the original graph for verification and analysis
    input_grads = []

    pass_specific_output_kwargs = {}
    parameter_dict = {p.get_name() : p.value(is_buda=False) for p in dev.get_parameters()}
    
    if verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation):
        do_verify("initial_graph", False, graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, False, targets=targets)

    if should_early_stop_compilation:
        return generate_compile_results(
            verify_cfg, 
            initial_graph_copy, outputs, 
            intermediate_tensors,
        )

    logger.info("Running post initial graph pass")
    should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.POST_INITIAL_GRAPH_PASS)

    inserted_node_id_mapping, fracture_chip_id_assignments = run_post_initial_graph_passes(graph, compiler_cfg, compiler_cfg.fracture_groups)

    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        # If we have multi-level of decomposition, some node id might not in the original 
        # intermediate tensor dict. 
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    dump_graph(graph, graph_name, "decomposed_graph")
    if verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation):
        do_verify("decomposed_graph", False, graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, False, targets=targets)

    if compiler_cfg.enable_consteval:
        run_consteval_graph_pass(graph)
        dump_graph(graph, graph_name, "consteval_graph")
        if verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation):
            do_verify("consteval_graph", False, graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, False, targets=targets)

    if compiler_cfg.match_subgraph_patterns:
        graph, match_result = pypattern_matcher.lower_pybuda_to_pattern_matcher(graph, compiler_cfg.match_subgraph_patterns)
        pass_specific_output_kwargs["match_result"] = match_result

        if match_result.is_subgraph_loopable:
            dump_graph(graph, graph_name, "looped_graph")

        if check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.POST_PATTERN_MATCHER):
            return generate_compile_results(
                pass_specific_output_kwargs = pass_specific_output_kwargs 
            )

    run_optimization_graph_passes(graph, device_cfg)
    dump_graph(graph, graph_name, "optimized_graph")
    inserted_node_id_mapping = run_post_optimize_decompose_graph_passes(graph, compiler_cfg)
    dump_graph(graph, graph_name, "decomposed_optimized_graph")
    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    if verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation):
        do_verify("optimized_graph", False, graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, False, targets=targets)

    # Workaround for TVM and lack of parameters at the time optimizer is created
    if dev.optimizer:
        if dev.optimizer.device_params:
            dev.optimizer.set_parameters_to_optimize(dev.modules[0].get_parameters())
        dev.optimizer.set_optimizer_parameters()

    if compiler_cfg.enable_training:

        autograd_config = pyautograd.AutogradConfig(recompute=compiler_cfg.enable_recompute, optimizer=dev.optimizer)
        autograd_engine = pyautograd.AutogradEngine(graph, autograd_config)
        
        graph = autograd_engine.run()
        dump_graph(graph, graph_name, "post_autograd")
        
        if verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation):
            losses = do_verify("post_autograd", compiler_cfg.enable_training, graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, False, losses, targets=targets)
        elif compiler_cfg.enable_training:
            losses = calculate_grads(outputs, dev, intermediate_tensors, False, losses)

        # Record calculated input grads from the previous do_verify call and save so that we don't keep re-calculating and
        # accumulating on each verification call
        input_grads = [i.value().grad for i in inputs if i.value().requires_grad and i.value().grad is not None]

        # run_post_autograd_graph_passes(graph)
        # dump_graph(graph, graph_name, "post_autograd_passes")
        
        # do_verify("post_autograd_passes", compiler_cfg.enable_training, graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, False, losses)

    logger.info("Running post autograd graph pass")
    inserted_node_id_mapping = run_post_autograd_graph_passes(graph, compiler_cfg)
    for inserted_node_id, original_node_id in inserted_node_id_mapping:
        if original_node_id in intermediate_tensors:
            intermediate_tensors[inserted_node_id] = intermediate_tensors[original_node_id]

    dump_graph(graph, graph_name, "post_autograd_passes")

    if verify_cfg.verify_all or verify_cfg.verify_post_autograd_passes or (verify_cfg.verify_last and should_early_stop_compilation):
        do_verify("post_autograd_passes", compiler_cfg.enable_training, graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, False, losses, targets=targets)
    elif compiler_cfg.enable_training:
        calculate_grads(outputs, dev, intermediate_tensors, False, losses)
        
    input_grads = [i.value().grad for i in inputs if i.value().requires_grad and i.value().grad is not None]

    if should_early_stop_compilation:
        return generate_compile_results(
            verify_cfg, 
            initial_graph_copy, outputs, 
            intermediate_tensors,
        )

    run_pre_lowering_passes(graph)
    dump_graph(graph, graph_name, "pre_lowering")

    should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.PRE_LOWERING_PASS)
    if should_early_stop_compilation:
        return generate_compile_results(
            verify_cfg, 
            initial_graph_copy, outputs, 
            intermediate_tensors,
        )

    logger.info("Lowering to Buda")
    for parameter in dev.get_parameters():
        parameter._set_fp32_fallback(dev.fp32_fallback)

    scheduler_config = pyscheduler.SchedulerConfig(
        scheduler_policy=pyscheduler.policy_from_string(os.environ.get("PYBUDA_SCHEDULER_POLICY", compiler_cfg.scheduler_policy)),
        scheduler_constraints=compiler_cfg.scheduler_constraints,
    )

    should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.BUDA_GRAPH_PRE_PLACER)

    post_placer_results = None
    placer_done = False
    placer_loop_count = 0

    policy_type = pybalancer.policy_from_string(os.environ.get("PYBUDA_BALANCER_POLICY_TYPE", compiler_cfg.balancer_policy))
    use_interactive_placer = (
        compiler_cfg.use_interactive_placer and
        not (bool(int(os.environ.get("PYBUDA_DISABLE_INTERACTIVE_PLACER", "0")))) and
        pybalancer.can_use_interactive_placer(policy_type)
    )
    while not placer_done:
        instructions = {} if post_placer_results is None else post_placer_results.ins_instructions
        temp_dict = {}; temp_dict.update(compiler_cfg.buffering_nops_to_insert); temp_dict.update(instructions)
        lowered_graph, placer_config_update = run_pre_placer_buda_passes(
                graph,
                scheduler_config,
                device_cfg, 
                device_cfg.chip_ids,
                list(map(placer_breaks_eval, compiler_cfg.op_names_to_chip_break)),
                list(map(placer_breaks_eval, compiler_cfg.op_names_to_epoch_break)),
                compiler_cfg.op_names_dont_fuse,
                compiler_cfg.op_names_manual_fuse,
                fracture_chip_id_assignments,
                compiler_cfg.default_df_override, 
                compiler_cfg.default_accumulate_df, 
                compiler_cfg.enable_broadcast_splitting or bool(int(os.environ.get("PYBUDA_ENABLE_BROADCAST_SPLITTING", "0"))),
                dev.fp32_fallback,
                compiler_cfg.default_math_fidelity,
                compiler_cfg.enable_auto_fusing,
                compiler_cfg.amp_level or int(os.environ.get("PYBUDA_AMP_LEVEL", "0")),
                compiler_cfg.enable_recompute,
                (bool(int(os.environ.get("PYBUDA_ENABLE_OUTPUT_QUEUES_ON_HOST", "1"))) and compiler_cfg.output_queues_on_host),
                temp_dict,
                compiler_cfg.insert_queues,
                compiler_cfg.amp_properties,
                compiler_cfg.op_intermediates_to_save,
                use_interactive_placer,
                compiler_cfg.enable_device_tilize)
        dump_graph(lowered_graph, graph_name, "pre_placer")

        # Convert to buda tensors - i.e. 4d / tile-snapped dims
        def to_buda_shapes(tensors):
            if tensors is None or not tensors:
                return tensors

            if isinstance(tensors[0], torch.Tensor):
                return [pad_pytorch_tensor_to_buda(t, lowered_graph.get_tile_broadcast_dims_for_bw_input(i)) for i, t in enumerate(tensors)]

            return [t.to_buda_shape(tile_broadcast_dims=lowered_graph.get_tile_broadcast_dims_for_target(i)) for i, t in enumerate(tensors)]

        #buda_intermediate_tensors = {}
        #for k, v in intermediate_tensors.items(): 
        #    buda_intermediate_tensors[k] = pad_pytorch_tensor_to_buda(v)
        buda_losses = to_buda_shapes(losses)
        buda_targets = to_buda_shapes(targets)

        if verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation):
            do_verify("pre_placer", compiler_cfg.enable_training, lowered_graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, True, buda_losses, targets=buda_targets)
        elif compiler_cfg.enable_training:
            calculate_grads(outputs, dev, intermediate_tensors, True, losses)

        if should_early_stop_compilation:
            return generate_compile_results(
                verify_cfg, 
                initial_graph_copy, outputs, 
                intermediate_tensors,
                lowered_graph,
            )

        op_name_to_placer_overrides = match_op_names_to_placer_overrides(lowered_graph, list(map(placer_op_overrides_eval, compiler_cfg.placer_op_overrides)))
        balancer_config = pybalancer.BalancerConfig(
            device_config=device_cfg,
            scheduler_config=scheduler_config,
            policy_type=policy_type,
            random_policy_seed=int(os.environ.get("PYBUDA_BALANCER_RANDOM_POLICY_SEED", 0)),
            chip_ids=device_cfg.chip_ids,
            chip_placement_policy=pybalancer.chip_placement_policy_from_string(compiler_cfg.chip_placement_policy),
            enable_t_streaming = (bool(int(os.environ.get("PYBUDA_ENABLE_T_STREAMING", "0"))) or compiler_cfg.enable_t_streaming),
            manual_t_streaming = compiler_cfg.manual_t_streaming,
            skip_l1_usage_validation=bool(int(os.environ.get("PYBUDA_SKIP_L1_USAGE_VALIDATION", "0"))),
            input_queues_on_host=compiler_cfg.input_queues_on_host,
            output_queues_on_host=compiler_cfg.output_queues_on_host,
            default_dram_parameters=(not verify_cfg.enabled and (microbatch_size == 1)) if compiler_cfg.default_dram_parameters is None else compiler_cfg.default_dram_parameters,
            op_names_to_epoch_break=placer_config_update.op_names_to_epoch_break,
            op_names_to_chip_break=placer_config_update.op_names_to_chip_break,
            op_overrides=compiler_cfg.balancer_op_overrides,
            op_names_to_chip_id_assignment=placer_config_update.op_to_chip_id_assignment,
            op_name_to_placer_overrides=op_name_to_placer_overrides,
            enable_auto_transposing_placement = compiler_cfg.enable_auto_transposing_placement,
            graph_solver_self_cut_type = pybalancer.graph_solver_self_cut_type_from_string(os.environ.get("PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE", compiler_cfg.graph_solver_self_cut_type)),
            use_interactive_placer = use_interactive_placer,
            enable_enumerate_u_kt = compiler_cfg.enable_enumerate_u_kt,
            enable_single_buffer_fallback = compiler_cfg.enable_single_buffer_fallback,
        )
        should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.BALANCER_PASS)
        try:
            balancer_solution, had_balancer_attempts = run_placer_buda_passes(lowered_graph, balancer_config, fracture_chip_id_assignments, compiler_cfg.paddings)
        except UnsupportedHWOpsError as e:
            logger.warning("Found unsupported HW ops, stopping compilation early:\n{}", e)
            assert not bool(int(os.environ.get("PYBUDA_ASSERT_UNSUPPORTED_HW_OP", "0")))
            return generate_compile_results(
                verify_cfg, 
                initial_graph_copy, outputs, 
                intermediate_tensors,
                lowered_graph,
            )

        placer_solution = balancer_solution.placer_solution
        pass_specific_output_kwargs["placer_solution"] = placer_solution
        pass_specific_output_kwargs["output_host_tms"] = balancer_solution.output_host_tms

        if had_balancer_attempts:
            dump_graph(lowered_graph, graph_name, "post_balancer_error")
            if verify_cfg.verify_all or (verify_cfg.verify_last and should_early_stop_compilation):
                pass
                #do_verify("post_balancer_error", compiler_cfg.enable_training, lowered_graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, True, buda_losses, targets=buda_targets)
            elif compiler_cfg.enable_training:
                calculate_grads(outputs, dev, intermediate_tensors, True, losses)

        # Golden back-end requires input and output queues to be large enough to store all inputs/outputs
        io_queue_multiplier = microbatch_count if dev.devtype == BackendType.Golden else 2  # double-buffer on silicon
        input_queue_multiplier = io_queue_multiplier
        output_queue_multiplier = io_queue_multiplier

        # For training, queues used across fwd & bwd must be large enough to store all microbatches
        if compiler_cfg.enable_training and microbatch_count > 1:
            input_queue_multiplier = max(input_queue_multiplier, microbatch_count)
            # output_queue_multiplier = max(output_queue_multiplier, microbatch_count)

        cross_chip_buffering = dev.arch == BackendDevice.Grayskull or bool(
            int(os.environ.get("PYBUDA_WORMHOLE_PIPELINED_PLACER", "0"))
        )

        post_placer_config = PostPlacerConfig(
                device_config=device_cfg,
                input_queue_multiplier=input_queue_multiplier,
                output_queue_multiplier=output_queue_multiplier,
                microbatch_size=microbatch_size,
                microbatch_count=microbatch_count,
                enable_t_streaming=balancer_config.enable_t_streaming,
                input_queues_on_host=balancer_config.input_queues_on_host,
                output_queues_on_host=balancer_config.output_queues_on_host,
                manual_dram_queue_placement=compiler_cfg.manual_dram_queue_placement,
                fork_join_tiles_treshold=balancer_config.fork_join_tiles_treshold,
                enable_cross_chip_buffering=cross_chip_buffering,
                placement_algorithm=compiler_cfg.dram_placement_algorithm)

        allocated_blocks = dev.allocated_blocks
        current_host_address = dev.current_host_address
        post_placer_results = run_post_placer_buda_passes(lowered_graph, graph_name, device_cfg, placer_solution, post_placer_config, balancer_solution, instructions, allocated_blocks, current_host_address)
        dump_graph(lowered_graph, graph_name, "post_placer", placer_solution, balancer_solution)

        # placer_done = len(post_placer_results.ins_instructions) == len(instructions) # no new instructions
        placer_done, _, _ = is_subset_of_instructions(post_placer_results.ins_instructions, instructions)

        if not placer_done:
            placer_loop_count += 1
            logger.debug(f"Previous  instructions: {len(instructions)}, new instructions: {len(post_placer_results.ins_instructions)}")
            logger.info(f"Placer failed, retrying loop count {placer_loop_count}")
            assert placer_loop_count < 20, " 20 loops of placer failed - aborting compile"

    if bool(int(os.environ.get("PYBUDA_REPRODUCE_SUBGRAPH", "0"))):
        intermediates = get_intermediate_tensors(lowered_graph, inputs, parameter_dict, dev, True)
        assert len(outputs) == 1, "Only single output supported for cut_graph"
        golden_output = intermediates[os.environ.get("PYBUDA_REPRODUCE_SUBGRAPH_OUTPUT")]
        verify_cfg.override_module_outptus = [golden_output]
    else:
        intermediates = {}
    run_pre_netlist_generation_buda_passes(lowered_graph, graph_name, device_cfg, intermediates, placer_solution, post_placer_config, balancer_solution, post_placer_results.allocated_blocks, post_placer_results.current_host_address)
    dump_graph(lowered_graph, graph_name, "pre_netlist")

    verify_cfg.dump_tensors_path = ci.get_netlist_dir() if ci.capture_tensors() else ""
    if verify_cfg.verify_all or verify_cfg.verify_post_placer or (verify_cfg.verify_last and should_early_stop_compilation) or verify_cfg.dump_tensors_path:
        do_verify("post_placer", compiler_cfg.enable_training, lowered_graph, inputs, parameter_dict, input_grads, outputs, dev, intermediate_tensors, verify_cfg, True, buda_losses, balancer_solution=balancer_solution, targets=buda_targets)
    elif compiler_cfg.enable_training:
        calculate_grads(outputs, dev, intermediate_tensors, True, losses)

    if should_early_stop_compilation:
        return generate_compile_results(
            verify_cfg,
            initial_graph_copy, outputs,
            intermediate_tensors,
            lowered_graph,
            pass_specific_output_kwargs = pass_specific_output_kwargs 
        )

    pass_specific_output_kwargs["consteval_trace"] = pygraph.record_consteval_operations(lowered_graph)

    logger.info("Generating Netlist")
    net : BudaNetlist = lower_to_buda_netlist(lowered_graph, graph_name, placer_solution, balancer_solution, device_cfg.chip_ids, device_cfg, compiler_cfg.enable_forked_dram_inputs)
    dev.compiled_netlists.append(net)

    dump_epoch_type_graphs(lowered_graph, graph_name, "post_placer", placer_solution, balancer_solution)
    dump_epoch_id_graphs(lowered_graph, graph_name, "post_placer", placer_solution, balancer_solution)

    netlist_filename = ci.write_netlist_and_buda_envs_config(net, graph_name, backend_output_directory)

    netlist_override = os.environ.get("PYBUDA_NETLIST_OVERRIDE", None)
    if netlist_override is not None:
        logger.info("PYBUDA_NETLIST_OVERRIDE={}", netlist_override)
        netlist_filename = netlist_override

    postfix = os.environ.get("PYBUDA_REPORTIFY_POSTFIX", "")
    if len(postfix) > 0:
        postfix = "." + postfix
    net2placement(graph_name + postfix, netlist_filename, device_yaml=device_cfg.device_yaml)
    if "PYBUDA_GENERATE_OVERRIDE_CONFIG" in os.environ:
        generate_override_config(lowered_graph, balancer_solution, placer_solution, post_placer_results.nop_instructions, graph_name)

    if verify_cfg.run_net2pipe or bool(int(os.environ.get("PYBUDA_VERIFY_NET2PIPE", "0"))):
        verify_net2pipe(netlist_filename, device_cfg.device_yaml, device_cfg.cluster_config_yaml)

    should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.GENERATE_NETLIST)
    if should_early_stop_compilation:
        return generate_compile_results(
            verify_cfg, 
            initial_graph_copy, outputs, 
            intermediate_tensors,
            lowered_graph,
            netlist_filename,
            post_placer_results.perf_model_results,
        )

    compile_results = generate_compile_results(
        verify_cfg, 
        initial_graph_copy, outputs, 
        intermediate_tensors,
        lowered_graph,
        netlist_filename,
        post_placer_results.perf_model_results,
        pass_specific_output_kwargs = pass_specific_output_kwargs 
    )

    # Verify on backend golden
    if verify_cfg.run_golden:
        verify_golden(netlist_filename, compiler_cfg.enable_training, compile_results, dev, inputs, outputs, verify_cfg)
        
    should_early_stop_compilation = check_for_compilation_early_stop(compiler_cfg.compile_depth, CompileDepth.BACKEND_GOLDEN_VERIFY)
    if should_early_stop_compilation:
        return compile_results

    return compile_results


def check_for_compilation_early_stop(desired_depth, current_depth):
    """
    Determines should current compilation process stop or not based on desired
    and current phase of execution.

    Parameters
    ----------
    desired_depth: CompileDepth
        Desired phase for compiler early stopping.

    current_depth: CompileDepth
        Current phase for compiler early stopping.

    Returns
    -------
    Boolean
    """
    # update global compile stage variable
    global LAST_SUCCESSFUL_STAGE
    LAST_SUCCESSFUL_STAGE = str(CompileDepth(current_depth.value-1).name)

    if not CompileDepth.has_value(desired_depth.value):
        raise Exception("Invalid compilation depth flag: {}".format(desired_depth.name))

    if desired_depth == current_depth:
        logger.info("Compilation early stopping after {}".format(current_depth.name))

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
