// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "buda_passes.hpp"

#include <algorithm>
#include <map>

#include "backend_api/device_config.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/query.hpp"
#include "graph_lib/utils.hpp"
#include "passes/bind_reshape_to_io.hpp"
#include "passes/constant_folding.hpp"
#include "passes/dataformat.hpp"
#include "passes/decomposing_context.hpp"
#include "passes/erase_consecutive_reshape.hpp"
#include "passes/erase_inverse_ops.hpp"
#include "passes/erase_unnecessary_4d_tm_sequence.hpp"
#include "passes/explicate_unsqueeze.hpp"
#include "passes/fork_join.hpp"
#include "passes/fuse_conv2d_bias.hpp"
#include "passes/fuse_ops.hpp"
#include "passes/fuse_pad_conv2d.hpp"
#include "passes/fuse_per_channel_ops.hpp"
#include "passes/fuse_redundant_tm_sequence.hpp"
#include "passes/fuse_reshape_transpose_into_slice.hpp"
#include "passes/generate_initial_flops_estimate.hpp"
#include "passes/hoist_transforms_to_inputs.hpp"
#include "passes/insert_inverse_on_io.hpp"
#include "passes/limit_to_4d_reshape.hpp"
#include "passes/link_past_cache_ios.hpp"
#include "passes/lower_concat_to_runtime_transform.hpp"
#include "passes/lower_reinterpret_shape.hpp"
#include "passes/lowering_context.hpp"
#include "passes/move_requantize.hpp"
#include "passes/move_select_after_matmul_optional.hpp"
#include "passes/pad_output_buffer.hpp"
#include "passes/passes_utils.hpp"
#include "passes/post_autograd_graph_passes.hpp"
#include "passes/pre_lowering_passes.hpp"
#include "passes/pre_placer_buda_passes.hpp"
#include "passes/print_graph.hpp"
#include "passes/replace_incommutable_patterns.hpp"
#include "passes/reproduce_subgraph.hpp"
#include "passes/set_tile_dim.hpp"
#include "passes/squeeze_to_reshape.hpp"
#include "passes/t_stream.hpp"
#include "perf_model/perf_model.hpp"
#include "placer/dram.hpp"
#include "placer/dram_allocator.hpp"
#include "placer/host_memory_allocator.hpp"
#include "placer/lower_to_placer.hpp"
#include "placer/utils.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"

namespace tt {

using NodeType = graphlib::NodeType;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;
using NodeId = graphlib::NodeId;
using PortId = graphlib::PortId;

void lower_reshape(Graph *, graphlib::OpNode *node)
{
    graphlib::OpType op_type = node->op_type();
    TT_ASSERT(op_type.attr.size() == 4);
    op_type.buda_attrs = {
        {"w", std::get<int>(op_type.attr[0])},
        {"z", std::get<int>(op_type.attr[1])},
        {"r", std::get<int>(op_type.attr[2])},
        {"c", std::get<int>(op_type.attr[3])},
    };
    node->change_op_type(op_type);
}

// *****************************************************************
//  ************************** Main APIs **************************
// *****************************************************************

// ********** Run post initial graph passes **********
std::tuple<std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>>, passes::FractureChipIdAssignments>
run_post_initial_graph_passes(graphlib::Graph *graph, py::object compiler_cfg_object, passes::FractureGroups const &fracture_groups)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "INITIAL");
    passes::generate_initial_flops_estimate(graph);
    passes::decompose_nd_reshape_split(graph);
    passes::limit_to_4d_reshape(graph);
    passes::erase_unnecessary_4d_tm_sequence(graph);
    passes::fuse_pad_conv2d(graph);
    passes::explicate_unsqueeze(graph);
    passes::fuse_conv2d_bias(graph);

    auto inserted_node_id_mapping = decompose_pybuda_graph(graph, "get_f_pybuda_decompose", compiler_cfg);
    auto chip_id_assignments = passes::fracture(graph, fracture_groups);
    return std::make_tuple(inserted_node_id_mapping, chip_id_assignments);
}

void run_optimization_graph_passes(graphlib::Graph *graph, const DeviceConfig &device_config)
{
    passes::print_graph(graph, "PRE OPTIMIZE");
    passes::lower_concat_to_runtime_transform(graph);

    passes::bypass_nop_tms(graph);
    // Fuses reshape and transpose pairs into slice or stack ops. More precisely, core ideas is
    // to fuse valid:
    //   - reshape + transpose => hslice
    //   - transpose + reshape => hstack
    //   - reshape => vslice
    //   - reshape => vstack
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);
    recalculate_shapes(graph);

    // Erase all inverse ops possible. 
    // Then, if no inverse ops are erased, then attempt to insert inverse ops on the output. 
    // Then, if no inverse ops can be inserted on the output, then attempt to insert inverse ops on the input.
    // Then, if no inverse ops can be inserted on the input, then attempt to bypass nop reshapes.
    //        NOTE: The reason we try this last is because nop reshapes may end up being inverses of other ops
    //              and we want to erase them that way first if possible
    // Commuting to input may have introduced clones, so attempt to erase inverse ops again
    // ...

    bool attempt_update = true;
    while (attempt_update)
    {
        passes::hoist_unsqueeze_squeeze_to_reshape(graph);

        bool skip_erase_redundant = false;
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update) {
            attempt_update = passes::insert_inverse_on_outputs(graph);
            if (attempt_update)
                skip_erase_redundant = true;
        }
        if (not attempt_update)
            attempt_update = passes::insert_inverse_on_inputs(graph);
        if (not attempt_update) {
            attempt_update = passes::insert_inverse_on_downstream_tms(graph);
            if (attempt_update)
                skip_erase_redundant = true;
        }
        if (not attempt_update)
            attempt_update = passes::replace_incommutable_patterns(graph);

        // These passes erase tms for non-inverse reasons. Usually we are fine with this.
        // However, we might insert tms on top or under of other tms for the purpose of erasing other inverse ops.
        // Skip in that case
        if (not skip_erase_redundant) {
            if (not attempt_update)
                attempt_update = passes::erase_consecutive_reshape(graph, true);
            if (not attempt_update)
                attempt_update = passes::fuse_tm_sequences(graph);
            passes::bypass_nop_tms(graph);
        }
    }
    passes::move_tm_through_requantize(graph);
    recalculate_shapes(graph);

    passes::hoist_transforms_to_inputs(graph);
    passes::erase_consecutive_reshape(graph, true);
    passes::pad_output_buffer(graph, device_config);
    passes::lower_reinterpret_shape(graph);
    passes::bind_reshape_to_io(graph);

    passes::fuse_per_channel_ops(graph);
    if (not env_as<bool>("PYBUDA_DISABLE_CONSTANT_FOLDING"))
        passes::constant_folding(graph);

    passes::move_select_after_matmul_optional(graph);

    passes::fuse_tm_sequences(graph);
    reportify::dump_graph(graph->name(), "post_erase_inverse_ops", graph);
}

std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_optimize_decompose_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "POST_OPTIMIZE");
    auto inserted_node_id_mapping = decompose_pybuda_graph(graph, "get_f_pybuda_decompose_post_optimize", compiler_cfg);

    return inserted_node_id_mapping;
}

// ********** Run post autograd graph passes **********
std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> run_post_autograd_graph_passes(
    graphlib::Graph *graph, py::object compiler_cfg_object)
{
    std::shared_ptr<void> compiler_cfg = make_shared_py_object(compiler_cfg_object);

    passes::print_graph(graph, "POST_AUTOGRAD");
    lower_bwd_gather_ops(graph);
    return decompose_pybuda_graph(graph, "get_f_pybuda_decompose_post_autograd", compiler_cfg);
}

// ********** Run pre-lowering passes **********
void run_pre_lowering_passes(graphlib::Graph *graph)
{
    passes::print_graph(graph, "PRE_LOWERING");
    // Recalculate shapes, and figure out implicit broadcasts that are missing
    recalculate_shapes(graph);

    // Fuse bias into matmuls
    fuse_bias(graph);

    // Fuse requantize into matmuls
    fuse_requantize(graph);
    
    // Fuse gelu into matmuls
    if (env_as<bool>("PYBUDA_FUSE_MATMUL_GELU")) {
        fuse_gelu(graph);
    }

    // Manually convert broadcast ops to tms, so insert tile broadcast ops can work generically
    // Note this is not lowering, these are still pybuda tms
    convert_broadcast_ops_to_tms(graph);

    // Insert tile broadcast ops
    insert_tile_broadcast_ops(graph);

    // Bypass embedding input nops
    bypass_embedding_input_nops(graph);

    // If there are any non-embedding users of the emdebbing table, it needs to be duplicated
    duplicate_embedding_table_if_needed(graph);

    // Fold tile broadcasts into reduce and inputs
    fold_tile_broadcast_ops_into_inputs(graph);
    fold_tile_broadcast_ops_into_reduce(graph);
}

// ********** Run lowering passes **********
std::pair<std::unique_ptr<graphlib::Graph>, placer::PlacerConfigUpdate> run_pre_placer_buda_passes(
    graphlib::Graph *graph,
    scheduler::SchedulerConfig scheduler_config,
    const DeviceConfig &device_config,
    std::vector<std::uint32_t> chip_ids,
    const placer::PredicatesToBreaks &predicates_to_chip_break,
    const placer::PredicatesToBreaks &predicates_to_epoch_break,
    const std::vector<std::string> &op_names_dont_fuse,
    const std::vector<std::string> &op_names_manual_fuse,
    const passes::FractureChipIdAssignments &fracture_chip_id_assignments,
    const std::optional<DataFormat> default_df_override,
    const std::optional<DataFormat> default_accumulate_df,
    const bool enable_broadcast_splitting,
    const DataFormat fp32_fallback,
    const MathFidelity default_math_fidelity,
    const bool enable_auto_fusing,
    const int amp_level,
    const bool enable_recompute,
    const bool output_queues_on_host,
    const bool input_queues_on_host,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &ins_instructions,
    const std::vector<std::tuple<std::string, std::string, int>> &insert_queues,
    std::vector<AMPNodeProperties> amp_properties,
    const std::vector<std::string> &op_intermediates_to_save,
    const bool use_interactive_placer,
    bool enable_device_tilize)
{
    log_debug(LogGraphCompiler, "Lowering target device\n{}", device_config);

    passes::print_graph(graph, "PRE_PLACER");

    // Create buda ops / tms
    std::unique_ptr<graphlib::Graph> lowered_graph = lower_to_buda_ops(graph);

    // lower user-defined buffering queues to actual queue types
    lower_to_buffering_queues(lowered_graph.get());

    split_unsupported_gradient_ops(lowered_graph.get(), device_config);
    recalculate_shapes(lowered_graph.get());

    // Remove nops
    remove_nops(lowered_graph.get());

    // Add buffer NOP between host input and ops if there are multiple ops reading from same host input.
    //
    if (input_queues_on_host and env_as<bool>("PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"))
    {
        fix_host_inputs(lowered_graph.get());
    }

    auto op_names_to_chip_break = placer::match_op_names_to_breaks(lowered_graph.get(), predicates_to_chip_break);
    auto op_names_to_epoch_break = placer::match_op_names_to_breaks(lowered_graph.get(), predicates_to_epoch_break);

    // Fuse ops
    if (enable_auto_fusing)
    {
        recalculate_shapes(lowered_graph.get());
        fuse_ops(
            lowered_graph.get(),
            device_config,
            op_names_to_chip_break,
            op_names_to_epoch_break,
            op_names_dont_fuse,
            op_names_manual_fuse,
            amp_properties);
    }

    // Sanitize past cache IOs
    sanitize_past_cache_ios(lowered_graph.get());

    // Remove transposes from srcB
    bool device_supports_tm_on_srcb = false;  // TODO: device descriptor
    if (!device_supports_tm_on_srcb)
        fix_transposes(lowered_graph.get(), device_config);

    // Remove TMs from output node
    fix_tms_on_output(lowered_graph.get());

    // Need to run before fixing ops that require untilize nop back to host
    insert_queues_for_op_intermediates(lowered_graph.get(), op_intermediates_to_save);

    // Add NOPs on ops feeding output that can't do it directly
    if (output_queues_on_host)
    {
        fix_untilized_outputs(lowered_graph.get(), device_config);
    }

    // Replace "buffer" placeholders with NOPs
    replace_buffers_with_nops(lowered_graph.get());

    insert_nop_on_matmul_input(lowered_graph.get());

    if (enable_device_tilize)
    {
        // If true, insert tilize op after activation (input)
        insert_tilize_op_on_input(lowered_graph.get());
    }

    // Recalculate shapes
    recalculate_shapes(lowered_graph.get());

    // Split big broadcasts into multiple smaller ones by adding nops between them
    if (enable_broadcast_splitting)
    {
        split_broadcasts(lowered_graph.get());
    }

    if (env_as<bool>("PYBUDA_ENABLE_CONSTANT_PRE_BROADCAST"))
    {
        constant_pre_broadcast(lowered_graph.get());
    }

    if (enable_recompute)
    {
        insert_recompute_ops(lowered_graph.get());
    }

    insert_partial_datacopy_tms(lowered_graph.get());

    // inserted fork-join NOPs
    for (auto instruction : ins_instructions)
    {
        instruction.second->insert(lowered_graph.get());
    }

    insert_user_defined_queues(lowered_graph.get(), insert_queues);

    //
    // Data formats
    //
    run_dataformat_passes(
        lowered_graph.get(),
        device_config,
        default_df_override,
        default_accumulate_df,
        fp32_fallback,
        default_math_fidelity,
        amp_level,
        amp_properties);

    // At this point, there should be no more graph mutations.
    placer::PlacerConfigUpdate placer_config_update = schedule_pre_placer_graph(
        lowered_graph.get(),
        device_config,
        scheduler_config,
        chip_ids,
        op_names_to_chip_break,
        op_names_to_epoch_break,
        fracture_chip_id_assignments,
        "" /* nops_remote_devices_postfix */,
        use_interactive_placer);

    return std::make_pair(std::move(lowered_graph), placer_config_update);
}

static std::vector<placer::DramAllocator> initialize_dram_allocators(const DeviceConfig& device_config, const PostPlacerConfig& config, const std::string &graph_name, std::vector<std::vector<placer::Blocks>> &pre_allocated_blocks)
{
    placer::DRAMPlacementAlgorithm placement_algorithm =
        env_as<bool>("PYBUDA_DRAM_CLOSEST") ? placer::CLOSEST : 
        env_as<bool>("PYBUDA_DRAM_PICK_CAPACITY") ? placer::GREATEST_CAPACITY : 
        env_as<bool>("PYBUDA_DRAM_FLIP_FLOP") ? placer::ROUND_ROBIN_FLIP_FLOP :
        config.placement_algorithm;
    std::vector<placer::DramAllocator> chip_dram_allocators;
    auto max_chip_id = *std::max_element(device_config.chip_ids.begin(),device_config.chip_ids.end());

    if (pre_allocated_blocks.size() <= max_chip_id)
    {
        pre_allocated_blocks.resize(max_chip_id + 1);
    }

    for (uint32_t chip_id = 0; chip_id <= max_chip_id; chip_id++) 
    {
        chip_dram_allocators.emplace_back(config.dram_placer_config, graph_name, chip_id, pre_allocated_blocks[chip_id], placement_algorithm);
    }

    return chip_dram_allocators;
}

static bool tensix_datacopy_eth_link_serialization_enabled() { return env_as<bool>("PYBUDA_ENABLE_ETH_SERIALIZATION"); }
static bool eth_datacopy_link_serialization_enabled() { return env_as<bool>("PYBUDA_ENABLE_ETH_DATACOPY_SERIALIZATION"); }

// ********** Run post-placer passes, like queue and buffer insertion **********
PostPlacerResults run_post_placer_buda_passes(
    graphlib::Graph *graph,
    const std::string &graph_name,
    const DeviceConfig &device_config,
    placer::PlacerSolution &placer_solution,
    PostPlacerConfig &config,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &previous_ins_instructions,
    std::vector<std::vector<placer::Blocks>> &pre_allocated_blocks,
    std::uint32_t last_host_address)
{
    set_prologue_queues(graph, balancer_solution->op_models);

    replace_recompute_with_checkpoint(graph, placer_solution);

    validate_subgraph_placement(graph, placer_solution);

    remove_buffering_queues_from_cross_epoch_edges(graph, placer_solution);

    // Insert queues between ops on different epochs.
    insert_epoch_to_epoch_queues(
        graph,
        placer_solution,
        {graphlib::NodeEpochType::Forward, graphlib::NodeEpochType::Backward, graphlib::Optimizer},
        balancer_solution->graph_solver_cut_edges);

    if (config.enable_t_streaming)
    {
        insert_t_stream_tms(graph, balancer_solution->op_models);
        optimize_tms(graph);
        recalculate_shapes(graph);
    }

    // Set queue entry sizes based on the configuration for different types of queues
    set_queue_sizes(graph, config, placer_solution);

    // Place and allocate DRAM queues
    std::vector<placer::DramAllocator> chip_dram_allocators = initialize_dram_allocators(device_config, config, graph_name, pre_allocated_blocks);
    placer::HostMemoryAllocator host_memory_allocator(config.host_memory_placer_config, last_host_address);

    placer::place_host_queues(
        config.host_memory_placer_config, host_memory_allocator, graph, placer_solution, *balancer_solution);
    placer::place_dram_queues(
        graph,
        placer_solution,
        *balancer_solution,
        config.host_memory_placer_config,
        config.dram_placer_config,
        chip_dram_allocators);

    if (eth_datacopy_link_serialization_enabled()) {
        TT_ASSERT(!tensix_datacopy_eth_link_serialization_enabled(), "Environment variables `PYBUDA_ENABLE_ETH_SERIALIZATION` and `PYBUDA_ENABLE_ETH_DATACOPY_SERIALIZATION` cannot be enabled at the same time.");
        // eth data copy ethernet stream serialization can be run prioer to dram queue allocation
        // Reduce ethernet stream usage runs after we insert queues because we may need to apply T streaming to e2e queues if they are on a remote chip
        // from their producer
        reduce_ethernet_stream_usage(config, graph, *(balancer_solution.get()), placer_solution, device_config);
        recalculate_shapes(graph);
        
        // allocate the new or flattened/serialized queues
        placer::place_host_queues(
            config.host_memory_placer_config, host_memory_allocator, graph, placer_solution, *balancer_solution);
        placer::place_dram_queues(
            graph,
            placer_solution,
            *balancer_solution,
            config.host_memory_placer_config,
            config.dram_placer_config,
            chip_dram_allocators);
    }
    
    if (tensix_datacopy_eth_link_serialization_enabled()) 
    {
        // Reduce ethernet stream usage runs after we insert queues because we may need to apply T streaming to e2e queues if they are on a remote chip
        // from their producer
        reduce_ethernet_stream_usage(config, graph, *(balancer_solution.get()), placer_solution, device_config);
        recalculate_shapes(graph);
        
        // allocate the new or flattened/serialized queues
        placer::place_host_queues(
            config.host_memory_placer_config, host_memory_allocator, graph, placer_solution, *balancer_solution);
        placer::place_dram_queues(
            graph,
            placer_solution,
            *balancer_solution,
            config.host_memory_placer_config,
            config.dram_placer_config,
            chip_dram_allocators);
    }

    // Lower additional buda attrs post placer
    post_placer_lower_buda_attrs(graph, device_config, balancer_solution->op_models);
    passes::validate_post_placer_data_formats(graph, device_config);

    PostPlacerResults results;

    results.current_host_address = host_memory_allocator.get_current_allocation_address();
    std::vector<std::vector<placer::Blocks>> allocated_blocks;
    for (auto &chip_dram_allocator : chip_dram_allocators) 
    {
        allocated_blocks.push_back(chip_dram_allocator.get_blocks());
    }
    results.allocated_blocks = allocated_blocks;

    if (!balancer_solution->placer_solution.fork_join_buffered)
    {
        // Add fork/join buffering, post-placement.
        //
        FJBufferingResult fj_buffering;
        fj_buffering = insert_fork_join_buffering(
            graph,
            &balancer_solution->op_models,
            nullptr /* inline op models */,
            config.device_config.get_l1_usable_size(),
            previous_ins_instructions,
            config.fork_join_tiles_treshold
            );

        results.ins_instructions = fj_buffering.instructions;

        if (!std::get<0>(is_subset_of_instructions(results.ins_instructions, previous_ins_instructions)))
        {
            return results;  // return here, since we'll have to redo anyway
        }
    }

    if (env_as<bool>("PYBUDA_UPSIZE_DRAM_INPUT"))
    {
        upsize_dram_input(graph, balancer_solution->op_models, config.device_config.get_l1_usable_size());
    }

    validate_multichip_queue_placements(config, graph, placer_solution);

    // Estimate model performance
    results.perf_model_results = perf_model::run_performance_model(
        graph,
        graph_name,
        device_config,
        balancer_solution,
        config.host_memory_placer_config.input_queues_on_host,
        config.host_memory_placer_config.output_queues_on_host);

    return results;
}

// ********** Last chance to run any non-iterative passes before netlist is generated **********
void run_pre_netlist_generation_buda_passes(
    graphlib::Graph *graph,
    const std::string &graph_name,
    const DeviceConfig &device_config,
    std::unordered_map<std::string, py::object> intermediates,
    placer::PlacerSolution &placer_solution,
    PostPlacerConfig &config,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    std::vector<std::vector<placer::Blocks>> &pre_allocated_blocks,
    std::uint32_t last_host_address
    )
{
    if (env_as<bool>("PYBUDA_REPRODUCE_SUBGRAPH"))
    {
        std::string input_name = env_as<string> ("PYBUDA_REPRODUCE_SUBGRAPH_INPUT");
        std::string output_name = env_as<string> ("PYBUDA_REPRODUCE_SUBGRAPH_OUTPUT");
        passes::reproduce_subgraph(graph, input_name, output_name, intermediates, balancer_solution, &placer_solution);
        // cutting graph will change the shape of some queuues and add new ones, so we re-place
        std::vector<placer::DramAllocator> chip_dram_allocators = initialize_dram_allocators(device_config, config, graph_name, pre_allocated_blocks);
        placer::HostMemoryAllocator host_memory_allocator(config.host_memory_placer_config, last_host_address);

        placer::place_host_queues(
            config.host_memory_placer_config, host_memory_allocator, graph, placer_solution, *balancer_solution);
        placer::place_dram_queues(
            graph,
            placer_solution,
            *balancer_solution,
            config.host_memory_placer_config,
            config.dram_placer_config,
            chip_dram_allocators);
    }
    return;
}
}  // namespace tt
