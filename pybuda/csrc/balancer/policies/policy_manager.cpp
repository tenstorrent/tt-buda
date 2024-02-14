// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policy_manager.hpp"

#include "balancer/policies/policy_utils.hpp"
#include "placer/lower_to_placer.hpp"

using NodeType = tt::graphlib::NodeType;

namespace tt::balancer
{

// Create policy manager and initialize with GS instance and schedule.
//
PolicyManager::PolicyManager(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    legalizer::GraphSolver& graph_solver,
    bool ribbon_policy) :
    graph(graph),
    config(config),
    interactive_placer(graph, config),
    interactive_placer_tester(graph, config),
    op_names_to_epoch_break(config.op_names_to_epoch_break),
    op_names_to_chip_break(config.op_names_to_chip_break),
    ribbon_policy(ribbon_policy)
{
    graph_solver_main = std::make_unique<legalizer::GraphSolver>(graph_solver);
    graph_solver_epoch_snapshot = nullptr;
    graph_solver_buffering_snapshot = nullptr;
    traversal_context = graph_solver_main->get_graph_traversal_context();

    std::tie(scheduled_ops, epoch_break_ops, chip_break_ops) =
        policy_run_scheduler(graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break, op_names_to_chip_break);
    op_nodes_to_process = scheduled_ops.size();

    if (ribbon_policy)
    {
        std::tie(next_ribbon_change, current_matmul_dim_r) = get_next_ribbon_change_op(graph, 0, scheduled_ops);
        current_ribbon_size = pick_ribbon_size(
            0, next_ribbon_change, graph, *graph_solver_main, scheduled_ops, config.device_config.grid_size.r);

        if (current_ribbon_size == 0)
        {
            current_ribbon_size = 1;
        }

        epoch_start_ribbon_size = current_ribbon_size;
        epoch_start_matmul_dim_r = current_matmul_dim_r;

        log_debug(
            LogBalancer,
            "Initial ribbon size set to {}, window of ops: {}-{}",
            current_ribbon_size,
            0,
            next_ribbon_change);

        // Try transposing op if it doesn't fit.
        //
        try_transpose_op = !env_as<bool>("PYBUDA_DISABLE_RIBBON_TRANSPOSE") && config.enable_auto_transposing_placement;
    }

    // Epoch snapshoting used by fj buffering or ribbon like policy for sparse-dense matmul epoch colocation.
    //
    if (use_interactive_fj_buffering or ribbon_policy)
    {
        graph_solver_epoch_snapshot = std::make_unique<legalizer::GraphSolver>(*graph_solver_main);
    }
}

// SETs op_model for currently balanced op_node and performs placing on chip.
// Returns tuple (op_commited, epoch_completed, new_epoch_forced).
// If epoch is completed, you need to finish_current_epoch() to proceed.
//
std::tuple<bool, bool, bool> PolicyManager::commit_op(const OpModel& selected_op_model)
{
    const graphlib::BudaOpNode* op = current_op_node->as<graphlib::BudaOpNode>();
    log_trace(LogBalancer, "Balancing: op_node {}.", current_op_node->name());

    bool force_new_epoch = epoch_break_ops.count(current_op_node->name()) > 0;
    bool skip_op_set = false;
    bool skip_op_place = false;

    bool new_epoch =
        (force_new_epoch || current_epoch_type != op->get_epoch_type()) && !interactive_placer.current_epoch_empty();

    // If transpose can help us align with current ribbon size, do it.
    //
    bool op_force_transposed = false;
    std::unordered_map<std::string, tt::placer::PlacerOpOverride>::iterator it;

    if (ribbon_policy and !op->is_sparse_matmul() and !buffered_op_model.has_value() and
        static_cast<std::uint32_t>(selected_op_model.grid_shape.r) != current_ribbon_size and
        static_cast<std::uint32_t>(selected_op_model.grid_shape.c) == current_ribbon_size and
        interactive_placer.can_fit_on_single_epoch(selected_op_model.grid_shape.c, selected_op_model.grid_shape.r) and
        interactive_placer.get_op_overrides().find(op->name()) == interactive_placer.get_op_overrides().end())
    {
        std::tie(it, op_force_transposed) =
            interactive_placer.get_op_overrides().emplace(op->name(), placer::PlacerOpOverride::force_op_transpose());
    }

    // Place, and figure out if it fits on the current epoch.
    //
    std::optional<placer::CoordRange> op_placement = std::nullopt;
    if (!new_epoch)
    {
        pair_two_ops_if_possible(op, selected_op_model, op_placement, skip_op_set, skip_op_place);

        if (!skip_op_place)
        {
            op_placement = interactive_placer.place_op(
                op->name(),
                selected_op_model.grid_shape,
                try_transpose_op /* enable_transpose */,
                chip_break_ops.find(op->name()) != chip_break_ops.end() /* chip_break */);
        }

        new_epoch = !op_placement.has_value() and !buffered_op_model;
    }

    if (new_epoch && interactive_placer.current_epoch_empty())
    {
        TT_THROW("Op {} doesn't fit on a single epoch", op->name());
    }

    // Make transpose op override expired so that it is not impacting rewinds(context might change and transpose is
    // not wanted anymore).
    //
    if (op_force_transposed)
    {
        interactive_placer.get_op_overrides().erase(it);
    }

    // Op placed, set it in graphsolver.
    //
    if (!new_epoch and !skip_op_set)
    {
        if (ribbon_policy)
        {
            set_op_model_for_node_ribbon(*graph_solver_main, current_op_node, selected_op_model, current_ribbon_size);
            update_ribbon_size();
        }
        else
        {
            set_op_model_for_node(
                *graph_solver_main, current_op_node, selected_op_model, config.device_config.arch_name);
        }

        current_epoch_ops.insert(current_op_node);
        current_epoch_selected_models.push_back(OpModelPair{selected_op_model, op});
        epoch_schedule.push_back(current_op_node->name());
    }

    // Return if op is commited, if current epoch is completed, and if switch to new epoch was forced.
    //
    return std::make_tuple(!new_epoch, new_epoch or op_index >= scheduled_ops.size(), force_new_epoch);
}

// Try buffering and pairing two ops for optimal(atomic) placing. Currently used for sparse-dense like matmul pairs.
//
void PolicyManager::pair_two_ops_if_possible(
    const graphlib::BudaOpNode* op,
    const OpModel& selected_op_model,
    std::optional<placer::CoordRange>& op_placement,
    bool& skip_op_set,
    bool& skip_op_place)
{
    // If we have sparse matmul followed by a pairable op(dense/depthwise matmul, reduce max)
    // buffer it for atomic placement in next iteration.
    //
    if (ribbon_policy and op->is_sparse_matmul() and op_index < scheduled_ops.size())
    {
        graphlib::Node* next_node = graph->get_node_by_name(scheduled_ops[op_index]);
        if (next_node->node_type() == NodeType::kBudaOp and epoch_break_ops.count(scheduled_ops[op_index]) == 0 and
            chip_break_ops.find(next_node->name()) == chip_break_ops.end())
        {
            const graphlib::BudaOpNode* dense_matmul_op = static_cast<graphlib::BudaOpNode*>(next_node);
            if (dense_matmul_op->should_pair_with_sparse(op, graph))
            {
                TT_ASSERT(!buffered_op_model.has_value());
                graph_solver_pairing_checkpoint = std::make_unique<legalizer::GraphSolver>(*graph_solver_main);
                set_op_model_for_node_ribbon(
                    *graph_solver_main, current_op_node, selected_op_model, current_ribbon_size);
                buffered_op_model = selected_op_model;
                skip_op_set = true;
                skip_op_place = true;
            }
        }
    }
    // We have a buffered op_model check compatibility with current selected one.
    //
    else if (buffered_op_model.has_value())
    {
        // Check if buffered + selected can fit on single epoch.
        //
        if (can_fit_on_single_epoch(
                interactive_placer_tester,
                buffered_op_model->buda_op_node->name(),
                buffered_op_model->grid_shape,
                op->name(),
                selected_op_model.grid_shape,
                try_transpose_op))
        {
            skip_op_place = true;
            skip_op_set = true;

            // Rowsize matches, place them as bound pair.
            //
            if (selected_op_model.grid_shape.r == buffered_op_model->grid_shape.r and
                interactive_placer.can_fit_on_single_epoch(
                    buffered_op_model->grid_shape.r,
                    buffered_op_model->grid_shape.c + selected_op_model.grid_shape.c,
                    try_transpose_op /* allow_transpose */))
            {
                op_placement = interactive_placer.place_two_ops_rowwise(
                    buffered_op_model->buda_op_node->name(),
                    buffered_op_model->grid_shape,
                    op->name(),
                    selected_op_model.grid_shape,
                    try_transpose_op, /* enable_transpose */
                    chip_break_ops.find(op->name()) != chip_break_ops.end() /* chip_break */
                );

                if (op_placement.has_value())
                {
                    current_epoch_ops.insert(buffered_op_model->buda_op_node);
                    current_epoch_selected_models.emplace_back(OpModelPair{*buffered_op_model, buffered_op_model->buda_op_node});
                    epoch_schedule.push_back(buffered_op_model->buda_op_node->name());
                    set_op_model_for_node_ribbon(
                        *graph_solver_main, current_op_node, selected_op_model, current_ribbon_size);
                    update_ribbon_size();
                    current_epoch_ops.insert(op);
                    current_epoch_selected_models.emplace_back(OpModelPair{selected_op_model, op});
                    epoch_schedule.push_back(op->name());
                }
            }
            // Rowsize does not match. Still try to place them next to each other in a single epoch.
            //
            else
            {
                op_placement = interactive_placer.place_op(
                    buffered_op_model->buda_op_node->name(),
                    buffered_op_model->grid_shape,
                    try_transpose_op /* enable_transpose */,
                    chip_break_ops.find(op->name()) != chip_break_ops.end() /* chip_break */);

                if (op_placement.has_value())
                {
                    op_placement = interactive_placer.place_op(
                        op->name(),
                        selected_op_model.grid_shape,
                        try_transpose_op /* enable_transpose */,
                        false /* chip_break */);

                    if (op_placement.has_value())
                    {
                        current_epoch_ops.insert(buffered_op_model->buda_op_node);
                        current_epoch_selected_models.push_back(OpModelPair{*buffered_op_model, buffered_op_model->buda_op_node});
                        epoch_schedule.push_back(buffered_op_model->buda_op_node->name());
                        set_op_model_for_node_ribbon(
                            *graph_solver_main, current_op_node, selected_op_model, current_ribbon_size);
                        update_ribbon_size();
                        current_epoch_ops.insert(op);
                        current_epoch_selected_models.push_back(OpModelPair{selected_op_model, op});
                        epoch_schedule.push_back(op->name());
                        skip_op_set = true;
                    }
                    else
                    {
                        // Revert buffered op placement as paired op placement failed. We dont want them in separate
                        // epochs.
                        //
                        interactive_placer.rewind_to(buffered_op_model->buda_op_node->name());
                    }
                }
            }
        }
        // If buffered + selected cannot fit on single epoch
        // we need to back out and place them separately. Buffered first with currently selected one coming after in
        // regular independent placement.
        //
        else
        {
            // Place only buffered one.
            //
            op_placement = interactive_placer.place_op(
                buffered_op_model->buda_op_node->name(),
                buffered_op_model->grid_shape,
                try_transpose_op /* enable_transpose */,
                chip_break_ops.find(op->name()) != chip_break_ops.end() /* chip_break */);

            if (op_placement.has_value())
            {
                current_epoch_ops.insert(buffered_op_model->buda_op_node);
                current_epoch_selected_models.push_back(OpModelPair{*buffered_op_model, buffered_op_model->buda_op_node});
                epoch_schedule.push_back(buffered_op_model->buda_op_node->name());
            }
        }

        if (!op_placement.has_value())
        {
            // Paired placement failed, revert to prebuffering pairing checkpoint.
            //
            traversal_context.reset();
            graph_solver_main = std::make_unique<legalizer::GraphSolver>(*graph_solver_pairing_checkpoint);
            traversal_context = graph_solver_main->get_graph_traversal_context();
            skip_op_set = true;
            skip_op_place = true;
            op_index--;
        }

        buffered_op_model.reset();
        graph_solver_pairing_checkpoint = nullptr;
    }
}

// Check for ribbon size changes.
//
void PolicyManager::update_ribbon_size()
{
    TT_ASSERT(ribbon_policy);

    if (op_index == next_ribbon_change and op_index > 0 and op_index < scheduled_ops.size())
    {
        std::tie(next_ribbon_change, current_matmul_dim_r) = get_next_ribbon_change_op(graph, op_index, scheduled_ops);
        std::uint32_t next_ribbon_size = pick_ribbon_size(
            op_index, next_ribbon_change, graph, *graph_solver_main, scheduled_ops, config.device_config.grid_size.r);
        if (next_ribbon_change < scheduled_ops.size())
            log_debug(LogBalancer, "Next change at {}", scheduled_ops[next_ribbon_change]);

        // Force epoch change if ribbon size changes. In the future, we can handle this with a queue, or padding.
        if ((current_ribbon_size != next_ribbon_size) && !interactive_placer.current_epoch_empty())
        {
            const graphlib::Node* node = graph->get_node_by_name(scheduled_ops[op_index]);
            TT_ASSERT(node->node_type() == NodeType::kBudaOp);
            cut_graph_solver_ribbon(graph, node, interactive_placer, *graph_solver_main);

            current_ribbon_size = next_ribbon_size;
            log_debug(LogBalancer, "Changing current ribbon size to {} at op {}", current_ribbon_size, node->name());
        }
    }
}

// Finish current epoch. Performs inline fork-join buffering if enabled.
// Returns true if epoch is finished. If balancing is not complete new epoch will be auto-started.
// Returns false if epoch is rewinded due to buffering. State/counters/current_op_node are reset to current epoch start.
//
bool PolicyManager::finish_current_epoch()
{
    TT_ASSERT(!interactive_placer.current_epoch_empty(), "Cannot finish empty epoch!");
    bool balancing_complete = op_index >= scheduled_ops.size() and current_epoch_ops.count(current_op_node) > 0;
    if (use_interactive_fj_buffering)
    {
        // Handle case when current epoch overflows to next one due to buffering.
        //
        if (!balancing_complete and !pre_buffering_epoch_ops.empty())
        {
            bool buffered_epoch_overflow = handle_epoch_buffering_overflow();
            if (buffered_epoch_overflow)
                return false;
        }

        // If we are at the end of current epoch try buffering fork joins.
        //
        bool epoch_buffered = buffer_epoch();
        if (epoch_buffered)
            return false;
    }

    TT_ASSERT(current_epoch_ops.size() == current_epoch_selected_models.size(), "Epoch ops and selected op models mismatch!");
    epoch_solutions.emplace_back(current_ribbon_size, &config, current_epoch_selected_models, graph, -1);

    if (!balancing_complete)
    {
        start_new_epoch(current_op_node->as<graphlib::BudaOpNode>()->get_epoch_type());
        Logger<kLoggerABI>::get().log_level_type(
            Logger<kLoggerABI>::Level::Info,
            LogBalancer,
            "Balancing {}% complete.",
            processed_nodes.size() * 100 / op_nodes_to_process);
    }
    else
    {
        processed_nodes.insert(current_epoch_ops.begin(), current_epoch_ops.end());
        processed_schedule.insert(processed_schedule.end(), epoch_schedule.begin(), epoch_schedule.end());
        epoch_schedule.clear();
        current_epoch_ops.clear();
        TT_ASSERT(processed_nodes.size() == op_nodes_to_process, "Not all nodes were processed!");
        Logger<kLoggerABI>::get().log_level_type(
            Logger<kLoggerABI>::Level::Info, LogBalancer, "Balancing 100% completed!");
    }

    return true;
}

// Starts new epoch with incoming op epoch type. Update and reset epoch related variables and counters.
//
void PolicyManager::start_new_epoch(graphlib::NodeEpochType epoch_type)
{
    last_epoch_start = op_index - 1;
    cut_graph_solver_epoch(graph, interactive_placer, *graph_solver_main);
    current_epoch_type = epoch_type;
    interactive_placer.next_epoch(current_epoch_type);
    if (use_interactive_fj_buffering or ribbon_policy)
    {
        // Starting new epoch, make graph solver snapshot, record processed nodes,
        // clear epoch overflow, clear previous epoch nodes and clear buffering instructions.
        //
        if (overflow_set_for_epoch)
        {
            op_names_to_epoch_break.pop_back();
            epoch_break_ops = placer::lowering::tag_ops_for_epoch_break(
                config.device_config.arch_name,
                op_names_to_epoch_break,
                op_names_to_chip_break,
                scheduled_ops,
                graph,
                true /* use_interactive_placer */);
        }

        overflow_set_for_epoch = false;
        graph_solver_epoch_snapshot = std::make_unique<legalizer::GraphSolver>(*graph_solver_main);
        graph_solver_buffering_snapshot = nullptr;
        pre_buffering_epoch_ops.clear();
        processed_nodes.insert(current_epoch_ops.begin(), current_epoch_ops.end());
        processed_schedule.insert(processed_schedule.end(), epoch_schedule.begin(), epoch_schedule.end());
        current_epoch_ops.clear();
        current_epoch_selected_models.clear();
        epoch_schedule.clear();
        inst.clear();

        if (ribbon_policy)
        {
            epoch_start_ribbon_size = current_ribbon_size;
            epoch_start_matmul_dim_r = current_matmul_dim_r;
        }
    }

    op_index--;

    // Start new epoch, place op again.
    //
    log_debug(LogBalancer, "Starting new epoch");
}

// Buffer current epoch. Returns true if epoch was buffered/graph was changed.
//
bool PolicyManager::buffer_epoch()
{
    graphlib::Graph* graph_modify = const_cast<graphlib::Graph*>(graph);
    OpModels* op_models = graph_solver_main->get_selected_op_models_for_buffering(current_epoch_ops);
    std::unordered_set<const tt::graphlib::Node*> current_epoch_nodes =
        calculate_current_epoch_nodes(graph, current_epoch_ops);
    FJBufferingResult fj_buffering;

    {
        // Generate buffering instructions if this epoch needs buffering.
        // We are scoping down FJ buffering algorithm to subgraph by setting GraphTraversalContext
        // to current epoch nodes.
        //
        std::unique_ptr<graphlib::GraphTraversalContext> epoch_traversal_context =
            graph_solver_main->get_graph_epoch_traversal_context(&current_epoch_nodes);
        fj_buffering = insert_fork_join_buffering(
            graph_modify,
            nullptr /* postplacer op models */,
            op_models,
            config.device_config.get_l1_usable_size(),
            prev_inst,
            config.fork_join_tiles_treshold,
            ribbon_policy ? &ribbon_buffering_factor : [](const tt::balancer::OpModel&) { return 1; });
    }

    inst = fj_buffering.instructions;
    if (!std::get<0>(is_subset_of_instructions(inst, prev_inst)))
    {
        // We need to buffer, so we need to rewind the epoch and place again with buffer nodes.
        // Revert graphsolver to snapshot. Release old traversal context.
        //
        bool graph_modified = false;
        interactive_placer.rewind_epoch();
        traversal_context.reset();

        // If we are buffering this epoch for the first time, save snapshot of current epoch nodes.
        //
        if (pre_buffering_epoch_ops.empty())
        {
            pre_buffering_epoch_ops.insert(
                pre_buffering_epoch_ops.end(), current_epoch_ops.begin(), current_epoch_ops.end());
        }

        graph_solver_main = std::make_unique<legalizer::GraphSolver>(
            graph_solver_buffering_snapshot ? *graph_solver_buffering_snapshot : *graph_solver_epoch_snapshot);
        {
            // Operate only within current epoch nodes.
            //
            std::unique_ptr<graphlib::GraphTraversalContext> epoch_traversal_context =
                graph_solver_main->get_graph_epoch_traversal_context(&current_epoch_nodes);
            graph_modified = buffer_graph(graph_modify, inst, *graph_solver_main);
        }

        // Reset current epoch nodes and traversal context to old state(snapshot).
        //
        current_epoch_ops.clear();
        current_epoch_selected_models.clear();
        epoch_schedule.clear();
        traversal_context = graph_solver_main->get_graph_traversal_context();
        if (graph_modified)
        {
            // If we added new non queue nodes we need to rerun scheduler.
            // Make scheduler ignore already processed nodes.
            //
            std::tie(scheduled_ops, epoch_break_ops, chip_break_ops) =
                policy_run_scheduler(graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break, op_names_to_chip_break);
            op_nodes_to_process = scheduled_ops.size() + processed_nodes.size();
            op_index = 0;
            last_epoch_start = 0;
        }
        else
        {
            // No new nodes added, continue from last epoch start.
            //
            op_index = last_epoch_start;
        }

        if (ribbon_policy)
        {
            std::tie(next_ribbon_change, current_matmul_dim_r) =
                get_next_ribbon_change_op(graph, op_index, scheduled_ops, epoch_start_matmul_dim_r);
            current_ribbon_size = epoch_start_ribbon_size;
        }

        // Record new snapshot and cache buffering instructions for next buffering cycle.
        //
        graph_solver_buffering_snapshot = std::make_unique<legalizer::GraphSolver>(*graph_solver_main);
        return true;
    }

    return false;
}

// In case buffering causes current epoch to overflow, cut graph before the overflow and retry epoch with new
// buffering(likely fewer buffers as smaller number of true OPs remain).
//
bool PolicyManager::handle_epoch_buffering_overflow()
{
    // Record all ops which were present in one epoch before buffering but overflowed for the current epoch.
    //
    scheduler::Schedule overflowed_ops;
    for (const Node* node : pre_buffering_epoch_ops)
    {
        if (current_epoch_ops.count(node) == 0)
        {
            overflowed_ops.push_back(node->name());
        }
    }

    if (!overflowed_ops.empty())
    {
        // If we have already set an overflow epoch break for this epoch, remove it.
        // Due to additional buffering it turns out that we need to cut earlier.
        //
        if (overflow_set_for_epoch)
        {
            op_names_to_epoch_break.pop_back();
        }

        // Mark these nodes as set of epoch break nodes. This will effectively resolve fork-join buffering for
        // this path with E2E queue(if one of these nodes was indeed part of the fork join).
        //
        op_names_to_epoch_break.push_back(overflowed_ops);
        overflow_set_for_epoch = true;

        // Rewind the epoch, reset state of all counters, revert buffering, reschedule and try again.
        //
        rewind_epoch();

        return true;
    }

    return false;
}

// Get next OP to balance.
//
const graphlib::Node* PolicyManager::get_next_op()
{
    if (op_index >= scheduled_ops.size())
    {
        current_op_node = nullptr;
    }
    else
    {
        current_op_node = graph->get_node_by_name(scheduled_ops[op_index++]);
        TT_ASSERT(current_op_node->node_type() == NodeType::kBudaOp);
    }

    return current_op_node;
}

// Rewinds epoch in progress from interactive placer. Reverts GS state to epoch start snapshot.
// Resets all epoch related counters.
//
void PolicyManager::rewind_epoch()
{
    TT_ASSERT(graph_solver_epoch_snapshot != nullptr, "Cannot rewind epoch without snapshot!");
    interactive_placer.rewind_epoch();
    traversal_context.reset();
    graph_solver_main = std::make_unique<legalizer::GraphSolver>(*graph_solver_epoch_snapshot);
    pre_buffering_epoch_ops.clear();
    current_epoch_ops.clear();
    current_epoch_selected_models.clear();
    epoch_schedule.clear();
    buffered_op_model.reset();
    graph_solver_pairing_checkpoint = nullptr;
    traversal_context = graph_solver_main->get_graph_traversal_context();

    // If epoch was buffered or overflowed(epoch break was set), we need to reschedule.
    //
    if (graph_solver_buffering_snapshot or overflow_set_for_epoch)
    {
        std::tie(scheduled_ops, epoch_break_ops, chip_break_ops) =
            policy_run_scheduler(graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break, op_names_to_chip_break);
        op_nodes_to_process = scheduled_ops.size() + processed_nodes.size();

        if (ribbon_policy)
        {
            std::tie(next_ribbon_change, current_matmul_dim_r) =
                get_next_ribbon_change_op(graph, 0, scheduled_ops, epoch_start_matmul_dim_r);
        }

        op_index = 0;
        last_epoch_start = 0;
    }
    else
    {
        op_index = last_epoch_start;

        if (ribbon_policy)
        {
            std::tie(next_ribbon_change, current_matmul_dim_r) =
                get_next_ribbon_change_op(graph, last_epoch_start, scheduled_ops, epoch_start_matmul_dim_r);
        }
    }

    if (ribbon_policy)
    {
        current_ribbon_size = epoch_start_ribbon_size;
    }

    graph_solver_buffering_snapshot = nullptr;
    inst.clear();
}

// Force current epoch to break at specified OP. Will automatically rewind current epoch so that new epoch break could
// be applied.
// Returns true if epoch break was successful.
//
bool PolicyManager::force_current_epoch_break(const std::string& op_name)
{
    // Can't break epoch on the first op.
    //
    if (scheduled_ops[last_epoch_start] == op_name)
    {
        return false;
    }

    // If we have already set an overflow epoch break for this epoch, remove it.
    //
    if (overflow_set_for_epoch)
    {
        op_names_to_epoch_break.pop_back();
    }

    scheduler::Schedule current_epoch_break;
    current_epoch_break.push_back(op_name);
    op_names_to_epoch_break.push_back(current_epoch_break);
    overflow_set_for_epoch = true;

    // Rewind the epoch, reset state of all counters, revert buffering, reschedule and try again.
    //
    rewind_epoch();

    return true;
}

// Commit and validate interactive placer solution.
//
tt::placer::PlacerSolution PolicyManager::commit_solution()
{
    if (use_interactive_fj_buffering or ribbon_policy)
    {
        // If we used fork join buffering, we rerun scheduler more than once so in order to validate
        // we need to reconstruct scheduled_ops.
        //
        std::unordered_set<const tt::graphlib::Node*> empty_set_processed_nodes;
        tt::scheduler::Schedule empty_processed_schedule;
        // op_names_to_epoch_break and op_names_to_chip_break are empty by now because processed nodes are removed as the placement was done
        // we want original epoch_breaks and chip_breaks, hence, we pass in config.op_names_to_epoch_break and config.op_names_to_chip_break
        std::tie(scheduled_ops, epoch_break_ops, chip_break_ops) =
            policy_run_scheduler(graph, config, empty_set_processed_nodes, empty_processed_schedule, config.op_names_to_epoch_break, config.op_names_to_chip_break);
    }

    tt::placer::PlacerSolution placer_solution = interactive_placer.commit(chip_break_ops);
    placer_solution.fork_join_buffered = use_interactive_fj_buffering;

    validate_solution(scheduled_ops, placer_solution);
    score_solution(epoch_solutions, config.device_config);

    return placer_solution;
}

}  // namespace tt::balancer
