// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <variant>

#include "balancer/policies/policy_ribbon.hpp"
#include "graph_lib/utils.hpp"
#include "passes/fork_join.hpp"
#include "placer/interactive_placer.hpp"
#include "placer/lower_to_placer.hpp"
#include "placer/placer.hpp"
#include "scheduler/scheduler.hpp"
#include "scheduler/utils.hpp"
#include "utils/assert.hpp"
#include "utils/logger.hpp"

using NodeType = tt::graphlib::NodeType;

namespace tt::balancer
{

OpModel get_closest_op_model(
    const legalizer::GraphSolver &graph_solver_snapshot,
    const OpModel &op_model_target,
    const BalancerConfig *balancer_config,
    const graphlib::Graph *graph,
    std::unordered_set<std::uint64_t> &validated_cache)
{
    std::optional<OpModel> closest_model = std::nullopt;
    bool is_sparse_matmul = op_model_target.buda_op_node->is_sparse_matmul();
    for (const auto &op_model : graph_solver_snapshot.at(op_model_target.buda_op_node))
    {
        if (is_sparse_matmul)
        {
            if (!validate_sparse_matmul_model(op_model_target.buda_op_node, op_model, graph, validated_cache))
            {
                continue;
            }
        }

        // try to set the same op model as before, if possible. If not, then pick the closest one
        if (op_model == op_model_target)
        {
            closest_model = op_model;
            break;
        }
        if (!closest_model.has_value())
        {
            closest_model = op_model;
        }
        else
        {
            auto my_delta = std::abs(
                get_limiter_cycles(op_model, graph, *balancer_config) -
                get_limiter_cycles(op_model_target, graph, *balancer_config));
            auto best_delta = std::abs(
                get_limiter_cycles(*closest_model, graph, *balancer_config) -
                get_limiter_cycles(op_model_target, graph, *balancer_config));

            if (my_delta < best_delta)
            {
                closest_model = op_model;
            }
            else if (my_delta == best_delta)
            {
                // Prefer the same shape
                if (op_model_target.grid_shape == op_model.grid_shape)
                {
                    closest_model = op_model;
                }
            }
        }
    }
    TT_ASSERT(closest_model.has_value());
    return closest_model.value();
}

// Optimize a solution by iteratively bumping up grids of the slowest ops, as long as that
// improves the utilization of the epoch. We ideally try to stick to the same ribbon size, but if
// that's not possible, we'll bump up the grid to anything available that's slightly better than
// the current grid.
EpochSolution optimize_solution(
    EpochSolution &solution,
    const legalizer::GraphSolver &graph_solver,
    placer::InteractivePlacer &interactive_placer,
    const graphlib::Graph *graph,
    std::unordered_set<std::uint64_t> &validated_cache,
    std::uint32_t max_iterations)
{
    log_trace(LogBalancer, "RIBBON2: optimize solution, score {}, coming in:", solution.get_score());
    solution.print();

    EpochSolution best_solution = solution;

    std::uint32_t iterations = 0;
    std::uint32_t bad_iterations = 0;  // number of iterations in a row that made thing worse
    const BalancerConfig *balancer_config = solution.get_balancer_config();  // save some typing
    while ((bad_iterations < 3) && (iterations < max_iterations))
    {
        // Find the slowest cycle count
        float slowest_cycles = 0;
        for (const auto &op_model : best_solution.get_selected_op_models())
        {
            float cycles = get_limiter_cycles(op_model, graph, *balancer_config);
            if (cycles > slowest_cycles)
                slowest_cycles = cycles;
        }

        // Now go through the models, and bump up the ones that are slowest
        auto graph_solver_snapshot = std::make_unique<legalizer::GraphSolver>(graph_solver);
        auto new_solution = best_solution;
        auto target_cycles = 0.9 * slowest_cycles;
        std::vector<OpModel> blacklisted_models;  // models from previous bad iterations that shouldn't be tried again
        log_trace(LogBalancer, "RIBBON2: target_cycles = {}", target_cycles);
        for (std::size_t op_index = 0; op_index < new_solution.get_selected_op_models().size(); op_index++)
        {
            const auto &op_model = new_solution.get_selected_op_models()[op_index];
            bool is_sparse_matmul = op_model.buda_op_node->is_sparse_matmul();
            float cycles = get_limiter_cycles(op_model, graph, *balancer_config);
            if (cycles < target_cycles)
            {
                log_trace(LogBalancer, "RIBBON2: op {} is fast enough", op_model.buda_op_node->name());
                auto closest_model =
                    get_closest_op_model(*graph_solver_snapshot, op_model, balancer_config, graph, validated_cache);
                graph_solver_snapshot->set(op_model.buda_op_node, closest_model);
                if (!(closest_model == op_model))
                {
                    log_trace(
                        LogBalancer,
                        "RIBBON2: had to change the grid to {} with cycles {}",
                        closest_model.grid_shape,
                        get_limiter_cycles(closest_model, graph, *balancer_config));
                    new_solution.update_model(op_index, closest_model);
                }
            }
            else
            {
                // Bump up the grid
                // Ideally, use the same ribbon size first
                log_trace(LogBalancer, "RIBBON2: op {} is too slow, bumping up grid", op_model.buda_op_node->name());
                std::optional<OpModel> new_op_model = std::nullopt;
                for (bool same_ribbon : {true, false})
                {
                    // Check for the case where none of the grids can have prologue, and then waive it
                    bool waive_prologue = true;
                    for (const auto &op_model : graph_solver_snapshot->at(op_model.buda_op_node))
                        if (prologue_ok(op_model))
                        {
                            waive_prologue = false;
                            break;
                        }

                    for (const auto &op_model : graph_solver_snapshot->at(op_model.buda_op_node))
                    {
                        log_trace(
                            LogBalancer,
                            "RIBBON2: trying grid {} with cycles {}, for same ribbon {}",
                            op_model.grid_shape,
                            get_limiter_cycles(op_model, graph, *balancer_config),
                            same_ribbon);
                        if (is_sparse_matmul)
                        {
                            log_trace(
                                LogBalancer,
                                "RIBBON2: trying sparse_matmul grid {} with cycles {}, u_kt = {}",
                                op_model.grid_shape,
                                get_limiter_cycles(op_model, graph, *balancer_config),
                                op_model.input_buffers.at(1).block_shape.ublock.rt);
                        }

                        if (std::find(blacklisted_models.begin(), blacklisted_models.end(), op_model) !=
                            blacklisted_models.end())
                        {
                            log_trace(LogBalancer, "RIBBON2: skipping blacklisted op_model");
                            continue;
                        }

                        if (!waive_prologue && !prologue_ok(op_model))
                            continue;

                        if (same_ribbon && (op_model.grid_shape.r != (int)new_solution.get_ribbon_size()))
                            continue;

                        if (get_limiter_cycles(op_model, graph, *balancer_config) >= slowest_cycles)
                            continue;

                        // Find the slowest improvement over the current op_model, to reduce drastic changes
                        if (!new_op_model.has_value() ||  // nothing has been picked

                            // current best is improvement, but not +10%
                            ((get_limiter_cycles(*new_op_model, graph, *balancer_config) >= target_cycles) &&
                             (get_limiter_cycles(op_model, graph, *balancer_config) < target_cycles)) ||

                            // pick slower improvement
                            (get_limiter_cycles(*new_op_model, graph, *balancer_config) <
                             get_limiter_cycles(op_model, graph, *balancer_config)))
                        {
                            bool op_ok = true;
                            if (is_sparse_matmul)
                            {
                                // Make sure that this sparse model can be encoded correctly
                                op_ok = validate_sparse_matmul_model(
                                    op_model.buda_op_node, op_model, graph, validated_cache);
                            }

                            if (op_ok)
                            {
                                new_op_model = op_model;
                                log_trace(
                                    LogBalancer,
                                    "RIBBON2: setting new grid for {}: {} with cycles {}",
                                    op_model.buda_op_node->name(),
                                    op_model.grid_shape,
                                    get_limiter_cycles(op_model, graph, *balancer_config));
                            }
                        }
                    }
                    if (same_ribbon && new_op_model.has_value())
                        break;  // don't try changing the ribbon size, since we found an improvement with the same
                                // ribbon
                }

                // If we found a larger grid, then use it
                if (new_op_model.has_value())
                {
                    log_trace(
                        LogBalancer,
                        "RIBBON2: bumping up {} from {} to {}",
                        op_model.buda_op_node->name(),
                        op_model.grid_shape,
                        new_op_model->grid_shape);
                    new_solution.update_model(op_index, new_op_model.value());
                    graph_solver_snapshot->set(op_model.buda_op_node, new_op_model.value());
                    blacklisted_models.push_back(new_op_model.value());  // record in case this bump ended up being bad
                }
                else
                {
                    // We haven't found anything better, set the same (or closest legal)
                    auto closest_model =
                        get_closest_op_model(*graph_solver_snapshot, op_model, balancer_config, graph, validated_cache);
                    new_solution.update_model(op_index, closest_model);
                    graph_solver_snapshot->set(op_model.buda_op_node, closest_model);
                }
            }
        }

        // We need to place this new solution to see how much of it actually fits
        std::size_t placed_ops = 0;
        for (std::size_t i = 0; i < new_solution.get_selected_op_models().size(); i++)
        {
            const auto &op_model = new_solution.get_selected_op_models()[i];
            std::optional<placer::CoordRange> op_placement;
            int placing_step = 1;

            const OpModel *next_op = i < new_solution.get_selected_op_models().size() - 1
                                         ? &new_solution.get_selected_op_models()[i + 1]
                                         : nullptr;

            // Special case for sparse-dense matmul pairing. We want to always place them atomically together if
            // possible.
            //
            if (next_op and can_bind_sparse_dense_matmul_pair(
                                graph,
                                op_model.buda_op_node,
                                op_model,
                                next_op->buda_op_node,
                                *next_op,
                                interactive_placer,
                                true /*allow_transpose*/))
            {
                op_placement = interactive_placer.place_two_ops_rowwise(
                    op_model.buda_op_node->name(),
                    op_model.grid_shape,
                    next_op->buda_op_node->name(),
                    next_op->grid_shape,
                    true);

                placing_step = 2;
                i++;
            }
            else
            {
                op_placement = interactive_placer.place_op(op_model.buda_op_node->name(), op_model.grid_shape, true);
            }

            if (op_placement.has_value())
            {
                placed_ops += placing_step;
            }
            else
            {
                break;
            }
        }
        interactive_placer.rewind_epoch();  // rewind, we were just testing what fits
        if (placed_ops < new_solution.get_selected_op_models().size())
        {
            // Trim the solution
            new_solution.set_op_count(placed_ops);
            log_trace(LogBalancer, "RIBBON2: trimmed solution to {} ops", placed_ops);
        }

        if (new_solution.get_score() > best_solution.get_score())
        {
            best_solution = new_solution;
            bad_iterations = 0;
            blacklisted_models.clear();
            log_trace(LogBalancer, "RIBBON2: improved to {}", best_solution.get_score());
        }
        else
        {
            bad_iterations++;
            log_trace(LogBalancer, "RIBBON2: solution got worse, bad iterations in a row = {}", bad_iterations);
        }
        iterations++;
    }

    log_trace(LogBalancer, "RIBBON2: optimized solution with score {}:", best_solution.get_score());
    best_solution.print();
    return best_solution;
}

bool handle_fork_join_nop_overflow(
    graphlib::Graph const *graph,
    const BalancerConfig &config,
    std::vector<std::vector<std::string>> &op_names_to_epoch_break,
    EpochSolution &solution,
    std::unique_ptr<EpochSolution> &pre_buffered_solution,
    std::unique_ptr<legalizer::GraphSolver> &graph_solver,
    std::unique_ptr<legalizer::GraphSolver> &pre_buffered_graph_snapshot,
    std::unordered_set<std::string> &epoch_break_ops,
    std::uint32_t &placed_op_index,
    scheduler::Schedule &scheduled_ops,
    const std::unordered_set<const Node *> &processed_nodes,
    const tt::scheduler::Schedule &processed_schedule,
    std::unique_ptr<graphlib::GraphTraversalContext> &traversal_context,
    std::uint32_t &nodes_to_process,
    std::uint32_t current_epoch,
    std::vector<const Node *> &fork_and_join_nodes,
    bool &epoch_breaks_added)
{
    const bool cleanup_buffering_nops = !env_as<bool>("PYBUDA_RIBBON2_DISABLE_CLEANUP_BUF_NOPS", 0);
    if (!cleanup_buffering_nops)
    {
        return false;
    }

    if (pre_buffered_graph_snapshot.get() == nullptr)
    {
        return false;
    }

    // Fork-join buffering for this epoch was added in previous iteration.
    // Check if added buffering caused any of the fork-joins to split into two epochs.
    // If that is the case, there is no point in keeping the added nops for buffering.

    // Get all ops in current epoch.
    std::unordered_set<const Node *> ops_in_curr_epoch;
    for (const auto &op_model : solution.get_selected_op_models())
    {
        ops_in_curr_epoch.insert(op_model.buda_op_node);
    }

    // Check if all fork and join nodes are in this epoch.
    bool needs_epoch_break = false;
    for (auto node : fork_and_join_nodes)
    {
        if (!ops_in_curr_epoch.count(node))
        {
            needs_epoch_break = true;
        }
    }

    if (!needs_epoch_break)
    {
        return false;
    }

    log_debug(LogBalancer, "Detected fork-join split due to buffering in epoch {}.", current_epoch);

    // Get all ops which we wanted to place in this epoch (pre_buffered_solution) and make explicit epoch breaks
    // for all of the ops which didn't fit.
    scheduler::Schedule epoch_break;
    for (const auto &op_model : pre_buffered_solution->get_selected_op_models())
    {
        // We don't mark nops for epoch break, since they won't exist when we revert the graph to the pre-buffered
        // snapshot.
        if (!ops_in_curr_epoch.count(op_model.buda_op_node) and !op_model.buda_op_node->is_buffering_op())
        {
            epoch_break.push_back(op_model.buda_op_node->name());
        }
    }

    if (epoch_breaks_added)
    {
        op_names_to_epoch_break.pop_back();
    }

    op_names_to_epoch_break.push_back(epoch_break);
    epoch_breaks_added = true;
    pre_buffered_solution.reset();

    // Since we can no longer fit all of the pre-buffered ops on a single epoch,
    // undo the buffering, reschedule everything (with explicit epoch breaks added) and continue to search for a
    // solution. This takes care of cases where we leave unnecessary fork-join buffering which spans multiple epochs.
    graph_solver = std::move(pre_buffered_graph_snapshot);

    traversal_context.reset();
    traversal_context = graph_solver->get_graph_traversal_context();

    std::tie(scheduled_ops, epoch_break_ops) =
        policy_run_scheduler(graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break);

    placed_op_index = 0;
    nodes_to_process = processed_nodes.size() + scheduled_ops.size();
    fork_and_join_nodes.clear();

    return true;
}

// Try to insert fork join buffering, and then apply solution to the graph solver.
// If graph has changed due to new ops, functions doesn't apply the solution and
// returns false. It is expected that the parent will then re-solve the epoch and
// call this again.
bool apply_solution(
    graphlib::Graph const *graph,
    const BalancerConfig &config,
    std::vector<std::vector<std::string>> &op_names_to_epoch_break,
    EpochSolution &solution,
    std::unique_ptr<legalizer::GraphSolver> &graph_solver,
    std::unique_ptr<legalizer::GraphSolver> &graph_solver_epoch_snapshot,
    placer::InteractivePlacer &interactive_placer,
    std::unordered_set<string> &epoch_break_ops,
    scheduler::Schedule &scheduled_ops,
    std::unordered_set<const tt::graphlib::Node *> &processed_nodes,
    tt::scheduler::Schedule &processed_schedule,
    std::uint32_t &placed_op_index,
    std::unique_ptr<graphlib::GraphTraversalContext> &traversal_context,
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        &prev_inst,
    std::uint32_t &nodes_to_process,
    std::vector<const Node *> &fork_and_join_nodes)
{
    // Apply the solution to the graph solver so that we can extract the pointer to its models and
    // buffer them appropriately. Otherwise, we will be buffering a local copy of models in the solution,
    // which will eventually get discarded.

    TT_LOG_ASSERT(solution.get_selected_op_models().size() > 0, "Solution should have at least one op placed");
    for (const auto &op_model : solution.get_selected_op_models())
    {
        log_trace(
            LogBalancer,
            "RIBBON2: Graph solver set for {} with grid {}",
            op_model.buda_op_node->name(),
            op_model.grid_shape);
        graph_solver->set(op_model.buda_op_node, op_model);
    }
    OpModels *op_models = graph_solver->get_selected_op_models_for_buffering(solution.get_current_epoch_ops());

    graphlib::Graph *graph_modify = const_cast<graphlib::Graph *>(graph);
    FJBufferingResult fj_buffering;
    {
        // Generate buffering instructions if this epoch needs buffering.
        // We are scoping down FJ buffering algorithm to subgraph by setting GraphTraversalContext
        // to current epoch nodes.
        //
        std::unique_ptr<graphlib::GraphTraversalContext> epoch_traversal_context =
            graph_solver->get_graph_epoch_traversal_context(&solution.get_current_epoch_nodes());
        fj_buffering = insert_fork_join_buffering(
            graph_modify,
            nullptr /* postplacer op models */,
            op_models,
            config.device_config.get_l1_usable_size(),
            prev_inst,
            config.fork_join_tiles_treshold,
            &ribbon_buffering_factor);

        for (auto &fj : fj_buffering.nop_buffered_fjs)
        {
            // Extract all fork and join nodes of nop buffered fork-joins.
            fork_and_join_nodes.push_back(fj.first[0]);
            fork_and_join_nodes.push_back(fj.first.back());
        }
    }

    if (!std::get<0>(is_subset_of_instructions(fj_buffering.instructions, prev_inst)))
    {
        // We need to buffer, so we need to rewind the epoch and place again with buffer nodes.
        // Revert graphsolver to snapshot. Release old traversal context.
        //

        bool graph_modified = false;
        log_trace(LogBalancer, "RIBBON2: buffering required, reverting to snapshot");
        graph_solver = std::make_unique<legalizer::GraphSolver>(
            *graph_solver_epoch_snapshot);  // reset to epoch snapshot to clear the set op models
        {
            // Operate only within current epoch nodes.
            std::unique_ptr<graphlib::GraphTraversalContext> epoch_traversal_context =
                graph_solver->get_graph_epoch_traversal_context(&solution.get_current_epoch_nodes());
            graph_modified = buffer_graph(graph_modify, fj_buffering.instructions, *graph_solver);
        }

        // Reset current epoch nodes and traversal context to old state(snapshot).
        //
        traversal_context.reset();
        traversal_context = graph_solver->get_graph_traversal_context();

        if (graph_modified)
        {
            // If we added new non queue nodes we need to rerun scheduler, and re-create the ribbon solution.
            // For most ops, we should be able to find the same op model, and for the others we'll have to pick
            // a new one. Those should only be nops, though.

            std::tie(scheduled_ops, epoch_break_ops) =
                policy_run_scheduler(graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break);
            placed_op_index = 0;  // we've reset the scheduled ops
            nodes_to_process = processed_nodes.size() + scheduled_ops.size();
        }

        return false;
    }

    log_trace(LogBalancer, "RIBBON2: Applying solution with score: {}", solution.get_score());
    solution.print();

    // Create a map for quicker retrieval as we go through the schedule
    std::unordered_map<std::string, OpModel> op_name_to_model;
    for (const auto &op_model : solution.get_selected_op_models())
    {
        log_trace(LogBalancer, "RIBBON2: emplacing op {}", op_model.buda_op_node->name());
        op_name_to_model.emplace(op_model.buda_op_node->name(), op_model);
    }

    std::uint32_t solution_ops_placed = 0;
    while (placed_op_index < scheduled_ops.size())
    {
        graphlib::Node *node = graph->get_node_by_name(scheduled_ops[placed_op_index]);
        TT_ASSERT(node->node_type() == NodeType::kBudaOp);

        const graphlib::BudaOpNode *op = static_cast<graphlib::BudaOpNode *>(node);
        auto it = op_name_to_model.find(scheduled_ops[placed_op_index]);
        TT_ASSERT(it != op_name_to_model.end(), "Model for {} is missing", scheduled_ops[placed_op_index]);
        std::optional<placer::CoordRange> op_placement;
        bool sparse_dense_pair = false;

        // Special case for sparse-dense matmul pairing. We want to always place them atomically together.
        //
        if (op->is_sparse_matmul() and solution_ops_placed < solution.get_selected_op_models().size() - 1)
        {
            graphlib::Node *next_node = graph->get_node_by_name(scheduled_ops[placed_op_index + 1]);
            const graphlib::BudaOpNode *dense_matmul_op = static_cast<graphlib::BudaOpNode *>(next_node);
            auto it_dense = op_name_to_model.find(scheduled_ops[placed_op_index + 1]);

            if (can_bind_sparse_dense_matmul_pair(
                    graph,
                    op,
                    it->second,
                    dense_matmul_op,
                    it_dense->second,
                    interactive_placer,
                    true /*allow_transpose*/))
            {
                sparse_dense_pair = true;
                op_placement = interactive_placer.place_two_ops_rowwise(
                    op->name(), it->second.grid_shape, dense_matmul_op->name(), it_dense->second.grid_shape, true);

                if (op_placement.has_value())
                {
                    processed_nodes.insert(op);
                    processed_schedule.emplace_back(op->name());
                    placed_op_index++;
                    solution_ops_placed++;
                    op = dense_matmul_op;
                }
            }
        }

        if (!sparse_dense_pair)
        {
            op_placement = interactive_placer.place_op(scheduled_ops[placed_op_index], it->second.grid_shape, true);
        }

        TT_ASSERT(op_placement.has_value(), "Failed to re-place the solution on op {}", scheduled_ops[placed_op_index]);
        log_trace(LogBalancer, "RIBBON2: placed {}", scheduled_ops[placed_op_index]);
        processed_nodes.insert(op);
        processed_schedule.emplace_back(op->name());
        placed_op_index++;
        solution_ops_placed++;

        if (solution_ops_placed == solution.get_selected_op_models().size())
        {
            // We've placed all the ops in the solution, so we're done
            break;
        }
    }

    cut_graph_solver_epoch(graph, interactive_placer, *graph_solver);
    return true;
}

// Calculates the target cycles for a given epoch and ribbon size.
// Exploration will start from initial_epoch_target_cycles and calculation will be reinvoked
// until the result converges or recalc_count reaches limit.
// Side effect of this function is that GS search space will be narrowed down according to
// the ribbon size and total grid size which corresponds to the calculated target_cycles.
// (graph_solver_epoch_snapshot will be updated to reflect this)
//
int calculate_epoch_target_cycles(
    const std::uint32_t epoch,
    const Graph *graph,
    const BalancerConfig &config,
    legalizer::GraphSolver &graph_solver_epoch_snapshot,
    placer::InteractivePlacer &interactive_placer,
    placer::InteractivePlacer &ip_fittment_tester,
    const uint32_t ribbon_size,
    std::unordered_set<uint64_t> &validated_cache,
    const std::vector<std::string> &scheduled_ops,
    const std::unordered_set<std::string> &epoch_break_ops,
    const graphlib::NodeEpochType current_epoch_type,
    uint32_t placed_op_index,
    const int initial_epoch_target_cycles)
{
    int epoch_target_cycles = initial_epoch_target_cycles;
    int recalculated_epoch_target_cycles = initial_epoch_target_cycles;
    int recalc_count = 0;

    while (recalc_count < 5)
    {
        recalculated_epoch_target_cycles = calculate_target_cycles_for_ribbon_size(
            graph,
            config,
            graph_solver_epoch_snapshot,
            interactive_placer,
            ip_fittment_tester,
            ribbon_size,
            validated_cache,
            scheduled_ops,
            epoch_break_ops,
            current_epoch_type,
            placed_op_index,
            epoch_target_cycles);

        if (epoch_target_cycles == recalculated_epoch_target_cycles)
        {
            break;
        }

        epoch_target_cycles = recalculated_epoch_target_cycles;
        recalc_count++;
    }

    log_debug(
        LogBalancer, "Epoch {} setting target_cycles={} for ribbon_size={}.", epoch, epoch_target_cycles, ribbon_size);

    return epoch_target_cycles;
}

legalizer::GraphSolverSolution run_policy_ribbon2(
    graphlib::Graph const *graph,
    const BalancerConfig &config,
    legalizer::GraphSolver &graph_solver,
    std::optional<placer::PlacerSolution> &placer_solution)
{
    //
    // Ribbon2 policy
    //
    // Balancer works epoch by epoch, and tries to optimize each epoch for the maximum matmul utilization. It explores
    // all possible ribbon sizes for each epoch to generate an initial set of solutions. Grids are picked based on some
    // heuristics, trying to stick to ribbon size, fit in prologue, and so on.
    //
    // Then, each of the solutions is optimized by iteratively bumping up grids of the slowest ops, as long as that
    // improves the utilization of the epoch. Once this is exhausted, all solutions are compared and the highest
    // utilization is picked as the best for the epoch.
    //
    // At that point, fork-join buffering is added, and epoch is applied to graph solver and interactive placer.
    //
    // The utilization of the epoch is based on sum of matmul utilizations on each core, where the utilization is
    // calculated as "theoretical lowest cycles / longest op in epoch cycles", where theoretical cycles in the
    // number of cycles it would take at 100% utilization.
    //
    // Limitations:
    //
    // - Ribbon2 does not force sparse and dense matmuls to be on the same epoch. This was previous done as a
    //   performance herustic, but it is not necessary in most situations. Accurate modeling of DRAM bandwidths could
    //   allow balancer to make an optimal decision without this heuristic.
    // - Because the optimal solution will have a much more random selection of grids vs. a clean ribbon, the blob sizes
    //   are likely to grow much larger for some ops. For example, resnet epoch 1, at the moment, needs 77KB of extra
    //   blob space, mobilenet v2 330kb!  However, once backend is given this space, resnet is significantly faster
    //   than with the original ribbon. Going forward, accurate tile modeling (currently worked on by Nick) will allow
    //   us to predict blob sizes better and add space only to cores that need it (or avoid combinations that create
    //   large blobs).
    // - Only one ribbon size is set per epoch. Having multiple ribbon sizes per epoch could explode the search space,
    //   and make the algorithm impractical. Because ribbon size is only used to seed the initial solution before
    //   optimization (which is free to change it), this appears to work well enough in limited testing.
    // - Success is heavily dependent on accurate modeling of the backend cycles. This isn't necessarily a limitation,
    //   of the algorithm itself, but because modeling is not completely accurate in all situations, Ribbon2 can
    //   make bad choices. Resnet epoch0 is a good example, where sparse matmuls are estimate to run 5-6x slower than
    //   they actually do, and the chosen solution is far from ideal.
    // - Each epoch takes longer to solve, due to the nature of the algorithm. None of it is particularly compute-
    //   intensive, but for a very large model, it could add up.
    // - Ribbon2 gives up on optimizing an epoch after changes don't increase utilization. However, it could be a case
    //   of a local minimum, and further iterations could continue to optimize. However, letting it always run for 10+
    //   iterations would add a lot to the runtime, and many of those searches will not be fruitful. Some kind of a
    //   heuristic to decide when to continue would be helpful.
    // - Ribbon2 arbitrarily stops after 10 iterations of optimizations of a particular solution. Further testing is
    //   needed to see if this is reasonable.
    //
    // Future improvements:
    //
    // - Convolution fracturing decision is made before Ribbon2 runs. However, letting the balancer determine which
    //   convolutions would benefit from fracturing would allow us to make better decisions.
    // - We could apply fork join buffering on each candidate solution, but due to the added complexity of graph changes
    //   and cuts, it is likely going to slow down the alogorithm too much to make it practical. Evaluation is needed to
    //   see if this would yield better solutions.
    // - Seed the initial epoch solution with multiple ribbon sizes and queues to break between dimension changes.
    // - This is a greedy algorithm which tries to optimize each epoch as it goes. However, choices made in current
    //   epoch can affect future ones. Cross-epoch search, with epoch back-tracking is likely to yield better results
    //   for some models.
    //

    log_info(LogBalancer, "Starting Ribbon2 balancing");
    placer::InteractivePlacer interactive_placer(graph, config);
    placer::InteractivePlacer ip_fittment_tester(graph, config);
    std::unordered_set<string> epoch_break_ops;
    scheduler::Schedule scheduled_ops;
    graphlib::NodeEpochType current_epoch_type = NodeEpochType::Forward;
    std::vector<const tt::graphlib::Node *> pre_buffering_epoch_nodes;
    const tt::ordered_map<InsInstructionUniqueId, std::shared_ptr<InsertionInstruction>, InsInstructionUniqueIdHash>
        prev_inst;
    std::unordered_set<const tt::graphlib::Node *> processed_nodes;
    std::vector<tt::scheduler::Schedule> op_names_to_epoch_break = config.op_names_to_epoch_break;
    tt::scheduler::Schedule processed_schedule;
    std::vector<EpochSolution> applied_solutions;

    std::unique_ptr<legalizer::GraphSolver> graph_solver_main = std::make_unique<legalizer::GraphSolver>(graph_solver);
    std::unique_ptr<graphlib::GraphTraversalContext> traversal_context =
        graph_solver_main->get_graph_traversal_context();
    std::tie(scheduled_ops, epoch_break_ops) =
        policy_run_scheduler(graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break);

    std::unordered_set<std::uint64_t> validated_cache;  // list of op model IDs that have been validated to be ok, so we
                                                        // don't have to validate them again

    // In case of recompile, we can offset the target cycles to get a different solution.
    const int target_cycles = env_as<int>("PYBUDA_RIBBON_TARGET_CYCLES", 95000) + config.target_cycles_offset;
    const int max_iterations = env_as<int>("PYBUDA_RIBBON2_OPTIMIZATION_ITERATIONS", 0);

    TT_ASSERT(config.op_names_to_chip_break.size() == 0, "Ribbon2 policy does not process chip breaks");

    std::uint32_t epoch = 0;
    bool done = false;
    std::uint32_t placed_op_index = 0;
    std::uint32_t nodes_to_process = scheduled_ops.size();
    std::unique_ptr<legalizer::GraphSolver>
        pre_buffered_graph_snapshot;                       // snapshot before any fork-join buffering
                                                           // graph modifications were made for the current epoch
    std::unique_ptr<EpochSolution> pre_buffered_solution;  // current epoch solution before the last fork-join
                                                           // buffering attempt - used to check if added buffering
                                                           // caused any fork-join to split accross epochs
    std::vector<const Node *>
        fork_and_join_nodes;  // fork and join nodes of every nop-buffered fork-join in current epoch.
    bool epoch_breaks_added = false;

    graph_solver_main->invalidate_suboptimal_op_models(
        legalizer::MatmulSparseDenseGridPairing | legalizer::DenseMatmulPrologue | legalizer::DenseMatmulBetterUkt);

    while (!done)
    {
        // Try placing an epoch for each ribbon size, and figure out the score for each
        std::vector<EpochSolution> solutions;
        std::exception_ptr first_error = nullptr;
        bool first_error_is_fatal = false;

        // Per-epoch overrides
        const int force_target_cycles =
            env_as<int>((std::string("PYBUDA_RIBBON2_TARGET_CYCLES_FOR_EPOCH") + std::to_string(epoch)).c_str(), 0);
        int epoch_target_cycles = (force_target_cycles != 0) ? force_target_cycles : target_cycles;

        const int force_optimization_iterations = env_as<int>(
            (std::string("PYBUDA_RIBBON2_OPTIMIZATION_ITERATIONS_FOR_EPOCH") + std::to_string(epoch)).c_str(), -1);
        const int epoch_max_iterations =
            (force_optimization_iterations != -1) ? force_optimization_iterations : max_iterations;

        const int force_ribbon =
            env_as<int>((std::string("PYBUDA_RIBBON2_RIBBON_FOR_EPOCH") + std::to_string(epoch)).c_str(), 0);

        log_debug(
            LogBalancer,
            "Epoch {} settings: target_cycles={}, max_iterations={}, force_ribbon={}",
            epoch,
            epoch_target_cycles,
            epoch_max_iterations,
            force_ribbon);

        for (std::uint32_t ribbon_size = 1; ribbon_size <= (std::uint32_t)config.device_config.grid_size.r;
             ribbon_size++)
        {
            // Per epoch ribbon size override
            if (force_ribbon != 0 && (int)ribbon_size != force_ribbon)
            {
                continue;
            }

            auto graph_solver_epoch_snapshot = std::make_unique<legalizer::GraphSolver>(*graph_solver_main);
            std::vector<OpModel> selected_models;

            try
            {
                std::unique_ptr<graphlib::GraphTraversalContext> epoch_snapshot_traversal_context =
                    graph_solver_epoch_snapshot->get_graph_traversal_context();

                if (force_target_cycles == 0 and env_as<bool>("PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES", false))
                {
                    // Legacy target_cycles are passed in as initial seed for calculation.
                    // This value has modest impact on outcome as
                    // value too low may limit exploration while value too high may lead to suboptimal solutions with
                    // slower op model preference. Note that value returned by calculation can be lower than
                    // target_cycles, this is mostly impacted by ribbon_size and other factors from op_model preference
                    // function. Generally it seems that value in range 80-120k provide good span of epoch estimates and
                    // results for current architecture but we can tweak this further in the future.
                    //
                    epoch_target_cycles = calculate_epoch_target_cycles(
                        epoch,
                        graph,
                        config,
                        *graph_solver_epoch_snapshot,
                        interactive_placer,
                        ip_fittment_tester,
                        ribbon_size,
                        validated_cache,
                        scheduled_ops,
                        epoch_break_ops,
                        current_epoch_type,
                        placed_op_index,
                        target_cycles);
                }

                // Pick op models
                for (std::uint32_t op_index = placed_op_index; op_index < scheduled_ops.size(); op_index++)
                {
                    graphlib::Node *node = graph->get_node_by_name(scheduled_ops[op_index]);
                    if (node->node_type() != NodeType::kBudaOp)
                        continue;

                    const graphlib::BudaOpNode *op = static_cast<const graphlib::BudaOpNode *>(node);

                    // check if there is a forced break at this op
                    bool new_epoch = (op_index > placed_op_index) && ((epoch_break_ops.count(node->name()) > 0) ||
                                                                      (current_epoch_type != op->get_epoch_type()));

                    if (!new_epoch)
                    {
                        // Pick the best op model.
                        //
                        const OpModel &selected_op_model = select_best_op_model_ribbon(
                            *graph_solver_epoch_snapshot,
                            op,
                            ribbon_size,
                            config,
                            graph,
                            validated_cache,
                            epoch_target_cycles);
                        log_trace(
                            LogBalancer,
                            "RIBBON2: (epoch={}, op_index={}, ribbon={}) {} best grid: {}, cycles: {} ",
                            epoch,
                            op_index,
                            ribbon_size,
                            node->name(),
                            selected_op_model.grid_shape,
                            get_limiter_cycles(selected_op_model, graph, config));
                        std::optional<placer::CoordRange> op_placement;
                        bool sparse_dense_pair = false;
                        bool op_already_set = false;

                        // Special case for sparse matmuls. Try to pair them with the next op if preferable(sparse-dense
                        // like pairs, see should_pair_with_sparse()).
                        //
                        if (op->is_sparse_matmul() and op_index < scheduled_ops.size() - 1)
                        {
                            graphlib::Node *next_node = graph->get_node_by_name(scheduled_ops[op_index + 1]);
                            if (next_node->node_type() == NodeType::kBudaOp)
                            {
                                const graphlib::BudaOpNode *dense_matmul_op =
                                    static_cast<const graphlib::BudaOpNode *>(next_node);
                                if (dense_matmul_op->should_pair_with_sparse(op, graph))
                                {
                                    graph_solver_epoch_snapshot->set(op, selected_op_model);
                                    op_already_set = true;

                                    const OpModel &selected_op_model_dense = select_best_op_model_ribbon(
                                        *graph_solver_epoch_snapshot,
                                        dense_matmul_op,
                                        ribbon_size,
                                        config,
                                        graph,
                                        validated_cache,
                                        epoch_target_cycles);

                                    // Place sparse and dense matmul paired and in the same epoch if possible.
                                    //
                                    op_placement = place_sparse_dense_pair(
                                        op,
                                        &selected_op_model,
                                        dense_matmul_op,
                                        &selected_op_model_dense,
                                        interactive_placer,
                                        ip_fittment_tester,
                                        sparse_dense_pair);

                                    // Pair has been placed, mark opmodels, and skip next op as it is already selected
                                    // and set.
                                    //
                                    if (op_placement.has_value())
                                    {
                                        selected_models.push_back(selected_op_model);
                                        selected_models.push_back(selected_op_model_dense);
                                        graph_solver_epoch_snapshot->set(dense_matmul_op, selected_op_model_dense);
                                        op_index++;
                                    }
                                }
                            }
                        }

                        if (!sparse_dense_pair)
                        {
                            op_placement = interactive_placer.place_op(op->name(), selected_op_model.grid_shape, true);
                        }

                        new_epoch = !op_placement.has_value() || (op_index == scheduled_ops.size() - 1);

                        if (op_placement.has_value())
                        {
                            if (!sparse_dense_pair)
                            {
                                selected_models.push_back(selected_op_model);
                                if (!op_already_set)
                                {
                                    graph_solver_epoch_snapshot->set(op, selected_op_model);
                                }
                            }
                        }
                        else
                        {
                            log_trace(LogBalancer, "RIBBON2: Doesn't fit, starting new epoch");
                        }
                    }

                    if (new_epoch)
                    {
                        TT_ASSERT(!new_epoch || selected_models.size() > 0);
                        // Record the solution
                        EpochSolution new_solution(ribbon_size, &config, selected_models, graph, epoch_target_cycles);

                        // Check if the same solution was provided by another ribbon
                        bool found_same_solution = false;
                        for (const auto &s : solutions)
                        {
                            if ((s.get_score() != new_solution.get_score()) ||
                                (s.get_selected_op_models().size() != selected_models.size()))
                                continue;

                            bool same = true;
                            for (std::size_t i = 0; i < s.get_selected_op_models().size(); i++)
                            {
                                if (!(s.get_selected_op_models()[i].id == selected_models[i].id))
                                {
                                    same = false;
                                    break;
                                }
                            }

                            if (same)
                            {
                                found_same_solution = true;
                                break;
                            }
                        }
                        if (!found_same_solution)
                        {
                            solutions.push_back(new_solution);
                        }

                        interactive_placer.rewind_epoch();
                        break;
                    }
                }
            }
            catch (const BalancerError &e)
            {
                log_debug(
                    LogBalancer,
                    "Encountered BalancerException while trying ribbon size {}: {}",
                    ribbon_size,
                    e.what());

                bool fatal_exception = std::holds_alternative<balancer::BalancerError::Fatal>(e.type);
                if ((first_error == nullptr) || (first_error_is_fatal && !fatal_exception))
                {
                    first_error = std::current_exception();
                    first_error_is_fatal = fatal_exception;
                }

                interactive_placer.rewind_epoch();
            }
        }

        if (solutions.size() == 0)
        {
            log_debug(LogBalancer, "No solution found, throwing first error encountered");
            TT_ASSERT(first_error != nullptr);
            std::rethrow_exception(first_error);
        }

        log_trace(LogBalancer, "RIBBON2: (epoch={}) number of solutions: {}", epoch, solutions.size());
        auto best_solution = solutions[0];
        for (auto &s : solutions)
        {
            try
            {
                auto optimized_solution = optimize_solution(
                    s, *graph_solver_main, interactive_placer, graph, validated_cache, epoch_max_iterations);
                if (optimized_solution.get_score() > best_solution.get_score())
                {
                    best_solution = optimized_solution;
                }
            }
            catch (const BalancerError &e)
            {
                log_debug(LogBalancer, "Encountered BalancerException while optimizing solution: {}", e.what());
                // Use the unoptimized solution
                if (s.get_score() > best_solution.get_score())
                {
                    best_solution = s;
                }
            }
        }

        bool rescheduled = handle_fork_join_nop_overflow(
            graph,
            config,
            op_names_to_epoch_break,
            best_solution,
            pre_buffered_solution,
            graph_solver_main,
            pre_buffered_graph_snapshot,
            epoch_break_ops,
            placed_op_index,
            scheduled_ops,
            processed_nodes,
            processed_schedule,
            traversal_context,
            nodes_to_process,
            epoch,
            fork_and_join_nodes,
            epoch_breaks_added);

        if (rescheduled)
        {
            // We have a new schedule, restart search.
            continue;
        }

        // Insert fj buffering as needed, and apply the solution to the main graph solver and placer
        std::unique_ptr<legalizer::GraphSolver> graph_solver_snapshot =
            std::make_unique<legalizer::GraphSolver>(*graph_solver_main);
        bool applied = apply_solution(
            graph,
            config,
            op_names_to_epoch_break,
            best_solution,
            graph_solver_main,
            graph_solver_snapshot,
            interactive_placer,
            epoch_break_ops,
            scheduled_ops,
            processed_nodes,
            processed_schedule,
            placed_op_index,
            traversal_context,
            prev_inst,
            nodes_to_process,
            fork_and_join_nodes);

        if (applied)
        {
            log_debug(
                LogBalancer,
                "RIBBON2: (epoch={}) applied solution with score: {} ribbon_size: {} pipeline_cycles: {}",
                epoch,
                best_solution.get_score(),
                best_solution.get_ribbon_size(),
                best_solution.get_pipeline_cycles());

            if (placed_op_index >= scheduled_ops.size())
            {
                Logger<kLoggerABI>::get().log_level_type(
                    Logger<kLoggerABI>::Level::Info, LogBalancer, "Balancing 100% completed!");
                break;
            }
            else
            {
                Logger<kLoggerABI>::get().log_level_type(
                    Logger<kLoggerABI>::Level::Info,
                    LogBalancer,
                    "Balancing {}% complete.",
                    processed_nodes.size() * 100 / nodes_to_process);
            }

            epoch++;

            graphlib::Node *next_node = graph->get_node_by_name(scheduled_ops[placed_op_index]);
            current_epoch_type = next_node->get_epoch_type();
            interactive_placer.next_epoch(current_epoch_type);

            if (epoch_breaks_added)
            {
                // Remove previously added epoch breaks, since we have successfully applied the solution.
                //
                // We also need to remove coresponding 'epoch break op' generated by the scheduler based on
                // epoch breaks we've added (in op_names_to_epoch_break). This is done because the chosen
                // epoch solution might not contain all nodes up to 'epoch break op' - and in that case
                // the next epoch created will be broken again on the 'epoch break op', which is not necessary
                // in our case and can cause perf degradation.
                op_names_to_epoch_break.pop_back();
                epoch_break_ops = placer::lowering::tag_ops_for_epoch_break(
                    config.device_config.arch_name,
                    op_names_to_epoch_break,
                    config.op_names_to_chip_break,
                    scheduled_ops,
                    graph,
                    true /* use_interactive_placer */);
                epoch_breaks_added = false;
            }

            pre_buffered_graph_snapshot.reset();
            pre_buffered_solution.reset();
            fork_and_join_nodes.clear();
            applied_solutions.push_back(best_solution);
        }
        else
        {
            if (pre_buffered_graph_snapshot.get() == nullptr)
            {
                // The solution hasn't been applied, which means fork-join buffering has been added for this epoch.
                // Save pre buffered state in case we need to revert to it.
                log_debug(LogBalancer, "Saving pre_buffered graph snapshot.");
                pre_buffered_graph_snapshot = std::move(graph_solver_snapshot);
            }

            pre_buffered_solution = std::make_unique<EpochSolution>(best_solution);
        }
    }

    placer_solution = interactive_placer.commit();
    placer_solution.value().fork_join_buffered = true;
    validate_solution(scheduled_ops, placer_solution.value());
    score_solution(applied_solutions, config.device_config);

    return graph_solver_main->finish();
}
}  // namespace tt::balancer
