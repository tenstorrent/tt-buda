// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "balancer/policies/policy_ribbon.hpp"
#include "balancer/types.hpp"
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

std::optional<OpModel> get_closest_op_model_conservative(
    const legalizer::GraphSolver &graph_solver_snapshot,
    const OpModel &op_model_target,
    EpochSolution &current_solution,
    const int target_limiter_cycles,
    const BalancerConfig *balancer_config,
    const graphlib::Graph *graph,
    const int ribbon_size)
{
    std::optional<OpModel> closest_model = std::nullopt;

    for (const auto &op_model : graph_solver_snapshot.at(op_model_target.buda_op_node))
    {
        if (op_model == op_model_target)
        {
            closest_model = op_model;
            return closest_model;
        }
        else if (op_model.is_similar(op_model_target))
        {
            closest_model = op_model;
            return closest_model;
        }
    }

    int best_delta = std::numeric_limits<int>::max();

    for (const auto &op_model : graph_solver_snapshot.at(op_model_target.buda_op_node))
    {
        if (ribbon_size == op_model.grid_shape.r || op_model_target.grid_shape.r == op_model.grid_shape.r)
        {
            int my_delta = target_limiter_cycles - get_limiter_cycles(
                                                       op_model,
                                                       graph,
                                                       *balancer_config,
                                                       current_solution.get_dram_access_core_count(),
                                                       current_solution.get_pcie_access_core_count(),
                                                       &current_solution.get_current_epoch_nodes(),
                                                       false /* invalidate_cache */,
                                                       &current_solution.current_epoch_op_models);

            if (my_delta >= 0)
            {
                if (my_delta < best_delta)
                {
                    closest_model = op_model;
                    best_delta = my_delta;
                }
                else if (my_delta == best_delta)
                {
                    // Prefer the same shape
                    //
                    if (op_model_target.grid_shape == op_model.grid_shape)
                    {
                        closest_model = op_model;
                    }
                }
            }
        }
    }

    return closest_model;
}

bool operand_of_linked_output_node(const graphlib::Graph *graph, const graphlib::Node *node)
{
    for (const graphlib::Node *user_node : graph->data_users(node))
    {
        if (user_node->node_type() == graphlib::NodeType::kOutput and is_linked_queue(graph, user_node))
        {
            return true;
        }
    }

    return false;
}

// Optimize a solution by iteratively bumping up grids of the slowest ops, as long as that
// improves the utilization of the epoch.
// Conservative version which tries to stick to the same ribbon and same OP count in epoch.
//
EpochSolution optimize_solution_conservative(
    const EpochSolution &solution,
    const legalizer::GraphSolver &graph_solver,
    placer::InteractivePlacer &interactive_placer,
    const graphlib::Graph *graph,
    std::uint32_t max_iterations)
{
    EpochSolution best_solution = solution;

    std::uint32_t iterations = 0;
    std::uint32_t bad_iterations = 0;
    const BalancerConfig *balancer_config = solution.get_balancer_config();
    std::unordered_set<std::uint64_t> blacklisted_models;

    // If all cores are all used no point in optimization.
    //
    max_iterations = solution.get_used_cores_ratio() < 0.98 ? max_iterations : 0;

    if (max_iterations > 0)
    {
        log_trace(
            LogBalancer,
            "RIBBON2: optimize solution, score {}, pipeline cycles {} coming in:",
            solution.get_score(),
            solution.get_pipeline_cycles());
        solution.print();
    }

    while ((bad_iterations < 3) && (iterations < max_iterations))
    {
        std::vector<std::uint64_t> models_used_iterration;

        // Find the slowest cycle count
        float slowest_cycles = 0;
        for (const OpModel &op_model : best_solution.get_selected_op_models())
        {
            float cycles = get_limiter_cycles(
                op_model,
                graph,
                *balancer_config,
                best_solution.get_dram_access_core_count(),
                best_solution.get_pcie_access_core_count(),
                &best_solution.get_current_epoch_nodes(),
                false /* invalidate_cache */,
                &best_solution.get_current_epoch_op_models());
            if (cycles > slowest_cycles)
                slowest_cycles = cycles;
        }

        std::unique_ptr<legalizer::GraphSolver> graph_solver_snapshot =
            std::make_unique<legalizer::GraphSolver>(graph_solver);
        const OpModels *selected_op_models = &graph_solver_snapshot->get_selected_op_models();
        std::unique_ptr<graphlib::GraphTraversalContext> opt_snapshot_traversal_context =
            graph_solver_snapshot->get_graph_traversal_context();
        EpochSolution new_solution = best_solution;
        float target_cycles = 0.9 * slowest_cycles;
        log_trace(LogBalancer, "RIBBON2: target_cycles = {}", target_cycles);
        bool bad_solution = false;

        // Now go through the models, and bump up the ones that are slowest.
        //
        for (std::size_t op_index = 0; op_index < new_solution.get_selected_op_models().size(); op_index++)
        {
            const OpModel &source_op_model = best_solution.get_selected_op_models()[op_index];

            // Mitigation for linked output nodes. We don't want to bump up the grid of the linked output node because
            // of higher chance of op_model mismatch on OPs feeding the fake output.
            //
            if (operand_of_linked_output_node(graph, source_op_model.buda_op_node))
            {
                continue;
            }

            int cycles = get_limiter_cycles(
                source_op_model,
                graph,
                *balancer_config,
                best_solution.get_dram_access_core_count(),
                best_solution.get_pcie_access_core_count(),
                &best_solution.get_current_epoch_nodes(),
                false /* invalidate_cache */,
                &best_solution.current_epoch_op_models);
            if (cycles <= target_cycles)
            {
                log_trace(LogBalancer, "RIBBON2: op {} is fast enough", source_op_model.buda_op_node->name());
                std::optional<OpModel> closest_model = get_closest_op_model_conservative(
                    *graph_solver_snapshot,
                    source_op_model,
                    new_solution,
                    cycles,
                    balancer_config,
                    graph,
                    solution.get_ribbon_size());
                if (closest_model.has_value() and
                    closest_model.value().grid_shape.volume() - source_op_model.grid_shape.volume() <=
                        new_solution.get_free_cores())
                {
                    graph_solver_snapshot->set(source_op_model.buda_op_node, closest_model.value());
                    if (!(closest_model.value() == source_op_model))
                    {
                        log_trace(
                            LogBalancer,
                            "RIBBON2: had to change the grid to {} with cycles {}",
                            closest_model.value().grid_shape,
                            get_limiter_cycles(
                                closest_model.value(),
                                graph,
                                *balancer_config,
                                new_solution.get_dram_access_core_count(),
                                new_solution.get_pcie_access_core_count(),
                                &new_solution.get_current_epoch_nodes(),
                                false /* invalidate_cache */,
                                &new_solution.current_epoch_op_models));
                        new_solution.update_model(op_index, closest_model.value());
                    }
                }
                else
                {
                    log_trace(
                        LogBalancer, "RIBBON2: no closest model found for {}", source_op_model.buda_op_node->name());
                    bad_solution = true;
                    break;
                }
            }
            else
            {
                // Bump up the grid.
                //
                log_trace(
                    LogBalancer, "RIBBON2: op {} is too slow, bumping up grid", source_op_model.buda_op_node->name());
                std::optional<OpModel> new_op_model = std::nullopt;

                for (const OpModel &op_model : graph_solver_snapshot->at(source_op_model.buda_op_node))
                {
                    log_trace(
                        LogBalancer,
                        "RIBBON2: trying grid {} with cycles {}",
                        op_model.grid_shape,
                        get_limiter_cycles(
                            op_model,
                            graph,
                            *balancer_config,
                            new_solution.get_dram_access_core_count(),
                            new_solution.get_pcie_access_core_count(),
                            &new_solution.get_current_epoch_nodes(),
                            false /* invalidate_cache */,
                            &new_solution.current_epoch_op_models));

                    if (op_model.grid_shape.volume() - source_op_model.grid_shape.volume() >
                        new_solution.get_free_cores())
                        continue;

                    if (blacklisted_models.find(op_model.id.id) != blacklisted_models.end())
                    {
                        log_trace(LogBalancer, "RIBBON2: skipping blacklisted op_model");
                        continue;
                    }

                    if (op_model.grid_shape.r != (int)new_solution.get_ribbon_size())
                        continue;

                    if (get_limiter_cycles(
                            op_model,
                            graph,
                            *balancer_config,
                            new_solution.get_dram_access_core_count(),
                            new_solution.get_pcie_access_core_count(),
                            &new_solution.get_current_epoch_nodes(),
                            false /* invalidate_cache */,
                            &new_solution.current_epoch_op_models) >= slowest_cycles)
                        continue;

                    if (!new_op_model.has_value() || (get_limiter_cycles(
                                                          op_model,
                                                          graph,
                                                          *balancer_config,
                                                          new_solution.get_dram_access_core_count(),
                                                          new_solution.get_pcie_access_core_count(),
                                                          &new_solution.get_current_epoch_nodes(),
                                                          false /* invalidate_cache */,
                                                          selected_op_models) <
                                                      get_limiter_cycles(
                                                          new_op_model.value(),
                                                          graph,
                                                          *balancer_config,
                                                          new_solution.get_dram_access_core_count(),
                                                          new_solution.get_pcie_access_core_count(),
                                                          &new_solution.get_current_epoch_nodes(),
                                                          false /* invalidate_cache */,
                                                          selected_op_models)))
                    {
                        new_op_model = op_model;
                        log_trace(
                            LogBalancer,
                            "RIBBON2: setting new grid for {}: {} with cycles {}",
                            op_model.buda_op_node->name(),
                            op_model.grid_shape,
                            get_limiter_cycles(
                                op_model,
                                graph,
                                *balancer_config,
                                new_solution.get_dram_access_core_count(),
                                new_solution.get_pcie_access_core_count(),
                                &new_solution.get_current_epoch_nodes(),
                                false /* invalidate_cache */,
                                &new_solution.current_epoch_op_models));
                    }
                }

                // If we found a larger grid, then use it.
                //
                if (new_op_model.has_value())
                {
                    log_trace(
                        LogBalancer,
                        "RIBBON2: bumping up {} from {} to {}",
                        source_op_model.buda_op_node->name(),
                        source_op_model.grid_shape,
                        new_op_model->grid_shape);
                    new_solution.update_model(op_index, new_op_model.value());
                    graph_solver_snapshot->set(source_op_model.buda_op_node, new_op_model.value());
                    models_used_iterration.push_back(
                        new_op_model.value().id.id);  // record in case this bump ended up being bad
                }
                else
                {
                    // We haven't found anything better, set the same (or closest legal).
                    //
                    std::optional<OpModel> closest_model = get_closest_op_model_conservative(
                        *graph_solver_snapshot,
                        source_op_model,
                        new_solution,
                        cycles,
                        balancer_config,
                        graph,
                        solution.get_ribbon_size());

                    if (!closest_model.has_value())
                    {
                        bad_solution = true;
                        break;
                    }

                    new_solution.update_model(op_index, closest_model.value());
                    graph_solver_snapshot->set(source_op_model.buda_op_node, closest_model.value());
                }
            }
        }

        if (bad_solution)
        {
            blacklisted_models.insert(models_used_iterration.begin(), models_used_iterration.end());
            bad_iterations++;
            iterations++;
            continue;
        }

        // We need to place this new solution to see how much of it actually fits.
        //
        std::size_t placed_ops = 0;
        for (std::size_t i = 0; i < new_solution.get_selected_op_models().size(); i++)
        {
            const OpModel &op_model = new_solution.get_selected_op_models()[i];
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

        // Rewind, we were just testing what fits.
        //
        interactive_placer.rewind_epoch();

        if (new_solution.get_pipeline_cycles() < best_solution.get_pipeline_cycles() and
            placed_ops == solution.get_selected_op_models().size())
        {
            best_solution = new_solution;
            bad_iterations = 0;
            log_trace(LogBalancer, "RIBBON2: improved to {}", best_solution.get_pipeline_cycles());
        }
        else
        {
            blacklisted_models.insert(models_used_iterration.begin(), models_used_iterration.end());
            bad_iterations++;
            log_trace(LogBalancer, "RIBBON2: solution got worse, bad iterations in a row = {}", bad_iterations);
        }
        iterations++;
    }

    if (best_solution.get_pipeline_cycles() < solution.get_pipeline_cycles())
    {
        log_debug(
            LogBalancer,
            "RIBBON2: optimized solution with score {} from base solution with score {}. Pipeline cycles changed from "
            "{} to {}.",
            best_solution.get_score(),
            solution.get_score(),
            solution.get_pipeline_cycles(),
            best_solution.get_pipeline_cycles());
        best_solution.print();
    }

    return best_solution;
}

bool handle_fork_join_nop_overflow(
    graphlib::Graph const *graph,
    const BalancerConfig &config,
    std::vector<std::vector<std::string>> &op_names_to_epoch_break,
    EpochSolution &solution,
    std::unordered_set<graphlib::NodeId> &pre_buffered_ops,
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
    std::unordered_set<graphlib::NodeId> &fork_and_join_nodes,
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

    if (fork_and_join_nodes.empty())
    {
        return false;
    }

    // Fork-join buffering for this epoch was added in previous iteration.
    // Check if added buffering caused any of the fork-joins to split into two epochs.
    // If that is the case, there is no point in keeping the added nops for buffering.

    // Get all ops in current epoch.
    std::unordered_set<graphlib::NodeId> ops_in_curr_epoch;
    for (const auto &op_model : solution.get_selected_op_models())
    {
        ops_in_curr_epoch.insert(op_model.buda_op_node->id());
    }

    // Check if all fork and join nodes are in this epoch.
    bool needs_epoch_break = false;
    for (auto node_id : fork_and_join_nodes)
    {
        if (!ops_in_curr_epoch.count(node_id))
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
    for (const auto &op_id : pre_buffered_ops)
    {
        if (!ops_in_curr_epoch.count(op_id))
        {
            const Node *node = graph->node_by_id(op_id);
            epoch_break.push_back(node->name());
        }
    }

    TT_ASSERT(epoch_break.size() > 0, "We should have at least one op to break the epoch");

    if (epoch_breaks_added)
    {
        op_names_to_epoch_break.pop_back();
    }

    op_names_to_epoch_break.push_back(epoch_break);
    epoch_breaks_added = true;
    pre_buffered_ops.clear();

    // Since we can no longer fit all of the pre-buffered ops on a single epoch,
    // undo the buffering, reschedule everything (with explicit epoch breaks added) and continue to search for a
    // solution. This takes care of cases where we leave unnecessary fork-join buffering which spans multiple epochs.
    graph_solver = std::make_unique<legalizer::GraphSolver>(
        *pre_buffered_graph_snapshot);  // reset to epoch snapshot to clear the set op models

    traversal_context.reset();
    traversal_context = graph_solver->get_graph_traversal_context();

    std::tie(scheduled_ops, epoch_break_ops) =
        policy_run_scheduler(graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break);

    placed_op_index = 0;
    nodes_to_process = processed_nodes.size() + scheduled_ops.size();
    fork_and_join_nodes.clear();

    return true;
}

bool buffer_fork_joins(
    Graph *graph,
    const BalancerConfig &config,
    OpModels *op_models,
    std::vector<std::vector<std::string>> &op_names_to_epoch_break,
    EpochSolution &solution,
    std::unique_ptr<legalizer::GraphSolver> &graph_solver,
    std::unique_ptr<legalizer::GraphSolver> &graph_solver_epoch_snapshot,
    std::unordered_set<std::string> &epoch_break_ops,
    std::uint32_t &placed_op_index,
    scheduler::Schedule &scheduled_ops,
    const std::unordered_set<const Node *> &processed_nodes,
    const tt::scheduler::Schedule &processed_schedule,
    std::unique_ptr<graphlib::GraphTraversalContext> &traversal_context,
    std::uint32_t &nodes_to_process,
    std::unordered_set<graphlib::NodeId> &fork_and_join_nodes,
    InsertionInstructionMap &prev_inst)
{
    FJBufferingResult fj_buffering;
    {
        // Generate buffering instructions if this epoch needs buffering.
        // We are scoping down FJ buffering algorithm to subgraph by setting GraphTraversalContext
        // to current epoch nodes.
        //
        std::unique_ptr<graphlib::GraphTraversalContext> epoch_traversal_context =
            graph_solver->get_graph_epoch_traversal_context(&solution.get_current_epoch_nodes());
        fj_buffering = insert_fork_join_buffering(
            graph,
            nullptr /* postplacer op models */,
            op_models,
            config.device_config.get_l1_usable_size(),
            prev_inst,
            &ribbon_buffering_factor);

        for (auto &fj : fj_buffering.fjs_buffered_with_instr)
        {
            // Extract all fork and join nodes of both nop and queue buffered fork-joins.
            fork_and_join_nodes.insert(fj.first[0]->id());
            fork_and_join_nodes.insert(fj.first.back()->id());
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
            graph_modified = buffer_graph(graph, fj_buffering.instructions, *graph_solver);
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

        prev_inst = fj_buffering.instructions;
        return true;
    }

    return false;
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
    InsertionInstructionMap &prev_inst,
    std::uint32_t &nodes_to_process,
    std::unordered_set<graphlib::NodeId> &fork_and_join_nodes)
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
    bool graph_modified = buffer_fork_joins(
        graph_modify,
        config,
        op_models,
        op_names_to_epoch_break,
        solution,
        graph_solver,
        graph_solver_epoch_snapshot,
        epoch_break_ops,
        placed_op_index,
        scheduled_ops,
        processed_nodes,
        processed_schedule,
        traversal_context,
        nodes_to_process,
        fork_and_join_nodes,
        prev_inst);

    if (graph_modified)
    {
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

EpochSolution find_solution_for_epoch(
    const std::unique_ptr<legalizer::GraphSolver> &graph_solver_main,
    const graphlib::Graph *graph,
    const BalancerConfig &config,
    placer::InteractivePlacer &interactive_placer,
    placer::InteractivePlacer &ip_fittment_tester,
    const std::vector<std::string> &scheduled_ops,
    const std::unordered_set<std::string> &epoch_break_ops,
    const graphlib::NodeEpochType current_epoch_type,
    const std::uint32_t placed_op_index,
    const std::uint32_t epoch)
{
    const int max_conservative_opt_iterations = env_as<int>("PYBUDA_RIBBON2_CONSERVATIVE_OPTIMIZATION_ITERATIONS", 10);

    // In case of recompile, we can offset the target cycles to get a different solution.
    const int target_cycles = env_as<int>("PYBUDA_RIBBON_TARGET_CYCLES", 95000) + config.target_cycles_offset;

    // Try placing an epoch for each ribbon size, and figure out the score for each
    std::vector<EpochSolution> solutions;
    std::exception_ptr first_error = nullptr;
    bool first_error_is_fatal = false;

    // Per-epoch overrides
    const int force_target_cycles =
        env_as<int>((std::string("PYBUDA_RIBBON2_TARGET_CYCLES_FOR_EPOCH") + std::to_string(epoch)).c_str(), 0);
    int epoch_target_cycles = (force_target_cycles != 0) ? force_target_cycles : target_cycles;

    const int force_ribbon =
        env_as<int>((std::string("PYBUDA_RIBBON2_RIBBON_FOR_EPOCH") + std::to_string(epoch)).c_str(), 0);

    log_debug(
        LogBalancer, "Epoch {} settings: target_cycles={}, force_ribbon={}", epoch, epoch_target_cycles, force_ribbon);

    for (std::uint32_t ribbon_size = 1; ribbon_size <= (std::uint32_t)config.device_config.grid_size.r; ribbon_size++)
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
                        *graph_solver_epoch_snapshot, op, ribbon_size, config, graph, epoch_target_cycles);
                    log_trace(
                        LogBalancer,
                        "RIBBON2: (epoch={}, op_index={}, ribbon={}) {} best grid: {}, cycles: {} ",
                        epoch,
                        op_index,
                        ribbon_size,
                        node->name(),
                        selected_op_model.grid_shape,
                        get_limiter_cycles(
                            selected_op_model, graph, config, &graph_solver_epoch_snapshot->get_selected_op_models()));
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
                    EpochSolution new_solution(
                        epoch, ribbon_size, &config, selected_models, graph, epoch_target_cycles);

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
                LogBalancer, "Encountered BalancerException while trying ribbon size {}: {}", ribbon_size, e.what());

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
    EpochSolution best_solution = solutions[0];
    for (const EpochSolution &solution : solutions)
    {
        try
        {
            if (max_conservative_opt_iterations > 0)
            {
                EpochSolution optimized_solution = optimize_solution_conservative(
                    solution, *graph_solver_main, interactive_placer, graph, max_conservative_opt_iterations);
                if (optimized_solution.get_score() > best_solution.get_score())
                {
                    best_solution = optimized_solution;
                }
            }
            else
            {
                // Fallback to non-optimized.
                //
                if (solution.get_score() > best_solution.get_score())
                {
                    best_solution = solution;
                }
            }
        }
        catch (const BalancerError &e)
        {
            log_debug(LogBalancer, "Encountered BalancerException while optimizing solution: {}", e.what());
            // Use the unoptimized solution
            if (solution.get_score() > best_solution.get_score())
            {
                best_solution = solution;
            }
        }
    }

    return best_solution;
}

BalancerPolicySolution run_policy_ribbon(
    graphlib::Graph const *graph, const BalancerConfig &config, legalizer::GraphSolver &graph_solver)
{
    //
    // Ribbon policy
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

    log_info(LogBalancer, "Starting Ribbon balancing");
    BalancerPolicySolution balancer_policy_solution;
    placer::InteractivePlacer interactive_placer(graph, config);
    placer::InteractivePlacer ip_fittment_tester(graph, config);
    std::unordered_set<string> epoch_break_ops;
    scheduler::Schedule scheduled_ops;
    graphlib::NodeEpochType current_epoch_type = NodeEpochType::Forward;
    std::vector<const tt::graphlib::Node *> pre_buffering_epoch_nodes;
    std::unordered_set<const tt::graphlib::Node *> processed_nodes;
    std::vector<tt::scheduler::Schedule> op_names_to_epoch_break = config.op_names_to_epoch_break;
    tt::scheduler::Schedule processed_schedule;
    std::vector<EpochSolution> applied_solutions;

    std::unique_ptr<legalizer::GraphSolver> graph_solver_main = std::make_unique<legalizer::GraphSolver>(graph_solver);
    std::unique_ptr<graphlib::GraphTraversalContext> traversal_context =
        graph_solver_main->get_graph_traversal_context();
    std::tie(scheduled_ops, epoch_break_ops) =
        policy_run_scheduler(graph, config, processed_nodes, processed_schedule, op_names_to_epoch_break);

    TT_ASSERT(config.op_names_to_chip_break.size() == 0, "Ribbon2 policy does not process chip breaks");

    std::uint32_t epoch = 0;
    bool done = false;
    std::uint32_t placed_op_index = 0;
    std::uint32_t nodes_to_process = scheduled_ops.size();
    bool epoch_breaks_added = false;

    graph_solver_main->invalidate_suboptimal_op_models(
        legalizer::MatmulSparseDenseGridPairing | legalizer::DenseMatmulPrologue | legalizer::DenseMatmulBetterUkt);

    while (!done)
    {
        bool epoch_complete = false;
        InsertionInstructionMap prev_inst;
        std::unordered_set<graphlib::NodeId>
            fork_and_join_nodes;  // fork and join nodes of every nop-buffered fork-join in current epoch.

        // Save the snapshot of the graph solver before the new epoch is created.
        // This will be used to revert the graph solver in case we need to modify the graph
        // and solve the epoch again.
        std::unique_ptr<legalizer::GraphSolver> graph_solver_snapshot =
            std::make_unique<legalizer::GraphSolver>(*graph_solver_main);

        std::unordered_set<graphlib::NodeId> pre_buffered_ops;  // Original solution for the epoch - before we applied
                                                                // fork-join buffering.

        while (!epoch_complete)
        {
            EpochSolution best_solution = find_solution_for_epoch(
                graph_solver_main,
                graph,
                config,
                interactive_placer,
                ip_fittment_tester,
                scheduled_ops,
                epoch_break_ops,
                current_epoch_type,
                placed_op_index,
                epoch);

            bool rescheduled = handle_fork_join_nop_overflow(
                graph,
                config,
                op_names_to_epoch_break,
                best_solution,
                pre_buffered_ops,
                graph_solver_main,
                graph_solver_snapshot,
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
                prev_inst.clear();
                continue;
            }

            pre_buffered_ops.clear();
            for (const auto &op_model : best_solution.get_selected_op_models())
            {
                // Keep track of non-buffering ops only.
                if (!op_model.buda_op_node->is_buffering_op())
                {
                    pre_buffered_ops.insert(op_model.buda_op_node->id());
                }
            }

            // Try to apply the solution. The solution won't be applied in case we need to modify the graph
            // for fork-join buffering (insert nops/queues).
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
                applied_solutions.push_back(best_solution);
                log_debug(
                    LogBalancer,
                    "RIBBON2: (epoch={} target_cycles={}) applied solution with score: {} ribbon_size: {} "
                    "pipeline_cycles: "
                    "{} dram_access: {} pcie_access: {}",
                    epoch,
                    best_solution.get_epoch_target_cycles(),
                    best_solution.get_score(),
                    best_solution.get_ribbon_size(),
                    best_solution.get_pipeline_cycles(),
                    best_solution.get_dram_access_core_count(),
                    best_solution.get_pcie_access_core_count());

                if (placed_op_index >= scheduled_ops.size())
                {
                    log_info(LogBalancer, "Balancing 100% completed!");
                    done = true;
                    break;
                }

                log_info(LogBalancer, "Balancing {}% complete.", processed_nodes.size() * 100 / nodes_to_process);

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
                        config.device_config,
                        op_names_to_epoch_break,
                        config.op_names_to_chip_break,
                        scheduled_ops,
                        graph,
                        true /* use_interactive_placer */);
                    epoch_breaks_added = false;
                }

                prev_inst.clear();
                epoch_complete = true;
            }
        }
    }

    balancer_policy_solution.placer_solution = interactive_placer.commit();
    balancer_policy_solution.placer_solution.value().fork_join_buffered = true;
    validate_solution(scheduled_ops, balancer_policy_solution.placer_solution.value());
    balancer_policy_solution.balancer_score = score_solution(applied_solutions, config.device_config);
    balancer_policy_solution.graph_solver_solution = graph_solver_main->finish();
    return balancer_policy_solution;
}
}  // namespace tt::balancer
