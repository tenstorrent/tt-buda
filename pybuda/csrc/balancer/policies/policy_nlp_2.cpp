// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstdlib>

#include "balancer/policies/policy_utils.hpp"
#include "balancer/policies/policy_manager.hpp"
#include "balancer/policies/policy_nlp.hpp"
#include "graph_lib/node_types.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using NodeType = tt::graphlib::NodeType;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;

namespace tt::balancer
{

bool is_small_ukt(OpModel op_model)
{
    balancer::BlockShape input0_block_shape = op_model.input_buffers[0].block_shape;
    int u_kt = input0_block_shape.ublock.ct;
    int m_k = op_model.op_shape.inputs[0].ct / input0_block_shape.ublock.ct;

    if ((u_kt < 4) && (m_k > 1))
        return true;

    return false;
}

bool is_small_grid_size(OpModel op_model, int limit_r, int limit_c)
{
    if (op_model.grid_shape.r <= limit_r and op_model.grid_shape.c <= limit_c)
        return true;

    return false;
}

legalizer::GraphSolverSolution run_policy_nlp_v2(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    legalizer::GraphSolver& graph_solver,
    std::optional<placer::PlacerSolution>& placer_solution,
    std::uint32_t target_cycles)
{
    (void)config;
    log_debug(LogBalancer, "Starting NLP balancing.");
    log_debug(LogBalancer, "Using interactive placer.");

    PolicyManager policy_manager(graph, config, graph_solver);
    bool epoch_completed = false;
    std::vector<tt::graphlib::Node*> topo_sort = tt::graphlib::topological_sort(*graph);

    // Get min prologue volume that fits for each parameter
    std::unordered_map<Node*, std::uint32_t> min_param_grid_volume =
        find_min_prologue_volumes(graph, topo_sort, policy_manager);

    if (target_cycles == 0)
    {
        if (auto manual_target_cycles = env_as_optional<int>("PYBUDA_NLP_MANUAL_TARGET"))
        {
            target_cycles = *manual_target_cycles;
            log_info(LogBalancer, "Manual override of target cycles to {}", target_cycles);
        }
        else
        {
            target_cycles = get_matmul_target_cycles(
                graph, topo_sort, policy_manager, min_param_grid_volume, config.device_config.arch_name);
        }

        // In case of recompile, we can offset the target cycles to get a different solution.
        target_cycles += config.target_cycles_offset;
    }

    std::vector<int> target_cycles_per_subgraph = env_as_vector<int>("PYBUDA_NLP_MANUAL_TARGET_PER_SUBGRAPH");
    std::map<int, int> target_cycles_per_subgraph_map;
    int default_target_cycles = target_cycles;
    if (not target_cycles_per_subgraph.empty())
    {
        for (size_t i = 0; i < target_cycles_per_subgraph.size(); i += 2)
        {
            target_cycles_per_subgraph_map[target_cycles_per_subgraph[i]] = target_cycles_per_subgraph[i + 1];
        }
        log_info(LogBalancer, "Target cycles per subgraph: {}", target_cycles_per_subgraph_map);
    }

    bool skip_small_ukt = env_as<bool>("PYBUDA_SKIP_SMALL_UKT", false);
    std::vector<int> limit_grid_shape_per_subgraph = env_as_vector<int>("PYBUDA_LIMIT_GRID_SHAPE_PER_SUBGRAPH");
    bool skip_large_grid = false;
    if (limit_grid_shape_per_subgraph.size() == 0)
    {
        limit_grid_shape_per_subgraph = {0, 0, 0};
    }
    unsigned int subgraph_id = limit_grid_shape_per_subgraph[0];
    int limit_r = limit_grid_shape_per_subgraph[1];
    int limit_c = limit_grid_shape_per_subgraph[2];
    if (limit_r > 0 or limit_c > 0)
    {
        skip_large_grid = true;
    }

    // Pick OpModel for each node.
    //
    while (const graphlib::Node* node = policy_manager.get_next_op())
    {
        const graphlib::BudaOpNode* op = node->as<graphlib::BudaOpNode>();
        std::string op_type = op->op_type().op;
        std::uint32_t min_prologue_volume = 0;  // min volume needed to remain prologue after other checks
        auto it = min_param_grid_volume.end();
        if (graph->data_operands(node).size() > 1)
            it = min_param_grid_volume.find(graph->data_operands(node)[1]);

        if (it != min_param_grid_volume.end())
            min_prologue_volume = it->second;

        // Find the actual smallest grid, with matching target rows, if possible
        auto op_models = policy_manager.at(node);
        const OpShape& op_shape = (*policy_manager.at(node).begin()).op_shape;
        std::uint32_t target_rows = std::uint32_t(op_shape.outputs.at(0).rt / 6);
        if (target_rows == 0)
            target_rows = 1;

        using pick = std::pair<std::uint32_t, OpModel>;
        std::unordered_map<std::string, pick> closest_distance;
        pick default_pick = {0, *op_models.begin()};
        closest_distance["best"] = default_pick;
        closest_distance["failed_prologue"] = default_pick;
        closest_distance["bad_rows"] = default_pick;
        closest_distance["bad_rows_failed_prologue"] = default_pick;
        closest_distance["too_slow"] = default_pick;
        closest_distance["too_slow_failed_prologue"] = default_pick;

        bool skip_large_grid_for_subgraph = (skip_large_grid & (subgraph_id == graph->get_subgraph_id_for_node(node->id())));
        if (not target_cycles_per_subgraph_map.empty())
        {
            if (target_cycles_per_subgraph_map.find(graph->get_subgraph_id_for_node(node->id())) != target_cycles_per_subgraph_map.end())
            {
                target_cycles = target_cycles_per_subgraph_map[graph->get_subgraph_id_for_node(node->id())];
            }
            else
            {
                target_cycles = default_target_cycles;
            }
        }
        bool available_not_small_ukt = false;
        bool available_small_grid = false;

        if (skip_large_grid_for_subgraph or skip_small_ukt)
        {
            for (auto op_model : op_models)
            {
                if (op_type == "matmul")
                {
                    if (not is_small_ukt(op_model))
                    {
                        available_not_small_ukt = true;
                    }
                }
                if (is_small_grid_size(op_model, limit_r, limit_c))
                {
                    available_small_grid = true;
                }
                if (available_not_small_ukt && available_small_grid)
                {
                    continue;
                }
            }
        }
        
        for (auto op_model : op_models)
        {
            std::uint32_t execution_cycles = op_model.get_execution_cycles(config.device_config.arch_name);

            if ((op_type == "matmul") && skip_small_ukt && available_not_small_ukt)
            {
                if (is_small_ukt(op_model))
                    continue;
            }
            if (available_small_grid and skip_large_grid_for_subgraph)
            {
                if (not is_small_grid_size(op_model, limit_r, limit_c))
                    continue;
            }

            pick current_test_pick = {execution_cycles, op_model};

            bool needs_prologue = (op_type == "matmul") &&  // others don't really matter, prologues are tiny
                                                            // it needs a prologue if there's either a dram or parameter
                                                            // buffer for the second operand
                                  (((op_model.parameter_buffers.size() > 1) && op_model.parameter_buffers[1]) ||
                                   ((op_model.dram_buffers.size() > 1) && op_model.dram_buffers[1]));

            // if ( (op_type == "matmul") && !needs_prologue)
            if ((op_type == "matmul") && node->as<graphlib::BudaOpNode>()->is_gradient_op())
            {
                // Matmul with two non-prologue operands, it's going to be slower than usual
                // execution_cycles *= 2;
            }

            bool has_prologue = false;

            if (needs_prologue)
            {
                if (node->as<graphlib::BudaOpNode>()->is_sparse_matmul())
                {
                    TT_ASSERT(op_model.parameter_buffers.size() == 3);
                    has_prologue = op_model.parameter_buffers[0] && op_model.parameter_buffers[2];
                }
                else
                {
                    TT_ASSERT(op_model.parameter_buffers.size() > 1);
                    has_prologue = op_model.parameter_buffers[1];
                }
            }

            bool prologue_ok = !needs_prologue ||
                               (has_prologue && ((std::uint32_t)op_model.grid_shape.volume() >= min_prologue_volume));

            // Check and save the pick if it's better, in the right category

            // Matching target rows - or target columns, in which case we can transpose the op on placement
            // For now, let's not transpose matmuls, that could get dangerous.
            bool matching_rows = ((std::uint32_t)op_model.grid_shape.r == target_rows) ||
                                 ((op_model.grid_shape.c < op_model.grid_shape.r) &&
                                  ((std::uint32_t)op_model.grid_shape.c == target_rows) && (op_type != "matmul"));

            // clang-format off
            std::string category =
                matching_rows ?
                      prologue_ok ? "best" : "failed_prologue"
                    : prologue_ok ? "bad_rows" : "bad_rows_failed_prologue";
            // clang-format on

            if (execution_cycles > target_cycles)
            {
                // Invalid, unless we really have nothing else, in which case we'll pick the fastest
                category = prologue_ok ? "too_slow" : "too_slow_failed_prologue";
                if ((execution_cycles < closest_distance[category].first) || (closest_distance[category].first == 0))
                    closest_distance[category] = current_test_pick;
            }
            else
            {
                // Check if we are close to target cycles, and if so, base pick preference on other attributes.
                // Currently try to pick biggest m block. We may extend this logic in future.
                //
                if (close_to_target(closest_distance[category].first, target_cycles))
                {
                    if (close_to_target(execution_cycles, target_cycles))
                    {
                        if (op_model.block_shape().volume_no_t() > closest_distance[category].second.block_shape().volume_no_t())
                            closest_distance[category] = current_test_pick;
                    }
                }
                else if (execution_cycles > closest_distance[category].first)
                {
                    closest_distance[category] = current_test_pick;
                }
            }

            log_trace(
                LogBalancer,
                "  Node {} grid {}: cat={}, cycles={}, closest_distance for category={}",
                node->name(),
                op_model.grid_shape,
                category,
                execution_cycles,
                closest_distance[category].first);
        }

        // Pick the grid. TODO: failed prologue is not always worse than prologue - it only is now where dram access is
        // too slow to be useful If we model the cycles with dram access accurately, we could pick no-prologue as the
        // best choice
        auto selected_op_model = *op_models.begin();
        for (std::string category : std::vector<std::string>{
                 "best",
                 "bad_rows",
                 "too_slow",
                 "failed_prologue",
                 "bad_rows_failed_prologue",
                 "too_slow_failed_prologue"})
        {
            if (closest_distance[category].first != 0)
            {
                selected_op_model = closest_distance[category].second;
                break;
            }
        }

        std::tie(std::ignore, epoch_completed, std::ignore) = policy_manager.commit_op(selected_op_model);

        // If we're done with the epoch, finish it.
        //
        if (epoch_completed)
        {
            policy_manager.finish_current_epoch();
        }
    }

    placer_solution = policy_manager.commit_solution();

    return policy_manager.finish();
}

}  // namespace tt::balancer
