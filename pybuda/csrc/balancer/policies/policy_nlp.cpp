// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policy_nlp.hpp"

#include <cstdint>
#include <cstdlib>

#include "balancer/policies/policy_utils.hpp"
#include "graph_lib/node_types.hpp"
#include "utils/logger.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using NodeType = tt::graphlib::NodeType;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;

namespace tt::balancer
{

// Return balancing target_cycles, i.e. the number of cycles all ops should be below, but as close as possible.
std::uint32_t calculate_target_cycles(
    graphlib::Graph const* graph, legalizer::GraphSolver& graph_solver, std::string const& arch_name)
{
    auto topo_sort = tt::graphlib::topological_sort(*graph);

    // Get min prologue volume that fits for each parameter
    std::unordered_map<Node*, std::uint32_t> min_param_grid_volume =
        find_min_prologue_volumes(graph, topo_sort, graph_solver);

    return get_matmul_target_cycles(graph, topo_sort, graph_solver, min_param_grid_volume, arch_name);
}

legalizer::GraphSolverSolution run_policy_nlp(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    legalizer::GraphSolver& graph_solver,
    std::uint32_t target_cycles)
{
    (void)config;
    log_debug(LogBalancer, "Starting NLP balancing.");
    std::vector<tt::graphlib::Node*> topo_sort = tt::graphlib::topological_sort(*graph);

    // Get min prologue volume that fits for each parameter
    std::unordered_map<Node*, std::uint32_t> min_param_grid_volume =
        find_min_prologue_volumes(graph, topo_sort, graph_solver);

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
                graph, topo_sort, graph_solver, min_param_grid_volume, config.device_config.arch_name);
        }

        // In case of recompile, we can offset the target cycles to get a different solution.
        target_cycles += config.target_cycles_offset;
    }

    bool skip_small_ukt = env_as<bool>("PYBUDA_SKIP_SMALL_UKT", false);

    // Pick OpModel for each node.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() != NodeType::kBudaOp)
            continue;

        const graphlib::BudaOpNode* op = node->as<graphlib::BudaOpNode>();
        std::string op_type = op->op_type().op;
        std::uint32_t min_prologue_volume = 0;  // min volume needed to remain prologue after other checks
        auto it = min_param_grid_volume.end();
        if (graph->data_operands(node).size() > 1)
            it = min_param_grid_volume.find(graph->data_operands(node)[1]);

        if (it != min_param_grid_volume.end())
            min_prologue_volume = it->second;

        // Find the actual smallest grid, with matching target rows, if possible
        auto op_models = graph_solver.at(node);
        const OpShape& op_shape = (*graph_solver.at(node).begin()).op_shape;
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
        for (auto op_model : op_models)
        {
            std::uint32_t execution_cycles = op_model.get_execution_cycles(config.device_config.arch_name);

            if ((op_type == "matmul") && skip_small_ukt)
            {
                balancer::BlockShape input0_block_shape = op_model.input_buffers[0].block_shape;
                int u_kt = input0_block_shape.ublock.ct;
                int m_k = op_model.op_shape.inputs[0].ct / input0_block_shape.ublock.ct;

                if ((u_kt < 4) && (m_k > 1))
                    continue;  // Skip bad u_kt settings. TODO: have a second pass that disables this if nothing is
                               // found
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
            else if (execution_cycles > closest_distance[category].first)
            {
                closest_distance[category] = current_test_pick;
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

        set_op_model_for_node(graph_solver, node, selected_op_model, config.device_config.arch_name);
    }

    return graph_solver.finish();
}

}  // namespace tt::balancer
