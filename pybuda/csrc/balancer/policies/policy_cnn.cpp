// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policy_cnn.hpp"

#include <cstdint>
#include <cstdlib>

#include "balancer/balancer.hpp"
#include "passes/fuse_ops.hpp"
#include "utils/logger.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using NodeType = tt::graphlib::NodeType;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;

namespace tt::balancer
{

namespace
{
std::unordered_map<Node*, std::uint32_t> find_min_prologue_volumes(
    Graph const* graph, const std::vector<Node*>& topo_sort, legalizer::GraphSolver& graph_solver)
{
    // Until back-end can reblock on prologue, we have to ensure that shared parameters (i.e. parameters read by
    // multiple matmuls) are reads by ops with the same grid sizes. Otherwise, the queue will have to be blocked for the
    // smaller grid, and then the op that needs a bigger grid will no longer fit in L1

    std::unordered_map<Node*, std::uint32_t> min_param_grid_volume;

    // Search for matmul parameters
    for (Node* node : topo_sort)
    {
        if ((node->node_type() != NodeType::kBudaOp) || (node->as<graphlib::BudaOpNode>()->op_type().op != "matmul"))
            continue;

        // Find minimum valid for prologue
        auto grids = graph_solver.at(node);
        bool found_prologue = false;
        std::uint32_t min_volume = 100000;
        for (auto grid : grids)
        {
            bool has_prologue = grid.parameter_buffers[1];
            std::uint32_t volume = grid.grid_shape.volume();
            if (has_prologue && (!found_prologue || (min_volume > volume)))
            {
                min_volume = volume;
                found_prologue = true;
            }
        }
        Node* param_node = graph->data_operands(node)[1];
        auto it = min_param_grid_volume.find(param_node);
        // Record max of all the min volumes
        if (found_prologue && ((it == min_param_grid_volume.end()) || (it->second < min_volume)))
        {
            min_param_grid_volume[param_node] = min_volume;
            log_debug(
                LogBalancer,
                "Setting minimum prologue volume on {} to {} due to {}",
                param_node->name(),
                min_volume,
                node->name());
            found_prologue = true;
        }
    }

    return min_param_grid_volume;
}
}  // namespace

BalancerPolicySolution run_policy_cnn(
    graphlib::Graph const* graph, BalancerConfig const& config, legalizer::GraphSolver& graph_solver, std::uint32_t)
{
    log_debug(LogBalancer, "Starting CNN balancing");
    auto topo_sort = tt::graphlib::topological_sort(*graph);

    // Get min prologue volume that fits for each parameter
    std::unordered_map<Node*, std::uint32_t> min_param_grid_volume =
        find_min_prologue_volumes(graph, topo_sort, graph_solver);

    // Pick a grid for each node.
    for (Node* node : topo_sort)
    {
        if (node->node_type() != NodeType::kBudaOp)
            continue;

        std::string op_type = node->as<graphlib::BudaOpNode>()->op_type().op;
        bool conv_matmul = (op_type == "matmul") && !node->as<graphlib::BudaOpNode>()->is_sparse_matmul();
        bool sparse_matmul = (op_type == "matmul") && node->as<graphlib::BudaOpNode>()->is_sparse_matmul();
        std::uint32_t min_prologue_volume = 0;  // min volume needed to remain prologue after other checks
        auto it = min_param_grid_volume.end();

        if (conv_matmul)
            it = min_param_grid_volume.find(graph->data_operands(node)[1]);
        else if (sparse_matmul)
            it = min_param_grid_volume.find(graph->data_operands(node)[2]);

        if (it != min_param_grid_volume.end())
            min_prologue_volume = it->second;

        // Find the largest row grid that works
        auto grids = graph_solver.at(node);
        std::uint32_t target_rows = 0;
        for (auto grid : grids)
        {
            // std::cout << "Looking for max row for " << node->name() << ": " << grid << std::endl;
            if ((std::uint32_t)grid.grid_shape.r > target_rows)
                target_rows = grid.grid_shape.r;
        }

        TT_ASSERT(target_rows > 0);

        using pick = std::pair<std::uint32_t, OpModel>;
        std::unordered_map<std::string, pick> closest_distance;
        pick default_pick = {0, *grids.begin()};
        closest_distance["best"] = default_pick;
        closest_distance["failed_prologue"] = default_pick;
        closest_distance["bad_rows"] = default_pick;
        closest_distance["bad_rows_failed_prologue"] = default_pick;
        closest_distance["too_slow"] = default_pick;
        closest_distance["too_slow_failed_prologue"] = default_pick;
        for (auto grid : grids)
        {
            std::uint32_t execution_cycles = grid.get_execution_cycles(config.device_config.arch_name);
            log_trace(
                LogBalancer,
                "Policy CNN considering {}: {}", 
                node->name(),
                grid);

            pick current_test_pick = {execution_cycles, grid};

            bool needs_prologue = (op_type == "matmul");  // others don't really matter, prologues are tiny
            bool has_prologue = false;
            if (needs_prologue)
            {
                if (node->as<graphlib::BudaOpNode>()->is_sparse_matmul())
                {
                    TT_ASSERT(grid.parameter_buffers.size() == 3);
                    has_prologue = grid.parameter_buffers[0] && grid.parameter_buffers[2];

                    if (grid.grid_shape.volume() > 40)
                        continue;  // TODO: this cases a pipegen error of too many DRAM readers
                }
                else
                {
                    TT_ASSERT(grid.parameter_buffers.size() > 1);
                    has_prologue = grid.parameter_buffers[1];
                }
            }

            bool prologue_ok =
                !needs_prologue || (has_prologue && ((std::uint32_t)grid.grid_shape.volume() >= min_prologue_volume));

            // Check and save the pick if it's better, in the right category
            // clang-format off
            std::string category =
                ((std::uint32_t)grid.grid_shape.r == target_rows)
                    ? prologue_ok ? "best" : "failed_prologue"
                    : prologue_ok ? "bad_rows" : "bad_rows_failed_prologue";
            // clang-format on

            /*if (execution_cycles > target_cycles)
            {
                // Invalid, unless we really have nothing else, in which case we'll pick the fastest
                category = prologue_ok ? "too_slow" : "too_slow_failed_prologue";
                if ( (execution_cycles < closest_distance[category].first) || (closest_distance[category].first == 0))
                    closest_distance[category] = current_test_pick;
            }
            else*/
            if (execution_cycles > closest_distance[category].first)
            {
                closest_distance[category] = current_test_pick;
            }

            log_trace(
                LogBalancer,
                "  Node {} grid {}: cat={}, cycles={}, closest_distance for category={}",
                node->name(),
                grid.grid_shape,
                category,
                execution_cycles,
                closest_distance[category].first);
        }

        // Pick the grid. TODO: failed prologue is not always worse than prologue - it only is now where dram access is
        // too slow to be useful If we model the cycles with dram access accurately, we could pick no-prologue as the
        // best choice
        auto picked_grid = *grids.begin();
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
                picked_grid = closest_distance[category].second;
                break;
            }
        }

        graph_solver.set(node, picked_grid);
        log_debug(
            LogBalancer,
            "Selected grid for node {} is {}, {}, cycles {}",
            node->name(),
            picked_grid.grid_shape,
            picked_grid.t_stream_factor,
            picked_grid.get_execution_cycles(config.device_config.arch_name));
    }

    return BalancerPolicySolution(graph_solver.finish());
}

}  // namespace tt::balancer

